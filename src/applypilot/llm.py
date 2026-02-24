"""
Unified LLM client for ApplyPilot.

Auto-detects provider from environment:
  GEMINI_API_KEY  -> Google Gemini (default: gemma-3-27b-it, tiered pool)
  OPENAI_API_KEY  -> OpenAI (default: gpt-4o-mini)
  LLM_URL         -> Local llama.cpp / Ollama compatible endpoint

LLM_MODEL env var overrides the model name for any provider.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider() -> tuple[str, str, str]:
    """Return (base_url, model, api_key) based on environment variables.

    Reads env at call time (not module import time) so that load_env() called
    in _bootstrap() is always visible here.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    local_url = os.environ.get("LLM_URL", "")
    model_override = os.environ.get("LLM_MODEL", "")

    if gemini_key and not local_url:
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            model_override or "gemma-3-27b-it",
            gemini_key,
        )

    if openai_key and not local_url:
        return (
            "https://api.openai.com/v1",
            model_override or "gpt-4o-mini",
            openai_key,
        )

    if local_url:
        return (
            local_url.rstrip("/"),
            model_override or "local-model",
            os.environ.get("LLM_API_KEY", ""),
        )

    raise RuntimeError(
        "No LLM provider configured. "
        "Set GEMINI_API_KEY, OPENAI_API_KEY, or LLM_URL in your environment."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds

# Base wait on first 429/503 (doubles each retry, caps at 60s).
# Gemini free tier is 15 RPM = 4s minimum between requests; 10s gives headroom.
_RATE_LIMIT_BASE_WAIT = 10


_GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Gemma free tier: 30 RPM.  We allow 25 to leave a buffer.
_GEMINI_RPM = int(os.environ.get("GEMINI_RPM", "25"))


class _TokenBucket:
    """Simple token-bucket rate limiter (thread-safe)."""

    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(rpm, 1)  # seconds between tokens
        self._lock = threading.Lock()
        self._last = time.monotonic()  # start hot — first call waits too

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            deadline = self._last + self._interval
            if now < deadline:
                sleep_for = deadline - now
                log.debug("Rate limiter: sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
            self._last = time.monotonic()

    def backoff(self) -> None:
        """Called after a 429 — push _last forward so next wait() sleeps longer."""
        with self._lock:
            self._last = time.monotonic() + self._interval * 2


class _TPMTracker:
    """Sliding-window tokens-per-minute tracker (thread-safe).

    Estimates input tokens before sending (~4 chars/token) and reserves
    budget.  If the window is full, sleeps until enough old entries expire.
    Uses 80% of the stated limit as the effective ceiling to leave headroom
    for completion tokens and estimation error.
    """

    def __init__(self, tpm_limit: int) -> None:
        self._limit = int(tpm_limit * 0.80)
        self._lock = threading.Lock()
        self._window: list[tuple[float, int]] = []

    @staticmethod
    def estimate_tokens(messages: list[dict]) -> int:
        """Rough token estimate: ~4 chars per token."""
        chars = sum(len(m.get("content", "")) for m in messages)
        return max(chars // 4, 1)

    def wait_and_reserve(self, tokens: int) -> None:
        """Block until there's TPM budget, then reserve tokens."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._window = [(t, n) for t, n in self._window if now - t < 60]
                used = sum(n for _, n in self._window)
                if used + tokens <= self._limit:
                    self._window.append((now, tokens))
                    return
                # Find when enough budget frees up
                sleep_for = 60.0 - (now - self._window[0][0]) + 0.5
            # Sleep outside the lock so other threads aren't blocked
            log.info("TPM limiter: %d/%d used, sleeping %.1fs", used, self._limit, sleep_for)
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Tiered model pool (Gemma free-tier degradation)
# ---------------------------------------------------------------------------

_PACIFIC = timezone(timedelta(hours=-8))


def _today_pacific() -> str:
    return datetime.now(_PACIFIC).strftime("%Y-%m-%d")


@dataclass
class _ModelSlot:
    """One model in the tiered pool, with its own rate limiter and counters."""

    name: str
    rpm: int
    tpm: int
    daily_limit: int
    requests_today: int = 0
    exhausted: bool = False
    last_reset: str = ""
    consecutive_429s: int = 0
    rate_limiter: _TokenBucket | None = field(default=None, repr=False)
    tpm_tracker: _TPMTracker | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.rate_limiter is None:
            self.rate_limiter = _TokenBucket(self.rpm)
        if self.tpm_tracker is None:
            self.tpm_tracker = _TPMTracker(self.tpm)
        if not self.last_reset:
            self.last_reset = _today_pacific()


class _ModelPool:
    """Tiered model pool with degradation on 429s.

    Models are stored in priority order (best quality first).
    ``current_model()`` returns the highest tier that isn't exhausted.
    When a model hits 429, ``mark_exhausted()`` flags it and subsequent
    calls return the next tier down.
    """

    def __init__(self, slots: list[_ModelSlot]) -> None:
        self._slots = slots
        self._lock = threading.Lock()
        self._index = 0

    def current_model(self) -> _ModelSlot | None:
        """Return the highest-priority non-exhausted slot, or None."""
        with self._lock:
            self._check_daily_reset()
            for slot in self._slots:
                if not slot.exhausted:
                    return slot
            return None

    def next_model(self) -> _ModelSlot | None:
        """Round-robin across non-exhausted models (for multi-worker throughput)."""
        with self._lock:
            self._check_daily_reset()
            available = [s for s in self._slots if not s.exhausted]
            if not available:
                return None
            slot = available[self._index % len(available)]
            self._index += 1
            return slot

    def mark_exhausted(self, name: str) -> None:
        """Flag a model as exhausted (hit 429 or timeout)."""
        with self._lock:
            for slot in self._slots:
                if slot.name == name:
                    slot.exhausted = True
                    log.info("Model pool: %s exhausted, degrading to next tier.", name)
                    break

    def record_success(self, name: str) -> None:
        """Increment daily counter; auto-exhaust when daily limit reached."""
        with self._lock:
            for slot in self._slots:
                if slot.name == name:
                    slot.requests_today += 1
                    if slot.requests_today >= slot.daily_limit:
                        slot.exhausted = True
                        log.info(
                            "Model pool: %s hit daily limit (%d), degrading.",
                            name,
                            slot.daily_limit,
                        )
                    break

    def _check_daily_reset(self) -> None:
        """Reset exhausted flags and counters at midnight Pacific."""
        today = _today_pacific()
        for slot in self._slots:
            if slot.last_reset != today:
                slot.exhausted = False
                slot.requests_today = 0
                slot.last_reset = today
                log.info("Model pool: daily reset for %s.", slot.name)


_DEFAULT_GEMMA_TIERS = [
    {"name": "gemma-3-27b-it", "rpm": 30, "tpm": 15000, "daily_limit": 14400},
    {"name": "gemma-3-12b-it", "rpm": 30, "tpm": 15000, "daily_limit": 14400},
    {"name": "gemma-3-4b-it", "rpm": 30, "tpm": 15000, "daily_limit": 14400},
]


def _build_model_pool() -> _ModelPool:
    """Build the default Gemma tiered pool.

    Override model list via ``GEMINI_MODELS`` env var (comma-separated,
    ordered best-to-worst).
    """
    override = os.environ.get("GEMINI_MODELS", "")
    if override:
        names = [n.strip() for n in override.split(",") if n.strip()]
        slots = [
            _ModelSlot(name=n, rpm=30, tpm=15000, daily_limit=14400) for n in names
        ]
    else:
        slots = [_ModelSlot(**cfg) for cfg in _DEFAULT_GEMMA_TIERS]
    return _ModelPool(slots)


class LLMClient:
    """Thin LLM client supporting OpenAI-compatible and native Gemini endpoints.

    For Gemini keys, starts on the OpenAI-compat layer. On a 403 (which
    happens with preview/experimental models not exposed via compat), it
    automatically switches to the native generateContent API and stays there
    for the lifetime of the process.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        model_pool: _ModelPool | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=_TIMEOUT)
        self._use_native_gemini: bool = False
        self._is_gemini: bool = base_url.startswith(_GEMINI_COMPAT_BASE)
        self._model_pool: _ModelPool | None = model_pool
        # Skip single-model rate limiter when pool manages per-model limiters
        self._rate_limiter: _TokenBucket | None = (
            _TokenBucket(_GEMINI_RPM) if self._is_gemini and not model_pool else None
        )
        # Anthropic fallback: if both GEMINI and ANTHROPIC keys are set,
        # use Gemini first but fall back to Claude on 429 rate limits.
        self._anthropic_fallback_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._anthropic_fallback_model = os.environ.get("ANTHROPIC_FALLBACK_MODEL", "claude-haiku-4-5-20251001")
        self._using_fallback: bool = False

    def _chat_anthropic_fallback(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Anthropic as a fallback when primary provider is rate-limited."""
        system_text = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg.get("content", "") + "\n"
            else:
                api_messages.append({"role": msg["role"], "content": msg.get("content", "")})

        payload: dict[str, Any] = {
            "model": self._anthropic_fallback_model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()

        resp = self._client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": self._anthropic_fallback_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    # -- Native Gemini API --------------------------------------------------

    def _chat_native_gemini(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        *,
        model_override: str | None = None,
    ) -> str:
        """Call the native Gemini generateContent API.

        Used automatically when the OpenAI-compat endpoint returns 403,
        which happens for preview/experimental models not exposed via compat.

        Converts OpenAI-style messages to Gemini's contents/systemInstruction
        format transparently.
        """
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                contents.append({"role": "model", "parts": [{"text": text}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"{_GEMINI_NATIVE_BASE}/models/{model_override or self.model}:generateContent"
        resp = self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # -- OpenAI-compat API --------------------------------------------------

    @staticmethod
    def _fold_system_messages(messages: list[dict]) -> list[dict]:
        """Merge system messages into the first user message.

        Gemma models on the OpenAI-compat endpoint don't support the
        'system' role ("Developer instruction is not enabled").  This
        folds any system content into the first user message instead.
        """
        system_parts: list[str] = []
        other: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg.get("content", ""))
            else:
                other.append(msg)
        if not system_parts or not other:
            return messages
        # Prepend system text to the first user message
        combined = "\n\n".join(system_parts)
        first = other[0]
        other[0] = {
            "role": first["role"],
            "content": f"{combined}\n\n{first.get('content', '')}",
        }
        return other

    def _chat_compat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        *,
        model_override: str | None = None,
    ) -> str:
        """Call the OpenAI-compatible endpoint."""
        model = model_override or self.model

        # Gemma models don't support system role on compat endpoint
        if "gemma" in model.lower():
            messages = self._fold_system_messages(messages)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

        # 403 on Gemini compat = model not available on compat layer.
        # Raise a specific sentinel so chat() can switch to native API.
        if resp.status_code == 403 and self._is_gemini:
            raise _GeminiCompatForbidden(resp)

        return self._handle_compat_response(resp)

    @staticmethod
    def _handle_compat_response(resp: httpx.Response) -> str:
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message text."""
        # Qwen3 optimization: prepend /no_think to skip chain-of-thought
        # reasoning, saving tokens on structured extraction tasks.
        if "qwen" in self.model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        # Tiered pool path: degrade through models on 429, then Anthropic
        if self._model_pool:
            return self._chat_with_pool(messages, temperature, max_tokens)

        for attempt in range(_MAX_RETRIES):
            try:
                # Throttle Gemini calls to stay within free tier RPM
                if self._rate_limiter and not self._using_fallback:
                    self._rate_limiter.wait()

                # Route to native Gemini if we've already confirmed it's needed
                if self._use_native_gemini:
                    return self._chat_native_gemini(messages, temperature, max_tokens)

                return self._chat_compat(messages, temperature, max_tokens)

            except _GeminiCompatForbidden as exc:
                # Model not available on OpenAI-compat layer — switch to native.
                log.warning(
                    "Gemini compat endpoint returned 403 for model '%s'. "
                    "Switching to native generateContent API. "
                    "(Preview/experimental models are often compat-only on native.)",
                    self.model,
                )
                self._use_native_gemini = True
                # Retry immediately with native — don't count as a rate-limit wait
                try:
                    return self._chat_native_gemini(messages, temperature, max_tokens)
                except httpx.HTTPStatusError as native_exc:
                    raise RuntimeError(
                        f"Both Gemini endpoints failed. Compat: 403 Forbidden. "
                        f"Native: {native_exc.response.status_code} — "
                        f"{native_exc.response.text[:200]}"
                    ) from native_exc

            except httpx.HTTPStatusError as exc:
                resp = exc.response
                if resp.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                    # Rate limiter: one retry with backoff, then fall through
                    # to Anthropic if that also 429s (daily limit scenario).
                    if self._rate_limiter and attempt == 0:
                        self._rate_limiter.backoff()
                        log.info(
                            "Rate limited (HTTP %s) — one backoff retry before fallback.",
                            resp.status_code,
                        )
                        continue

                    # Try Anthropic fallback if available
                    if self._anthropic_fallback_key and not self._using_fallback:
                        log.warning(
                            "LLM rate limited (HTTP %s). Falling back to Anthropic (%s) for this request.",
                            resp.status_code, self._anthropic_fallback_model,
                        )
                        self._using_fallback = True
                        # Try Anthropic with up to 3 attempts and short backoff
                        for fb_attempt in range(3):
                            try:
                                if fb_attempt > 0:
                                    time.sleep(5 * fb_attempt)
                                result = self._chat_anthropic_fallback(messages, temperature, max_tokens)
                                self._using_fallback = False
                                return result
                            except httpx.HTTPStatusError as fb_exc:
                                if fb_exc.response.status_code == 429 and fb_attempt < 2:
                                    log.warning("Anthropic also rate limited, waiting %ds...", 5 * (fb_attempt + 1))
                                    continue
                                break
                            except Exception:
                                break
                        log.warning("Anthropic fallback exhausted — retrying primary.")
                        self._using_fallback = False
                        # Fall through to normal retry logic

                    # Normal retry with backoff
                    retry_after = (
                        resp.headers.get("Retry-After")
                        or resp.headers.get("X-RateLimit-Reset-Requests")
                    )
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except (ValueError, TypeError):
                            wait = _RATE_LIMIT_BASE_WAIT * (2 ** attempt)
                    else:
                        wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)

                    log.warning(
                        "LLM rate limited (HTTP %s). Waiting %ds before retry %d/%d.",
                        resp.status_code, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning(
                        "LLM request timed out, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError("LLM request failed after all retries")

    def _chat_with_pool(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Chat using tiered model pool with automatic degradation."""
        while True:
            slot = self._model_pool.next_model()  # type: ignore[union-attr]
            if slot is None:
                break  # All tiers exhausted → Anthropic fallback

            try:
                slot.rate_limiter.wait()  # type: ignore[union-attr]
                # TPM throttle: estimate input tokens and wait if budget is full
                est_tokens = _TPMTracker.estimate_tokens(messages) + max_tokens
                slot.tpm_tracker.wait_and_reserve(est_tokens)  # type: ignore[union-attr]

                if self._use_native_gemini:
                    result = self._chat_native_gemini(
                        messages, temperature, max_tokens, model_override=slot.name,
                    )
                else:
                    result = self._chat_compat(
                        messages, temperature, max_tokens, model_override=slot.name,
                    )

                slot.consecutive_429s = 0  # Reset on success
                self._model_pool.record_success(slot.name)  # type: ignore[union-attr]
                return result

            except _GeminiCompatForbidden:
                log.info("Compat 403 for %s — switching to native API.", slot.name)
                self._use_native_gemini = True
                continue  # Retry same model with native

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (429, 503):
                    slot.consecutive_429s += 1
                    if slot.consecutive_429s >= 3:
                        # Repeated 429s = likely daily limit, exhaust this model
                        log.warning(
                            "Model pool: %s hit %d consecutive 429s — exhausting.",
                            slot.name, slot.consecutive_429s,
                        )
                        self._model_pool.mark_exhausted(slot.name)  # type: ignore[union-attr]
                    else:
                        # Transient RPM 429 — backoff and let other workers use other models
                        slot.rate_limiter.backoff()  # type: ignore[union-attr]
                        log.info(
                            "Model pool: %s rate-limited (HTTP %s), backoff #%d.",
                            slot.name, exc.response.status_code, slot.consecutive_429s,
                        )
                    continue
                if exc.response.status_code == 400 and self._is_gemini:
                    # 400 on Gemma = payload too large (TPM) or unsupported
                    # — skip all pool models, go straight to Anthropic fallback
                    log.warning(
                        "Model pool: %s returned 400 — payload likely exceeds limits. "
                        "Falling back to Anthropic. Detail: %s",
                        slot.name, exc.response.text[:200],
                    )
                    break  # Exit while loop → Anthropic fallback
                raise

            except httpx.TimeoutException:
                log.warning("Model pool: %s timed out — degrading.", slot.name)
                self._model_pool.mark_exhausted(slot.name)  # type: ignore[union-attr]
                continue

        # All tiers exhausted → Anthropic fallback
        if self._anthropic_fallback_key:
            log.warning(
                "All pool tiers exhausted — Anthropic fallback (%s).",
                self._anthropic_fallback_model,
            )
            for fb_attempt in range(3):
                try:
                    if fb_attempt > 0:
                        time.sleep(5 * fb_attempt)
                    return self._chat_anthropic_fallback(
                        messages, temperature, max_tokens,
                    )
                except httpx.HTTPStatusError as fb_exc:
                    if fb_exc.response.status_code == 429 and fb_attempt < 2:
                        log.warning(
                            "Anthropic rate-limited, waiting %ds...",
                            5 * (fb_attempt + 1),
                        )
                        continue
                    raise

        raise RuntimeError(
            "All model pool tiers exhausted and no Anthropic fallback configured."
        )

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


class _GeminiCompatForbidden(Exception):
    """Sentinel: Gemini OpenAI-compat returned 403. Switch to native API."""
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Gemini compat 403: {response.text[:200]}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: LLMClient | None = None


def get_client() -> LLMClient:
    """Return (or create) the module-level LLMClient singleton."""
    global _instance
    if _instance is None:
        base_url, model, api_key = _detect_provider()

        # Build tiered model pool for Gemini when no explicit model override
        model_pool: _ModelPool | None = None
        if base_url.startswith(_GEMINI_COMPAT_BASE) and not os.environ.get("LLM_MODEL"):
            model_pool = _build_model_pool()
            model = model_pool._slots[0].name
            log.info(
                "LLM provider: Gemini (tiered pool: %s)",
                ", ".join(s.name for s in model_pool._slots),
            )
        else:
            log.info("LLM provider: %s  model: %s", base_url, model)

        _instance = LLMClient(base_url, model, api_key, model_pool=model_pool)
    return _instance


def get_worker_count() -> int:
    """Recommended concurrent workers: 1 per model slot, or 1 if no pool."""
    override = os.environ.get("LLM_WORKERS", "")
    if override:
        return max(1, int(override))
    client = get_client()
    if client._model_pool:
        return len(client._model_pool._slots)
    return 1
