"""Call the Anthropic API for one extraction and report what it cost."""

import json
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic
from dotenv import load_dotenv

from src.extraction.prompt import ExtractionPrompt

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "low"
DEFAULT_MAX_TOKENS = 16000

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0

# Models that still accept a sampling temperature. Every newer model rejects the
# parameter with a 400, so for any model not listed here it is left out.
TEMPERATURE_MODEL_PREFIXES: tuple[str, ...] = (
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-opus-4-6",
)

# Failures worth retrying: rate limits, timeouts, dropped connections, 5xx/overloaded.
TRANSIENT_ERRORS: tuple[type[anthropic.APIError], ...] = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,  # includes APITimeoutError
    anthropic.InternalServerError,
)


@dataclass(frozen=True)
class ExtractionConfig:
    """Model settings for extraction. `None` means the parameter is not sent."""

    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str | None = DEFAULT_EFFORT
    temperature: float | None = None


@dataclass(frozen=True)
class Usage:
    """Raw cost data for one API call."""

    input_tokens: int
    output_tokens: int
    latency_seconds: float


@dataclass(frozen=True)
class ModelResponse:
    """The model's answer to one extraction request."""

    payload: dict[str, Any] | None
    """The response decoded from JSON; None if the model did not return a JSON object."""
    raw_text: str
    stop_reason: str | None
    usage: Usage
    model: str


def load_config() -> ExtractionConfig:
    """Read model settings from the environment (and .env), falling back to defaults.

    EXTRACTION_EFFORT and EXTRACTION_TEMPERATURE set to an empty string mean "do not
    send this parameter". A temperature is only sent to models that accept one; for
    other models it is ignored with a warning.
    """
    load_dotenv()
    effort = os.environ.get("EXTRACTION_EFFORT", DEFAULT_EFFORT)
    temperature = os.environ.get("EXTRACTION_TEMPERATURE", "")
    return ExtractionConfig(
        model=os.environ.get("EXTRACTION_MODEL") or DEFAULT_MODEL,
        effort=effort or None,
        temperature=float(temperature) if temperature else None,
    )


def make_client() -> anthropic.Anthropic:
    """Build an API client with the key from ANTHROPIC_API_KEY (environment or .env)."""
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; add it to .env (see .env.example)")
    # Retries are handled in request_extraction so they are counted and testable.
    return anthropic.Anthropic(api_key=api_key, max_retries=0)


def supports_temperature(model: str) -> bool:
    """Whether the API accepts a `temperature` for this model."""
    return model.startswith(TEMPERATURE_MODEL_PREFIXES)


def _request_params(prompt: ExtractionPrompt, config: ExtractionConfig) -> dict[str, Any]:
    output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": prompt.output_schema}}
    if config.effort is not None:
        output_config["effort"] = config.effort
    params: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": prompt.system,
        "messages": [{"role": "user", "content": prompt.user_message}],
        "output_config": output_config,
    }
    if config.temperature is not None:
        if supports_temperature(config.model):
            # anthropic 1.x has no `temperature` argument; the API still takes the field.
            params["extra_body"] = {"temperature": config.temperature}
        else:
            warnings.warn(
                f"EXTRACTION_TEMPERATURE={config.temperature} ignored: model {config.model!r} "
                "does not accept a sampling temperature. Use EXTRACTION_EFFORT to control it instead.",
                stacklevel=2,
            )
    return params


def _decode(raw_text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def request_extraction(
    client: anthropic.Anthropic,
    prompt: ExtractionPrompt,
    config: ExtractionConfig,
    sleep: Callable[[float], None] = time.sleep,
) -> ModelResponse:
    """Send one extraction request, constrained to the prompt's output schema.

    Transient API errors are retried up to MAX_ATTEMPTS times with exponential backoff;
    the last one is re-raised if every attempt fails. Other API errors (bad request,
    authentication) are raised immediately. `latency_seconds` covers the successful
    request only, not earlier attempts or backoff.
    """
    params = _request_params(prompt, config)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            message = client.messages.create(**params)
        except TRANSIENT_ERRORS:
            if attempt == MAX_ATTEMPTS:
                raise
            sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
            continue
        latency = time.perf_counter() - started
        raw_text = "".join(block.text for block in message.content if block.type == "text")
        return ModelResponse(
            payload=_decode(raw_text),
            raw_text=raw_text,
            stop_reason=message.stop_reason,
            usage=Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                latency_seconds=latency,
            ),
            model=message.model,
        )
    raise AssertionError("unreachable: the loop returns or raises")
