"""Tests for how API-level failures are classified, retried and surfaced."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from autointent.generation import FatalAPIError, Generator, RetriesExceededError
from autointent.generation.chat_templates import Role

from .test_retries import VALID_PERSON_JSON, Person, _resp

if TYPE_CHECKING:
    from respx.router import MockRouter

    from autointent.generation.chat_templates import Message

MESSAGES: list[Message] = [{"role": Role.USER, "content": "ok"}]


@pytest.fixture
def generator(respx_openai: MockRouter) -> Generator:
    return Generator(max_tokens=1000, use_cache=False)


@pytest.fixture
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep backoff out of the test runtime while still exercising the retry path."""
    monkeypatch.setattr("autointent.generation._generator._INITIAL_RETRY_DELAY", 0.0)
    monkeypatch.setattr("autointent.generation._generator._MAX_RETRY_DELAY", 0.0)


@pytest.mark.parametrize("status_code", [401, 403, 404, 422])
@pytest.mark.usefixtures("_no_waiting")
def test_fatal_status_is_not_retried(generator: Generator, respx_openai: MockRouter, status_code: int) -> None:
    route = respx_openai.post("/v1/chat/completions").mock(return_value=httpx.Response(status_code, json={}))

    with pytest.raises(FatalAPIError) as exc_info:
        generator.get_structured_output_sync(messages=MESSAGES, output_model=Person, max_retries=3)

    assert exc_info.value.status_code == status_code
    assert str(status_code) in str(exc_info.value)
    # one call, not max_retries + 1: the openai client does not retry these either
    assert route.call_count == 1


@pytest.mark.usefixtures("_no_waiting")
def test_transient_status_is_retried_then_succeeds(generator: Generator, respx_openai: MockRouter) -> None:
    respx_openai.post("/v1/chat/completions").mock(side_effect=[httpx.Response(503, json={}), _resp(VALID_PERSON_JSON)])

    result = generator.get_structured_output_sync(messages=MESSAGES, output_model=Person, max_retries=3)

    assert isinstance(result, Person)


@pytest.mark.usefixtures("_no_waiting")
def test_transient_status_exhausts_retries(generator: Generator, respx_openai: MockRouter) -> None:
    respx_openai.post("/v1/chat/completions").mock(return_value=httpx.Response(503, json={}))

    # RetriesExceededError, not the raw APIStatusError: callers degrade this utterance and keep going
    with pytest.raises(RetriesExceededError):
        generator.get_structured_output_sync(messages=MESSAGES, output_model=Person, max_retries=1)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_waiting")
async def test_fatal_status_is_not_retried_async(generator: Generator, respx_openai: MockRouter) -> None:
    respx_openai.post("/v1/chat/completions").mock(return_value=httpx.Response(401, json={}))

    with pytest.raises(FatalAPIError):
        await generator.get_structured_output_async(messages=MESSAGES, output_model=Person, max_retries=3)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_waiting")
async def test_transient_status_is_retried_then_succeeds_async(generator: Generator, respx_openai: MockRouter) -> None:
    respx_openai.post("/v1/chat/completions").mock(side_effect=[httpx.Response(429, json={}), _resp(VALID_PERSON_JSON)])

    result = await generator.get_structured_output_async(messages=MESSAGES, output_model=Person, max_retries=3)

    assert isinstance(result, Person)


@pytest.mark.usefixtures("_no_waiting")
def test_connection_error_is_retried(generator: Generator, respx_openai: MockRouter) -> None:
    respx_openai.post("/v1/chat/completions").mock(side_effect=[httpx.ConnectError("boom"), _resp(VALID_PERSON_JSON)])

    result = generator.get_structured_output_sync(messages=MESSAGES, output_model=Person, max_retries=3)

    assert isinstance(result, Person)


def _api_status_error(status_code: int, headers: dict[str, str] | None = None) -> Exception:
    import openai

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, headers=headers or {}, request=request, json={})
    return openai.APIStatusError("boom", response=response, body=None)


def test_retry_delay_honours_retry_after_header() -> None:
    from autointent.generation._generator import _retry_delay

    assert _retry_delay(_api_status_error(429, {"retry-after": "7"}), attempt=0) == 7.0


def test_retry_delay_falls_back_to_backoff() -> None:
    from autointent.generation._generator import _retry_delay

    # no header, and an HTTP-date header that cannot be read as seconds
    assert _retry_delay(_api_status_error(429), attempt=2) == 4.0
    assert _retry_delay(_api_status_error(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), attempt=1) == 2.0


def test_retry_delay_is_capped() -> None:
    from autointent.generation._generator import _retry_delay

    assert _retry_delay(_api_status_error(429, {"retry-after": "3600"}), attempt=0) == 60.0
    assert _retry_delay(_api_status_error(503), attempt=20) == 60.0
