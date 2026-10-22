"""Wrapper class for accessing OpenAI API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from textwrap import dedent
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from autointent._deps import require
from autointent.generation.chat_templates import Message, Role

from ._cache import StructuredOutputCache

if TYPE_CHECKING:
    from pathlib import Path

    from autointent.generation.chat_templates import Message

logger = logging.getLogger(__name__)

load_dotenv()

T = TypeVar("T", bound=BaseModel)
"""Type variable for Pydantic models used in structured output generation."""


class GeneratorDumpData(TypedDict):
    use_cache: bool
    model_name: str
    base_url: str | None
    generation_params: dict[str, Any]


FATAL_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 405, 422})
"""HTTP statuses that retrying cannot fix: bad request, auth, permissions, missing model, no credits."""

_INITIAL_RETRY_DELAY = 1.0
_MAX_RETRY_DELAY = 60.0


class FatalAPIError(RuntimeError):
    """Exception raised when the API rejects a request in a way that retrying cannot fix.

    Raised instead of retrying so that a misconfigured key, model or exhausted budget
    fails immediately with a readable message rather than after spending more calls.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        """Initialize the error with the HTTP status and the message reported by the API.

        Args:
            status_code: HTTP status code returned by the API
            detail: Error message as reported by the API
        """
        msg = f"The API returned HTTP {status_code}, which retrying cannot fix: {detail}"
        super().__init__(msg)
        self.status_code = status_code


def _raise_if_fatal(error: Exception) -> None:
    """Re-raise an API error as :py:class:`FatalAPIError` if retrying it would be pointless.

    Args:
        error: Exception raised by the OpenAI client.

    Raises:
        FatalAPIError: If the API answered with one of :py:data:`FATAL_STATUS_CODES`.
    """
    from openai import APIStatusError

    if isinstance(error, APIStatusError) and error.status_code in FATAL_STATUS_CODES:
        raise FatalAPIError(error.status_code, str(error)) from error


def _retry_delay(error: Exception, attempt: int) -> float:
    """Seconds to wait before the next attempt, honouring ``Retry-After`` when the API sends it.

    Args:
        error: Exception raised by the OpenAI client.
        attempt: Zero-based index of the attempt that just failed.

    Returns:
        Delay in seconds, capped at one minute.
    """
    from openai import APIStatusError

    if isinstance(error, APIStatusError):
        retry_after = error.response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return min(float(retry_after), _MAX_RETRY_DELAY)
            except ValueError:
                # Retry-After may also be an HTTP date; fall back to the backoff below.
                logger.debug("Could not read Retry-After value %r as seconds", retry_after)
    return min(_INITIAL_RETRY_DELAY * 2.0**attempt, _MAX_RETRY_DELAY)


class RetriesExceededError(RuntimeError):
    """Exception raised when LLM call fails after all retry attempts."""

    def __init__(self, max_retries: int, messages: list[Message]) -> None:
        """Initialize the error with retry count and messages.

        Args:
            max_retries: Maximum number of retry attempts that were made
            messages: Messages that were sent to the LLM
        """
        msg = f"LLM call failed after {max_retries + 1} attempts. Messages: {messages}"
        super().__init__(msg)


class Generator:
    """Wrapper class for accessing OpenAI-compatible API endpoints for LLM generation.

    This class provides a unified interface for interacting with OpenAI-compatible APIs,
    supporting both synchronous and asynchronous operations. It includes built-in caching,
    retry logic for structured output, and automatic environment variable detection.

    The Generator can work with various OpenAI-compatible services including:
    - OpenAI's official API
    - Azure OpenAI
    - Local inference servers (vLLM, Ollama, etc.)
    - Other OpenAI-compatible endpoints

    Environment Variables:
        The following environment variables can be used for configuration:

        **OPENAI_API_KEY** (required):
            API key for authentication with the OpenAI-compatible service.
            This is required for most API endpoints.

        **OPENAI_BASE_URL** (optional):
            Base URL for the API endpoint. If not provided, defaults to OpenAI's API.
            - https://api.openai.com/v1 (OpenAI official)
            - https://your-org.openai.azure.com (Azure OpenAI)
            - http://localhost:8000/v1 (local vLLM server)

        **OPENAI_MODEL_NAME** (optional):
            Default model name to use if not specified in the constructor.
            Examples: "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-haiku"

    Args:
        base_url: HTTP endpoint for API requests. If None, uses OPENAI_BASE_URL environment variable.
        model_name: Name of the language model. If None, uses OPENAI_MODEL_NAME environment variable.
        use_cache: Whether to enable caching for structured outputs (default: True).
        client_params: Additional parameters passed to the OpenAI client constructor.
        **generation_params: Additional parameters passed to the chat completion API calls.

    Example:
    --------
    .. code-block::

        import os
        from autointent.generation import Generator

        # Method 1: Using environment variables
        # Set these in your environment or .env file:
        # OPENAI_API_KEY=your-api-key-here
        # OPENAI_MODEL_NAME=gpt-4o-mini
        # OPENAI_BASE_URL=https://api.openai.com/v1  # optional

        generator = Generator()

        # Method 2: Explicit configuration
        generator = Generator(
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000
        )

        # Basic chat completion
        from autointent.generation.chat_templates import Message, Role

        messages = [{"role": Role.USER, "content": "Hello, how are you?"}]
        response = generator.get_chat_completion(messages)

    Raises:
        ValueError: If model_name is not provided and OPENAI_MODEL_NAME is not set.
    """

    _dump_data_filename = "init_params.json"

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        use_cache: bool = True,
        client_params: dict[str, Any] | None = None,
        **generation_params: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the Generator with API configuration.

        Args:
            base_url: OpenAI API compatible server URL.
            model_name: Name of the language model to use.
            use_cache: Whether to use caching for structured outputs.
            client_params: Additional parameters for client.
            **generation_params: Additional generation parameters to override defaults passed to OpenAI completions API.
        """
        require("openai")
        import openai

        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        model_name = model_name or os.getenv("OPENAI_MODEL_NAME")

        if model_name is None:
            msg = "Specify model_name arg or OPENAI_MODEL_NAME environment variable"
            raise ValueError(msg)

        self.model_name = model_name
        self.base_url = base_url
        self.use_cache = use_cache

        self.client = openai.OpenAI(base_url=base_url, **(client_params or {}))
        self.async_client = openai.AsyncOpenAI(base_url=base_url, **(client_params or {}))
        self.generation_params = generation_params
        self.cache = StructuredOutputCache(use_cache=use_cache)

    def get_chat_completion(self, messages: list[Message]) -> str:
        """Prompt LLM and return its answer.

        Args:
            messages: List of messages to send to the model.
        """
        response = self.client.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]  # reason: list[Message] (TypedDict) structurally matches openai ChatCompletionMessageParam union but mypy can't verify
            model=self.model_name,
            **self.generation_params,
        )
        return response.choices[0].message.content  # type: ignore[return-value]  # reason: openai stub types content as str | None; we accept the runtime contract that content is set for non-streaming completions

    async def get_chat_completion_async(self, messages: list[Message]) -> str:
        """Prompt LLM and return its answer asynchronously.

        Args:
            messages: List of messages to send to the model.
        """
        response = await self.async_client.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]  # reason: list[Message] (TypedDict) structurally matches openai ChatCompletionMessageParam union but mypy can't verify
            model=self.model_name,
            **self.generation_params,
        )

        if response is None or not response.choices:
            msg = "No response received from the model."
            raise RuntimeError(msg)
        return response.choices[0].message.content  # type: ignore[return-value]  # reason: openai stub types content as str | None; we accept the runtime contract that content is set for non-streaming completions

    def _create_retry_messages(self, error_message: str, raw: str | None) -> list[Message]:
        """Create a follow-up message for retry with error details and schema."""
        res: list[Message] = []
        if raw is not None:
            res.append({"role": Role.ASSISTANT, "content": raw})
        res.append(
            {
                "role": Role.USER,
                "content": dedent(
                    f"""The previous response failed validation with the following error: {error_message}

                    Make sure to:
                    1. Follow the exact schema structure
                    2. Use the correct data types for each field
                    3. Include all required fields
                    4. Ensure the response is valid JSON"""
                ),
            }
        )
        return res

    async def _get_structured_output_openai_async(
        self, messages: list[Message], output_model: type[T]
    ) -> tuple[T | None, str | None, str | None]:
        """Get structured output using OpenAI's beta parse endpoint asynchronously.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.

        Returns:
            Tuple of (parsed_result, error_message, raw_response).
        """
        from openai import LengthFinishReasonError

        res: T | None = None
        msg: str | None = None
        raw: str | None = None

        try:
            response = await self.async_client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]  # reason: list[Message] (TypedDict) structurally matches openai ChatCompletionMessageParam union
                response_format=output_model,
                **self.generation_params,
            )
            raw = response.choices[0].message.content
            res = response.choices[0].message.parsed
        except (ValidationError, ValueError, LengthFinishReasonError) as e:
            msg = f"Failed to obtain structured output for model {self.model_name} and messages {messages}: {e!s}"
            logger.warning(msg)
        else:
            if res is None:
                msg = "For some reason output wasn't parsed."
                logger.warning(msg)

        return res, msg, raw

    async def get_structured_output_async(
        self,
        messages: list[Message],
        output_model: type[T],
        max_retries: int = 3,
    ) -> T:
        """Prompt LLM and return structured output parsed into the provided Pydantic model asynchronously.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            max_retries: Maximum number of retry attempts for failed validations.

        Returns:
            Parsed response as an instance of the provided Pydantic model.
        """
        # Check cache first
        cached_result = await self.cache.get_async(
            messages, output_model, self.generation_params, self.model_name, self.base_url
        )
        if cached_result is not None:
            return cached_result

        current_messages = messages.copy()
        res: T | None = None

        from openai import APIConnectionError, APIStatusError

        for attempt in range(max_retries + 1):
            try:
                res, error, raw = await self._get_structured_output_openai_async(current_messages, output_model)
            except (APIConnectionError, APIStatusError) as e:
                _raise_if_fatal(e)
                delay = _retry_delay(e, attempt)
                logger.warning("API call failed (%s), retrying in %.1fs", e, delay)
                await asyncio.sleep(delay)
                continue

            if res is not None:
                break

            if error is None:
                msg = "Structured output returned None but no error was caught."
                logger.exception(msg)
                raise RuntimeError(msg)

            current_messages.extend(self._create_retry_messages(error, raw))

        if res is None:
            msg = (
                f"Failed to generate valid structured output after {max_retries + 1} attempts.\n"
                f"Messages: {current_messages}"
            )
            logger.exception(msg)
            raise RetriesExceededError(max_retries=max_retries, messages=current_messages)

        # Cache the successful result
        await self.cache.set_async(messages, output_model, self.generation_params, res, self.model_name, self.base_url)

        return res

    def _get_structured_output_openai_sync(
        self, messages: list[Message], output_model: type[T]
    ) -> tuple[T | None, str | None, str | None]:
        """Get structured output using OpenAI's beta parse endpoint synchronously.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.

        Returns:
            Tuple of (parsed_result, error_message, raw_response).
        """
        from openai import LengthFinishReasonError

        res: T | None = None
        msg: str | None = None
        raw: str | None = None

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]  # reason: list[Message] (TypedDict) structurally matches openai ChatCompletionMessageParam union
                response_format=output_model,
                **self.generation_params,
            )
            raw = response.choices[0].message.content
            res = response.choices[0].message.parsed
        except (ValidationError, ValueError, LengthFinishReasonError) as e:
            msg = f"Failed to obtain structured output for model {self.model_name} and messages {messages}: {e!s}"
            logger.warning(msg)
        else:
            if res is None:
                msg = "For some reason output wasn't parsed."
                logger.warning(msg)

        return res, msg, raw

    def get_structured_output_sync(
        self,
        messages: list[Message],
        output_model: type[T],
        max_retries: int = 3,
    ) -> T:
        """Prompt LLM and return structured output parsed into the provided Pydantic model.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            max_retries: Maximum number of retry attempts for failed validations.

        Returns:
            Parsed response as an instance of the provided Pydantic model.
        """
        # Check cache first
        cached_result = self.cache.get(messages, output_model, self.generation_params, self.model_name, self.base_url)
        if cached_result is not None:
            return cached_result

        current_messages = messages.copy()
        res: T | None = None

        from openai import APIConnectionError, APIStatusError

        for attempt in range(max_retries + 1):
            try:
                res, error, raw = self._get_structured_output_openai_sync(current_messages, output_model)
            except (APIConnectionError, APIStatusError) as e:
                _raise_if_fatal(e)
                delay = _retry_delay(e, attempt)
                logger.warning("API call failed (%s), retrying in %.1fs", e, delay)
                time.sleep(delay)
                continue

            if res is not None:
                break

            if error is None:
                msg = "Structured output returned None but no error was caught."
                logger.exception(msg)
                raise RuntimeError(msg)

            current_messages.extend(self._create_retry_messages(error, raw))

        if res is None:
            msg = "Structured output returned None but no error was caught."
            logger.exception(msg)
            raise RetriesExceededError(max_retries=max_retries, messages=current_messages)

        # Cache the successful result
        self.cache.set(messages, output_model, self.generation_params, res, self.model_name, self.base_url)

        return res

    def dump(self, path: Path, exist_ok: bool = True) -> None:
        data: GeneratorDumpData = {
            "base_url": self.base_url,
            "generation_params": self.generation_params,
            "model_name": self.model_name,
            "use_cache": self.use_cache,
        }

        path.mkdir(exist_ok=exist_ok, parents=True)

        with (path / self._dump_data_filename).open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> Generator:
        with (path / cls._dump_data_filename).open(encoding="utf-8") as file:
            data: GeneratorDumpData = json.load(file)

        generation_params = data.pop("generation_params")  # type: ignore[misc]

        return cls(**data, **generation_params)
