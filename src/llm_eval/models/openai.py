"""OpenAI model integration."""

from typing import List, Dict, Any
import os

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)
from loguru import logger

from .base import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI API model integration."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def _initialize_client(self):
        """Initialize OpenAI client."""
        # Validate required client configuration
        if not os.getenv(self.client_config["api_key_env"]):
            logger.warning(
                f"API key not found in environment variable {self.client_config['api_key_env']}"
            )

        # Use only client parameters for client initialization
        client_params = {
            "api_key": os.getenv(self.client_config["api_key_env"]),
            "base_url": self.client_config["base_url"],
            **{
                k: v
                for k, v in self.client_config.items()
                if k not in ["api_key_env", "base_url"]
            },
        }

        self._client = openai.OpenAI(**client_params)
        logger.info("Initialized OpenAI client")

    @retry(
        wait=wait_random_exponential(min=1, max=300),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, "INFO"),
    )
    def _create_completion_with_backoff(self, client, **api_params):
        """Create completion with exponential backoff retry."""
        return client.chat.completions.create(**api_params)

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate response for conversation messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional API parameters

        Returns:
            Dict[str, Any]: Dict with 'content' and 'raw_response' keys
        """
        client = self.get_client()

        # Build API parameters
        api_params = {"messages": messages}

        # Add parameters from base config and kwargs (this includes temperature, etc.)
        params = self._prepare_request_params(**kwargs)
        api_params.update(params.get("request", {}))

        # Make API call with backoff
        try:
            response = self._create_completion_with_backoff(client, **api_params)
        except Exception as e:
            logger.error(f"API call failed after retries: {e}")
            return {"content": "", "raw_response": {"error": str(e)}, "error": str(e)}

        # Handle response with safety check
        logger.debug(
            f"API response type: {type(response)}, has choices: {hasattr(response, 'choices')}"
        )

        try:
            if not response or not hasattr(response, "choices") or not response.choices:
                logger.error(f"Invalid response structure: {response}")
                output = ""
            else:
                output = response.choices[0].message.content or ""
        except (IndexError, AttributeError, TypeError) as e:
            logger.error(f"Error accessing response: {e}, response: {response}")
            output = ""

        # Build response
        result = {"content": output}
        raw_response = self._sanitize_raw_response(response.model_dump())
        reasoning = self._extract_reasoning_from_raw_response(raw_response)
        if reasoning:
            result["reasoning"] = reasoning
        result["raw_response"] = raw_response

        return result

    def generate_batch(
        self, messages_batch: List[List[Dict[str, str]]], **kwargs
    ) -> Dict[str, Any]:
        """Generate responses for multiple conversation message lists.

        Args:
            messages_batch: List of message lists
            **kwargs: Additional API parameters

        Returns:
            Dict[str, Any]: Dict with 'responses' (list) and 'early_terminated' (bool) keys
        """
        responses = []
        consecutive_failures = 0
        max_consecutive_failures = 2
        early_terminated = False

        for i, messages in enumerate(messages_batch):
            response = self.generate(messages, **kwargs)
            responses.append(response)

            # Check if the response contains an error (indicating retry limit exceeded)
            if response.get("error"):
                consecutive_failures += 1
                logger.warning(
                    f"API call failed for sample {i + 1}: {response.get('error')}"
                )

                if consecutive_failures >= max_consecutive_failures:
                    early_terminated = True
                    logger.error(
                        f"Stopping batch generation after {consecutive_failures} consecutive failures. "
                        f"Processed {i + 1}/{len(messages_batch)} samples."
                    )
                    break
            else:
                # Reset counter on successful response
                consecutive_failures = 0

            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{len(messages_batch)} responses")

        return {"responses": responses, "early_terminated": early_terminated}

    def _extract_reasoning_from_raw_response(self, raw_response: Dict[str, Any]) -> str:
        """Extract reasoning from OpenAI API raw response if present."""
        try:
            # Check if the response has reasoning field (for models that support reasoning)
            choices = raw_response.get("choices", [])
            if not choices:
                return ""

            choice = choices[0]
            message = choice.get("message", {})

            # Claude Sonnet 4 format: Check for thinking content in message.content array
            content = message.get("content")
            if isinstance(content, list):
                for content_block in content:
                    if isinstance(content_block, dict):
                        # Check for thinking block
                        if content_block.get("type") == "thinking":
                            thinking = content_block.get("thinking", "")
                            if thinking:
                                return thinking
                        # Check for redacted thinking block (fallback)
                        elif content_block.get("type") == "redacted_thinking":
                            # For redacted thinking, we can't extract the content as it's encrypted
                            # But we can indicate that thinking was present
                            return (
                                "[Thinking content was present but redacted for safety]"
                            )

            # Standard format: Check for reasoning in message
            reasoning = message.get("reasoning", "")
            if reasoning:
                return reasoning

            # Alternative format: Check for reasoning_content in message
            reasoning_content = message.get("reasoning_content", "")
            if reasoning_content:
                return reasoning_content

            # Fallback: check in choice level
            reasoning = choice.get("reasoning", "")
            if reasoning:
                return reasoning

            # Fallback: check reasoning_content in choice level
            reasoning_content = choice.get("reasoning_content", "")
            if reasoning_content:
                return reasoning_content

            return ""
        except (KeyError, IndexError, AttributeError, TypeError):
            return ""
