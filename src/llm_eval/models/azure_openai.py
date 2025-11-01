"""
Azure OpenAI model integration.
Based on https://learn.microsoft.com/ja-jp/azure/ai-foundry/openai/how-to/reasoning?tabs=python-secure%2Cpy#reasoning-summary.
Please do `uv pip install openai --upgrade` to ensure you have the latest version.
"""

from typing import List, Dict, Any
import os

from openai import AzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)
from loguru import logger

from .base import BaseModel


class AzureOpenAIModel(BaseModel):
    """Azure OpenAI API model integration."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        # Validate required client configuration
        if not os.getenv(self.client_config["api_key_env"]):
            logger.warning(
                f"API key not found in environment variable {self.client_config['api_key_env']}"
            )

        if not os.getenv(self.client_config["azure_endpoint_env"]):
            logger.warning(
                f"Azure endpoint not found in environment variable {self.client_config['azure_endpoint_env']}"
            )

        # Use only client parameters for client initialization
        client_params = {
            "api_key": os.getenv(self.client_config["api_key_env"]),
            "azure_endpoint": os.getenv(self.client_config["azure_endpoint_env"]),
            "api_version": self.client_config["api_version"],
            **{
                k: v
                for k, v in self.client_config.items()
                if k not in ["api_key_env", "azure_endpoint_env", "api_version"]
            },
        }

        self._client = AzureOpenAI(**client_params)
        logger.info("Initialized Azure OpenAI client")

    @retry(
        wait=wait_random_exponential(min=1, max=300),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, "INFO"),
    )
    def _create_response_with_backoff(self, client, **api_params):
        """Create response with exponential backoff retry."""
        return client.responses.create(**api_params)

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate response for conversation messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional API parameters

        Returns:
            Dict[str, Any]: Dict with 'content' and 'raw_response' keys
        """
        client = self.get_client()

        # Convert messages to single input (responses API expects single input)
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # Build API parameters
        api_params = {"input": input_text}

        # Add parameters from base config and kwargs
        params = self._prepare_request_params(**kwargs)
        api_params.update(params.get("request", {}))

        # Make API call with backoff
        try:
            response = self._create_response_with_backoff(client, **api_params)
        except Exception as e:
            logger.error(f"API call failed after retries: {e}")
            return {"content": "", "raw_response": {"error": str(e)}, "error": str(e)}

        # Handle response with safety check
        logger.debug(
            f"API response type: {type(response)}, has output: {hasattr(response, 'output')}"
        )

        try:
            if not response or not hasattr(response, "output") or not response.output:
                logger.error(f"Invalid response structure: {response}")
                output = ""
            else:
                # Extract content from response
                output = ""

                for output_item in response.output:
                    if output_item.type == "message":
                        for content in output_item.content:
                            if content.type == "output_text":
                                output += content.text
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
        """Extract reasoning from Azure OpenAI API raw response if present."""
        try:
            # Check for reasoning in the output array
            output = raw_response.get("output", [])
            if not output:
                return ""

            reasoning_texts = []

            for output_item in output:
                if output_item.get("type") == "reasoning":
                    # Extract summary from reasoning item
                    summary = output_item.get("summary", [])
                    for summary_item in summary:
                        if summary_item.get("type") == "summary_text":
                            text = summary_item.get("text", "")
                            if text:
                                reasoning_texts.append(text)

            return "\n\n".join(reasoning_texts) if reasoning_texts else ""

        except (KeyError, IndexError, AttributeError, TypeError):
            return ""
