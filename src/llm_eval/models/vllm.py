"""vLLM model integration for local LLM inference."""

import torch
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from loguru import logger

from .base import BaseModel


class VLLMModel(BaseModel):
    """vLLM local model integration for Qwen2.5."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def _initialize_client(self):
        """Initialize vLLM client."""
        logger.info(f"Initializing vLLM with model: {self.client_config['model']}")

        # Auto-detect GPU count for tensor parallelism
        client_config = self.client_config.copy()
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            client_config["tensor_parallel_size"] = gpu_count
            logger.info(
                f"Auto-detected {gpu_count} GPUs, setting tensor_parallel_size={gpu_count}"
            )
        else:
            client_config["tensor_parallel_size"] = 1
            logger.info("No CUDA GPUs detected, setting tensor_parallel_size=1")

        # Use client_config directly as LLM parameters
        self._client = LLM(**client_config)

        logger.info(f"Initialized vLLM client for model: {self.client_config['model']}")

    def generate_batch(
        self, messages_batch: List[List[Dict[str, str]]], **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate responses for multiple conversations using vLLM chat interface."""
        client = self.get_client()

        # Get all parameters from config and kwargs
        params = self._prepare_request_params(**kwargs)

        outputs = client.chat(
            messages_batch,
            SamplingParams(**params.get("sampling", {})),
            chat_template_kwargs=params.get("chat_template_kwargs", {}),
        )

        responses = []
        for output in outputs:
            raw_text = output.outputs[0].text.strip()
            parsed = self._parse_reasoning_content(raw_text)

            # Build response
            response = {"content": parsed["content"]}
            if parsed["reasoning"]:
                response["reasoning"] = parsed["reasoning"]
            response["raw_response"] = self._sanitize_raw_response(
                self._convert_request_output_to_dict(output)
            )

            responses.append(response)

        return {"responses": responses, "early_terminated": False}

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate response using vLLM."""
        batch_result = self.generate_batch([messages], **kwargs)
        return batch_result["responses"][0]

    def _convert_request_output_to_dict(self, output) -> Dict[str, Any]:
        """Convert vLLM RequestOutput to dictionary format recursively."""

        def _convert_to_serializable(obj):
            """Recursively convert objects to JSON-serializable format."""
            if hasattr(obj, "__dict__"):
                # Custom object with __dict__
                return {k: _convert_to_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [_convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _convert_to_serializable(v) for k, v in obj.items()}
            elif obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            else:
                # Fallback: convert to string for unknown types
                return str(obj)

        return _convert_to_serializable(output)

    def _parse_reasoning_content(self, text: str) -> Dict[str, str]:
        """Parse content to separate reasoning and final response with robust tag-based parsing.

        Handles multiple cases:
        1. Normal: <think>reasoning</think>content
        2. Malformed: <think>reasoning</think>content</think>more_content
        3. No Start Tag: reasoning</think>content
        """
        reasoning_config = self.params.get("reasoning_parser", {})

        if not reasoning_config.get("enabled", False):
            return {"content": text, "reasoning": ""}

        start_tag = reasoning_config.get("start_tag", "")
        end_tag = reasoning_config.get("end_tag", "")

        if not start_tag or not end_tag:
            return {"content": text, "reasoning": ""}

        # Find start and end positions
        start_pos = text.find(start_tag)
        end_pos = text.rfind(end_tag)

        # Case 1: Normal format with both start and end tags
        if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
            # Extract reasoning content between tags
            reasoning_start = start_pos + len(start_tag)
            reasoning = text[reasoning_start:end_pos].strip()

            # Extract final content by removing the entire reasoning section
            before_reasoning = text[:start_pos].strip()
            after_reasoning = text[end_pos + len(end_tag) :].strip()

            # Combine before and after parts for final content
            content_parts = [
                part for part in [before_reasoning, after_reasoning] if part
            ]
            final_content = " ".join(content_parts).strip()

            return {"content": final_content, "reasoning": reasoning}

        # Case 2: Only end tag, no start tag
        elif start_pos == -1 and end_pos != -1:
            # Everything before end_tag is reasoning content
            reasoning = text[:end_pos].strip()
            # Everything after end_tag is final content
            final_content = text[end_pos + len(end_tag) :].strip()

            return {"content": final_content, "reasoning": reasoning}

        # Case 3: No valid tags found, return as-is
        else:
            return {"content": text, "reasoning": ""}
