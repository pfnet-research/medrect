"""vLLM model integration with LoRA adapter support."""

import copy
import torch
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from loguru import logger

from .vllm import VLLMModel


class VLLMLoRAModel(VLLMModel):
    """vLLM model with LoRA adapter support for fine-tuned models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def _initialize_client(self):
        """Initialize vLLM client with LoRA support."""
        client_config = copy.deepcopy(self.client_config)

        logger.info(f"Initializing vLLM with model: {client_config['model']}")

        # Auto-detect GPU count for tensor parallelism
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            client_config["tensor_parallel_size"] = gpu_count
            logger.info(
                f"Auto-detected {gpu_count} GPUs, setting tensor_parallel_size={gpu_count}"
            )
        else:
            client_config["tensor_parallel_size"] = 1
            logger.info("No CUDA GPUs detected, setting tensor_parallel_size=1")

        # Initialize vLLM with LoRA support
        self._client = LLM(**client_config)

        logger.info("Initialized vLLM client with LoRA support")
        logger.info(
            f"Model: {client_config['model']}, LoRA rank: {client_config.get('max_lora_rank', 64)}"
        )

    def generate_batch(
        self, messages_batch: List[List[Dict[str, str]]], **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate responses for multiple conversations using vLLM with LoRA adapter."""
        client = self.get_client()

        # Get all parameters from config and kwargs
        params = self._prepare_request_params(**kwargs)
        sampling_params = SamplingParams(**params.get("sampling", {}))

        # Create LoRA request
        lora_request = LoRARequest(
            lora_name="fine_tuned_adapter",
            lora_int_id=1,
            lora_local_path=params["lora_adapter_path"],
        )

        logger.info(
            f"Generating batch of {len(messages_batch)} conversations with LoRA adapter"
        )

        try:
            outputs = client.chat(
                messages=messages_batch,
                sampling_params=sampling_params,
                lora_request=lora_request,
                chat_template_kwargs=params.get("chat_template_kwargs", {}),
            )
        except Exception as e:
            logger.error(f"Error during LoRA generation: {e}")
            raise

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

        logger.info(f"Successfully generated {len(responses)} responses")
        return {"responses": responses, "early_terminated": False}

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate response using vLLM with LoRA adapter."""
        batch_result = self.generate_batch([messages], **kwargs)
        return batch_result["responses"][0]
