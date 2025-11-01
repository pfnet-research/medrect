"""Base model class for LLM integrations."""

import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Any



class BaseModel(ABC):
    """Base class for all model integrations."""

    def __init__(self, config: Dict[str, Any]):
        self.client_config = config.get("client", {})
        self.secret_params = config.get("secret", {})
        self.params = {k: v for k, v in config.items() if k not in ["client", "secret"]}
        self._client = None

    @abstractmethod
    def _initialize_client(self):
        """Initialize the model client.

        Implementation Notes:
            - Use self.client_config for client configuration parameters
            - Store the initialized client in self._client
        """
        pass

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate response for conversation messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Example: [{"role": "user", "content": "Hello"}]

        Returns:
            Dictionary containing the model response and metadata:
            {
                "content": str,  # The generated text response
                "raw_response": dict  # Raw model response (use self._sanitize_raw_response())
            }

        Implementation Notes:
            - Use self._prepare_request_params(**kwargs) for parameter preparation
            - Apply self._sanitize_raw_response(raw_response) to raw_response before returning
        """
        pass

    @abstractmethod
    def generate_batch(
        self, messages_batch: List[List[Dict[str, str]]], **kwargs
    ) -> Dict[str, Any]:
        """Generate responses for multiple conversation message lists.

        Args:
            messages_batch: List of message lists, each in the same format as generate()
            **kwargs: Additional parameters passed to each generation

        Returns:
            Dict with 'responses' (list) and 'early_terminated' (bool) keys
        """
        pass

    def get_client(self):
        """Get the model client, initializing if necessary."""
        if self._client is None:
            self._initialize_client()
        return self._client

    def _prepare_request_params(self, **kwargs) -> Dict[str, Any]:
        """Prepare request parameters from config and kwargs."""
        params = {}

        # Add regular API parameters from params
        for key, value in self.params.items():
            params[key] = value

        # Add secret parameters (these will go to API but not logged)
        for key, value in self.secret_params.items():
            params[key] = value

        # Override with kwargs
        for key, value in kwargs.items():
            params[key] = value

        return params

    def _sanitize_raw_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """Remove secret parameters from raw response for safe storage/logging."""
        # If no secret parameters, return as-is
        if not self.secret_params:
            return raw_response

        # Create a deep copy to avoid modifying original
        sanitized = copy.deepcopy(raw_response)

        # Recursively remove secret parameter keys from the response
        def remove_secret_keys(obj, secret_keys):
            if isinstance(obj, dict):
                return {
                    k: remove_secret_keys(v, secret_keys)
                    for k, v in obj.items()
                    if k not in secret_keys
                }
            elif isinstance(obj, list):
                return [remove_secret_keys(item, secret_keys) for item in obj]
            else:
                return obj

        # Get secret parameter keys to remove
        secret_keys = set(self.secret_params.keys())

        return remove_secret_keys(sanitized, secret_keys)
