"""Inference executor for generating model responses."""

from typing import List, Dict, Any, Tuple

from loguru import logger

from .component_factory import component_factory
from .output.output_saver import OutputSaver


class InferenceExecutor:
    """Handles model inference and response generation."""

    def __init__(
        self, model_type: str, model_config: Dict[str, Any], output_saver: OutputSaver
    ):
        """Initialize inference executor and create model instance.

        Args:
            model_type: Type of model to create (e.g., 'openai', 'azure_openai')
            model_config: Configuration dictionary for model initialization
            output_saver: OutputSaver instance for saving results
        """
        self.model = component_factory.get_component_instance(
            "models", model_type, model_config
        )
        self.output_saver = output_saver

    def generate_responses_for_samples(
        self,
        template_class: str,
        samples: List[Any],
        template_name: str,
        dataset_config: Dict[str, Any],
        model_config: Dict[str, Any],
        save: bool = True,
        **kwargs,
    ) -> Tuple[List[Any], List[List[Dict[str, str]]], List[Dict[str, Any]], bool]:
        """Generate model responses for a specific list of samples and optionally save.

        Args:
            template_class: Class name for template component
            samples: List of samples to process
            template_name: Name of the template to use
            dataset_config: Dataset configuration
            model_config: Model configuration
            save: Whether to save raw responses (default: True)
            **kwargs: Additional parameters for model generation

        Returns:
            Tuple of (samples, messages_batch, outputs, early_terminated)
        """
        logger.info(f"Processing {len(samples)} provided samples")

        # Generate messages
        messages_batch = self._generate_messages(template_class, samples, template_name)
        logger.info(f"Generated {len(messages_batch)} message sets")

        # Get model outputs
        logger.info("Generating model outputs...")
        batch_result = self.model.generate_batch(messages_batch, **kwargs)
        outputs = batch_result["responses"]
        early_terminated = batch_result.get("early_terminated", False)
        logger.info(f"Generated {len(outputs)} outputs")

        if early_terminated:
            logger.warning("Early termination detected during inference")

        # Save raw responses if save=True
        if save:
            self.output_saver.save_raw_responses(
                samples=samples,
                messages_batch=messages_batch,
                outputs=outputs,
                model_config=model_config,
                dataset_config=dataset_config,
            )

        return samples, messages_batch, outputs, early_terminated

    def _generate_messages(
        self, template_class: str, samples: List[Any], template_name: str
    ) -> List[List[Dict[str, str]]]:
        """Generate message sets using specified template."""
        # Create template instance
        template = component_factory.get_component_instance(
            "templates", template_class, template_name
        )

        messages_batch = []
        for sample in samples:
            messages = template.generate_messages(sample)
            messages_batch.append(messages)

        logger.info(
            f"Generated {len(messages_batch)} message sets using template '{template_name}'"
        )
        return messages_batch
