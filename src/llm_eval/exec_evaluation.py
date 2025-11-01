"""Evaluation executor for computing metrics on model outputs."""

import hashlib
import json
from typing import List, Dict, Any, Tuple

from loguru import logger

from .component_factory import component_factory
from .output.output_saver import OutputSaver


class EvaluationExecutor:
    """Handles evaluation of model outputs using metrics."""

    def __init__(self, output_saver: OutputSaver):
        """Initialize evaluation executor.

        Args:
            output_saver: OutputSaver instance for saving results
        """
        self._metric_cache = {}  # Cache for metric instances to avoid reloading heavy models
        self._parser_cache = {}  # Cache for parser instances
        self.output_saver = output_saver

    def evaluate_outputs(
        self,
        outputs: List[Any],
        samples: List[Any],
        metrics_config: Dict[str, Dict[str, Any]],
        save: bool = True,
    ) -> Tuple[
        Dict[str, Dict[str, float]],
        Dict[str, List[Any]],
        Dict[str, List[Dict[str, Any]]],
    ]:
        """Evaluate model outputs using specified metrics and optionally save predictions.

        Args:
            outputs: List of model outputs
            samples: List of samples (already converted to proper objects)
            metrics_config: List of metric configurations
            save: Whether to save predictions (default: True)

        Returns:
            Tuple of (metric_results, predictions, detailed_results)
        """
        if not metrics_config:
            logger.warning("No metrics config provided, returning empty results")
            return {}, {}, {}

        metric_names = list(metrics_config.keys())
        logger.info(f"Computing metrics: {metric_names}")

        results = {}
        detailed_results = {}

        # Handle different output formats
        if outputs and isinstance(outputs[0], dict):
            # Extract content from dictionary format (e.g., from OpenAI model)
            text_outputs = [output.get("content", "") for output in outputs]
        else:
            # Assume it's already a list of strings
            text_outputs = outputs

        # Step 1: Get unique parsers to avoid duplicate processing
        unique_parsers = self._get_unique_parsers(metric_names, metrics_config)
        logger.info(
            f"Using {len(unique_parsers)} unique parsers for {len(metric_names)} metrics"
        )

        # Step 2: Execute each unique parser once and store results
        parsed_results = {}
        for parser_name, parser in unique_parsers.items():
            # Parse all outputs with this parser
            parser_results = []
            for output in text_outputs:
                parsed = parser.parse(output)
                parser_results.append(parsed)

            parsed_results[parser_name] = parser_results
            logger.debug(
                f"Parser '{parser_name}' processed {len(parser_results)} outputs"
            )

        # Step 3: Compute metrics using shared parser results
        for metric_name in metric_names:
            metric_config = metrics_config[metric_name]
            parser_name = metric_config["parser"]
            predictions = parsed_results[parser_name]

            # Get metric class and compute scores
            # Use 'class' field directly (required in config)
            registry_metric_name = metric_config["class"]

            # Extract parameters for metric initialization (exclude structural fields)
            metric_params = {
                k: v
                for k, v in metric_config.items()
                if k not in ["name", "class", "parser"]
            }

            # Get cached metric instance to avoid reloading heavy models
            metric = self._get_metric(metric_name, registry_metric_name, metric_params)
            references = metric.extract_references(samples)

            metric_scores = metric.compute_metric(predictions, references)
            results[metric_name] = metric_scores["aggregated"]
            detailed_results[metric_name] = metric_scores["per_item"]

            logger.info(f"Computed {metric_name}: {metric_scores['aggregated']}")

        # Save predictions if save=True
        if save and results:
            self.output_saver.save_predictions(
                samples=samples,
                predictions=parsed_results,
                metric_results=results,
                detailed_results=detailed_results,
                metrics_config=metrics_config,
            )

        return results, parsed_results, detailed_results

    def _get_metric(
        self, metric_name: str, registry_metric_name: str, metric_params: Dict[str, Any]
    ) -> Any:
        """Get cached metric instance."""
        # Create cache key based on metric type and parameters
        # Create a stable key from metric parameters
        param_str = json.dumps(metric_params, sort_keys=True)
        cache_key = (
            f"{registry_metric_name}_{hashlib.md5(param_str.encode()).hexdigest()[:8]}"
        )

        if cache_key not in self._metric_cache:
            logger.info(
                f"Creating new metric instance for {metric_name} (key: {cache_key})"
            )
            metric = component_factory.get_component_instance(
                "metrics", registry_metric_name, **metric_params
            )
            self._metric_cache[cache_key] = metric
        else:
            logger.info(
                f"Using cached metric instance for {metric_name} (key: {cache_key})"
            )

        return self._metric_cache[cache_key]

    def _get_unique_parsers(
        self, metric_names: List[str], metrics_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get unique parsers needed for the specified metrics."""
        unique_parsers = {}

        for metric_name in metric_names:
            parser_name = metrics_config[metric_name]["parser"]

            # Use cached parser or create new one
            if parser_name not in self._parser_cache:
                self._parser_cache[parser_name] = (
                    component_factory.get_component_instance("parsers", parser_name)
                )
                logger.debug(f"Created new parser instance: {parser_name}")

            unique_parsers[parser_name] = self._parser_cache[parser_name]

        return unique_parsers
