"""Batch processing runner with resume functionality for LLM evaluation and inference."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import argparse
import json

from loguru import logger

from .run_batch import BatchRunner
from .load_config import ConfigLoader
from .load_sample import SampleLoader
from .exec_inference import InferenceExecutor


class BatchRunnerWithRetry(BatchRunner):
    """Orchestrates LLM batch evaluation and inference workflows with retry functionality."""

    def __init__(self, output_dir: str = "results", log_level: str = "INFO"):
        """Initialize pipeline with core dependencies."""
        # Initialize parent class
        super().__init__(output_dir, log_level)

        logger.info("Initialized batch processing pipeline with retry functionality")

    def infer_batch_with_retry(
        self,
        task_name: str,
        model_names: List[str],
        dataset_names: Optional[List[str]] = None,
        template_names: Optional[List[str]] = None,
        checkpoint_interval: Optional[int] = None,
        num_retry: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run inference batch with retry logic."""
        logger.info(
            f"[INFER WITH RETRY] Starting inference batch with {num_retry} retry round(s)"
        )

        final_result = {}
        total_resumed = 0

        for retry_round in range(num_retry):
            logger.info(
                f"[RETRY ROUND {retry_round + 1}/{num_retry}] Starting inference batch..."
            )

            # Run inference batch with resume
            result = self.infer_batch_with_resume(
                task_name=task_name,
                model_names=model_names,
                dataset_names=dataset_names,
                template_names=template_names,
                checkpoint_interval=checkpoint_interval,
                **kwargs,
            )

            final_result = result
            total_resumed += result.get("resumed_experiments", 0)

            # Check if there's still incomplete work
            has_incomplete = self._has_incomplete_work(
                task_name, dataset_names, model_names, template_names
            )

            if not has_incomplete:
                logger.info(
                    f"[RETRY ROUND {retry_round + 1}/{num_retry}] All experiments completed successfully!"
                )
                break
            else:
                if retry_round < num_retry - 1:
                    logger.info(
                        f"[RETRY ROUND {retry_round + 1}/{num_retry}] Incomplete work detected, starting retry round {retry_round + 2}..."
                    )
                else:
                    logger.warning(
                        f"[RETRY ROUND {retry_round + 1}/{num_retry}] Reached maximum retry rounds, some work may remain incomplete"
                    )

        # Update final result with total statistics
        final_result["total_resumed_experiments"] = total_resumed
        final_result["retry_rounds_completed"] = min(retry_round + 1, num_retry)
        final_result["status"] = (
            "completed"
            if not self._has_incomplete_work(
                task_name, dataset_names, model_names, template_names
            )
            else "partially_completed"
        )

        logger.info(
            f"[INFER WITH RETRY] Completed: {final_result.get('total_experiments', 'N/A')} experiments, "
            f"{total_resumed} total resumed across {final_result['retry_rounds_completed']} round(s)"
        )

        return final_result

    def _has_incomplete_work(
        self,
        task_name: str,
        dataset_names: List[str],
        model_names: List[str],
        template_names: List[str],
    ) -> bool:
        """Check if there's any incomplete work that could benefit from resume functionality."""
        for dataset_name in dataset_names:
            for model_name in model_names:
                for template_name in template_names:
                    existing_result = self._find_resume_target(
                        task_name, dataset_name, model_name, template_name
                    )
                    if existing_result:
                        interactions = existing_result["raw_data"].get(
                            "interactions", []
                        )
                        # Check if there are any failed responses
                        for interaction in interactions:
                            raw_response = interaction.get("raw_response", {})
                            if (
                                raw_response.get("error")
                                or not raw_response.get("content", "").strip()
                            ):
                                return True
        return False

    def _infer_single_with_resume(
        self,
        task_name: str,
        model_name: str,
        dataset_name: str,
        template_name: str,
        inference_executor,
        task_configs: Dict[str, Any],
        model_config: Dict[str, Any],
        checkpoint_interval: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Process a single experiment with resume support."""
        # Check for existing results
        existing_result = self._find_resume_target(
            task_name, dataset_name, model_name, template_name
        )

        if existing_result:
            logger.info(f"Found existing results: {existing_result['timestamp']}")

            # Load all samples to determine total count
            dataset_config = task_configs["datasets"][dataset_name]
            all_samples = SampleLoader.load_dataset(dataset_config, dataset_name)
            total_samples = len(all_samples)

            # Identify samples to process: failed and remaining
            failed_indices = self._identify_failed_samples(
                existing_result["raw_data"], total_samples
            )

            # Determine final samples, messages, and outputs
            if failed_indices:
                logger.info(
                    f"Starting resume processing for {len(failed_indices)} samples..."
                )

                # Get component classes from config
                template_class = task_configs["components"]["template_class"]

                # Process only failed samples
                partial_samples = [all_samples[i] for i in failed_indices]

                if (
                    checkpoint_interval
                    and checkpoint_interval > 0
                    and len(partial_samples) > checkpoint_interval
                ):
                    # Checkpoint mode for resume processing
                    logger.info(
                        f"Checkpoint mode enabled for resume: processing {len(partial_samples)} failed samples in chunks of {checkpoint_interval}"
                    )

                    all_new_samples = []
                    all_new_messages = []
                    all_new_outputs = []

                    for chunk_start in range(
                        0, len(partial_samples), checkpoint_interval
                    ):
                        chunk_end = min(
                            chunk_start + checkpoint_interval, len(partial_samples)
                        )
                        chunk_samples = partial_samples[chunk_start:chunk_end]

                        logger.info(
                            f"Processing resume chunk {chunk_start + 1}-{chunk_end}/{len(partial_samples)}"
                        )

                        # Generate responses for chunk
                        _, chunk_messages, chunk_outputs, _ = (
                            inference_executor.generate_responses_for_samples(
                                template_class,
                                chunk_samples,
                                template_name,
                                dataset_config,
                                model_config,
                                save=False,
                                **kwargs,
                            )
                        )

                        # Accumulate chunk results
                        all_new_samples.extend(chunk_samples)
                        all_new_messages.extend(chunk_messages)
                        all_new_outputs.extend(chunk_outputs)

                        # Get sample class from config
                        sample_class = task_configs["components"]["sample_class"]

                        # Merge results with current chunk
                        final_samples, final_messages_batch, final_outputs = (
                            self._merge_resume_results(
                                existing_result,
                                all_new_samples,
                                all_new_messages,
                                all_new_outputs,
                                failed_indices[: len(all_new_samples)],
                                sample_class,
                            )
                        )

                        # Save checkpoint (merged results)
                        self.output_saver.save_raw_responses(
                            samples=final_samples,
                            messages_batch=final_messages_batch,
                            outputs=final_outputs,
                            model_config=model_config,
                            dataset_config=dataset_config,
                        )

                        logger.info(
                            f"Resume checkpoint saved: {len(all_new_samples)}/{len(partial_samples)} failed samples processed"
                        )
                else:
                    # Standard resume mode: process all failed samples at once
                    _, new_messages_batch, new_outputs, _ = (
                        inference_executor.generate_responses_for_samples(
                            template_class,
                            partial_samples,
                            template_name,
                            dataset_config,
                            model_config,
                            save=False,
                            **kwargs,
                        )
                    )

                    # Get sample class from config
                    sample_class = task_configs["components"]["sample_class"]

                    # Merge results
                    final_samples, final_messages_batch, final_outputs = (
                        self._merge_resume_results(
                            existing_result,
                            partial_samples,
                            new_messages_batch,
                            new_outputs,
                            failed_indices,
                            sample_class,
                        )
                    )

                    # Save raw responses manually for merged results
                    self.output_saver.save_raw_responses(
                        samples=final_samples,
                        messages_batch=final_messages_batch,
                        outputs=final_outputs,
                        model_config=model_config,
                        dataset_config=dataset_config,
                    )

                logger.info(
                    f"Resume inference completed: {len(final_outputs)} responses generated"
                )
                return True
            else:
                # All samples completed successfully, no processing needed
                logger.info(
                    "All samples completed successfully! No resume processing required."
                )
                return True
        else:
            logger.info("No existing results found, running normal inference")
            # No existing results, run normal inference from parent class
            self._infer_single(
                task_name=task_name,
                model_name=model_name,
                dataset_name=dataset_name,
                template_name=template_name,
                inference_executor=inference_executor,
                task_configs=task_configs,
                model_config=model_config,
                checkpoint_interval=checkpoint_interval,
                **kwargs,
            )
            return False

    def infer_batch_with_resume(
        self,
        task_name: str,
        model_names: List[str],
        dataset_names: Optional[List[str]] = None,
        template_names: Optional[List[str]] = None,
        checkpoint_interval: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run batch inference with resume functionality for partial results."""
        # Load configurations (no metrics needed for inference)
        task_configs = ConfigLoader.load_task_configs(
            task_name, dataset_names, template_names, []
        )
        model_configs_with_types = ConfigLoader.load_model_configs(model_names)

        logger.info(
            f"[RESUME MODE] Loaded task config and {len(model_configs_with_types)} model configs"
        )

        # Calculate total experiments for progress tracking
        total_experiments = (
            len(model_configs_with_types) * len(dataset_names) * len(template_names)
        )
        experiment_count = 0
        resumed_count = 0

        # Generate batch-wide common timestamp
        batch_timestamp = datetime.now()
        logger.info(
            f"[RESUME MODE] Batch started at: {batch_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Triple loop: Model -> Dataset -> Template
        for model_name, (model_config, model_type) in zip(
            model_names, model_configs_with_types
        ):
            inference_executor = InferenceExecutor(
                model_type, model_config, self.output_saver
            )
            logger.info(f"Created model: {model_name} ({model_type})")

            for dataset_name in dataset_names:
                for template_name in template_names:
                    experiment_count += 1
                    try:
                        experiment_start_time = datetime.now()
                        logger.info(
                            f"[{experiment_count}/{total_experiments}] Starting: {task_name}/{model_name}/{dataset_name}/{template_name} (started at: {experiment_start_time.strftime('%H:%M:%S')})"
                        )

                        self._start_session(
                            task_name,
                            model_name,
                            template_name,
                            dataset_name,
                            batch_timestamp,
                            "resume_session.log",
                        )

                        # Process single experiment with resume support
                        was_resumed = self._infer_single_with_resume(
                            task_name=task_name,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            template_name=template_name,
                            inference_executor=inference_executor,
                            task_configs=task_configs,
                            model_config=model_config,
                            checkpoint_interval=checkpoint_interval,
                            **kwargs,
                        )

                        if was_resumed:
                            resumed_count += 1

                        logger.info(
                            f"[{experiment_count}/{total_experiments}] Completed: {task_name}/{model_name}/{dataset_name}/{template_name}"
                        )

                    finally:
                        # Stop session-specific logging
                        self._stop_session()

        # Batch inference completed
        logger.info(
            f"[RESUME MODE] Batch completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(
            f"[RESUME MODE] Resumed {resumed_count} out of {total_experiments} experiments"
        )
        return {
            "total_experiments": total_experiments,
            "resumed_experiments": resumed_count,
            "batch_timestamp": batch_timestamp,
            "status": "inference_completed",
        }

    def _find_resume_target(
        self, task_name: str, dataset_name: str, model_name: str, template_name: str
    ) -> Optional[Dict[str, Any]]:
        """Find existing results for resume functionality using parent class method.

        Returns:
            Dict containing resume data if found, None otherwise
        """
        result = self._find_latest_raw_responses(
            task_name, dataset_name, model_name, template_name
        )
        if not result:
            return None

        filepath, timestamp = result
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            return {
                "raw_responses_file": filepath,
                "raw_data": raw_data,
                "timestamp": timestamp.strftime("%Y%m%d_%H%M%S"),
            }
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load raw responses from {filepath}: {e}")
            return None

    def _identify_failed_samples(
        self, raw_data: Dict[str, Any], total_samples: int
    ) -> List[int]:
        """Identify failed and remaining samples that need processing.

        Args:
            raw_data: Raw responses data
            total_samples: Total number of samples expected

        Returns:
            List of sample indices that need processing (failed + remaining)
        """
        failed_indices = []
        interactions = raw_data.get("interactions", [])
        existing_count = len(interactions)

        # Check existing samples for failures
        for i in range(min(existing_count, total_samples)):
            interaction = interactions[i]
            raw_response = interaction.get("raw_response", {})

            # Check for errors or empty content
            if raw_response.get("error") or not raw_response.get("content", "").strip():
                failed_indices.append(i)

        # Add remaining unprocessed samples
        remaining_indices = list(range(existing_count, total_samples))
        all_indices = failed_indices + remaining_indices

        logger.info(
            f"Resume status: {existing_count}/{total_samples} processed, "
            f"{len(failed_indices)} failed, {len(remaining_indices)} remaining, "
            f"{len(all_indices)} to process"
        )

        return all_indices

    def _merge_resume_results(
        self,
        existing_data: Dict[str, Any],
        new_samples: List[Any],
        new_messages_batch: List[List[Dict[str, str]]],
        new_outputs: List[Any],
        failed_indices: List[int],
        sample_class: str,
    ) -> Tuple[List[Any], List[List[Dict[str, str]]], List[Any]]:
        """Merge existing results with new results for failed samples.

        Returns:
            Tuple of (merged_samples, merged_messages_batch, merged_outputs)
        """
        # Load existing results
        existing_samples, existing_messages_batch, existing_outputs = (
            SampleLoader.load_raw_responses(
                existing_data["raw_responses_file"], sample_class
            )
        )

        # Create copies for merging
        merged_samples = existing_samples.copy()
        merged_messages_batch = existing_messages_batch.copy()
        merged_outputs = existing_outputs.copy()

        # Replace failed samples with new results
        for i, failed_idx in enumerate(failed_indices):
            if i < len(new_outputs):  # Safety check
                # Extend arrays if necessary
                while len(merged_samples) <= failed_idx:
                    merged_samples.append(None)
                    merged_messages_batch.append(None)
                    merged_outputs.append(None)

                merged_samples[failed_idx] = new_samples[i]
                merged_messages_batch[failed_idx] = new_messages_batch[i]
                merged_outputs[failed_idx] = new_outputs[i]

        logger.info(f"Merged {len(new_outputs)} new results with existing results")
        return merged_samples, merged_messages_batch, merged_outputs


def main():
    """CLI entry point for batch processing with retry functionality."""
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework - Batch Processing with Retry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Both inference and evaluation with retry (default)
  uv run --env-file .env python -m llm_eval.run_batch_with_retry --mode both --config configs/batch/test.yaml
  uv run --env-file .env python -m llm_eval.run_batch_with_retry --config configs/batch/test.yaml  # same as above (both is default)

  # Inference only with retry (single retry)
  uv run --env-file .env python -m llm_eval.run_batch_with_retry --mode infer --config configs/batch/test.yaml

  # Inference only with multiple retries
  uv run --env-file .env python -m llm_eval.run_batch_with_retry --mode infer --config configs/batch/test.yaml --num-retry 3

  # Both mode with multiple retries for inference
  uv run --env-file .env python -m llm_eval.run_batch_with_retry --mode both --config configs/batch/test.yaml --num-retry 5

  # Evaluation only with latest detection
  uv run --env-file .env python -m llm_eval.run_batch_with_retry --mode evaluate --config configs/batch/test.yaml

  # Config with CLI overrides
  uv run --env-file .env python -m llm_eval.run_batch_with_retry --mode both --config configs/batch/test.yaml --models plamo-2_0-prime

  # CLI only (no config file)
  uv run --env-file .env python -m llm_eval.run_batch_with_retry --mode both --task medec --models gpt-4_1 --datasets small --templates paper_p1
""",
    )

    # Priority order: CLI args > YAML config > src defaults (if exists)

    parser.add_argument(
        "--mode",
        type=str,
        choices=["infer", "evaluate", "both"],
        default="both",
        help="Execution mode: infer (inference only), evaluate (evaluation only), both (inference + evaluation)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to batch configuration YAML file (optional, can use CLI args instead)",
    )
    parser.add_argument("--task", type=str, help="Task name (overrides config)")
    parser.add_argument("--models", nargs="+", help="Model names (overrides config)")
    parser.add_argument(
        "--datasets", nargs="+", help="Dataset names (overrides config)"
    )
    parser.add_argument(
        "--templates", nargs="+", help="Template names (overrides config)"
    )
    parser.add_argument("--metrics", nargs="+", help="Metric names (overrides config)")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N samples (default: 10)",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory path")
    parser.add_argument(
        "--num-retry",
        type=int,
        help="Number of retry rounds for inference (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    config = {}
    if args.config:
        config = ConfigLoader.load_batch_config(args.config)

    if args.output_dir:
        output_dir = args.output_dir
    elif config.get("output_dir"):
        output_dir = config["output_dir"]
    else:
        output_dir = "results"

    runner = BatchRunnerWithRetry(output_dir=output_dir, log_level=args.log_level)

    logger.info(f"Starting {args.mode} mode with retry functionality")

    task_name = args.task or config.get("task")
    model_names = args.models or config.get("models")
    dataset_names = args.datasets or config.get("datasets")
    template_names = args.templates or config.get("templates")
    metric_names = args.metrics or config.get("metrics")
    checkpoint_interval = args.checkpoint_interval or config.get("checkpoint_interval")
    num_retry = args.num_retry or config.get("num_retry", 1)

    if not task_name or not model_names:
        raise ValueError(
            "--task and --models are required for evaluation/inference (can be provided via --config or CLI args)"
        )

    if args.mode == "infer":
        # Inference only mode with retry
        logger.info(
            f"Starting inference-only batch with {args.num_retry} retry round(s)..."
        )
        result = runner.infer_batch_with_retry(
            task_name=task_name,
            model_names=model_names,
            dataset_names=dataset_names,
            template_names=template_names,
            checkpoint_interval=checkpoint_interval,
            num_retry=num_retry,
        )

        logger.info(
            f"Inference mode completed: {result.get('total_experiments', 'N/A')} experiments finished, "
            f"{result.get('total_resumed_experiments', 0)} total resumed, "
            f"status: {result.get('status', 'unknown')}"
        )

    elif args.mode == "evaluate":
        # Evaluation only mode (always uses latest detection)
        logger.info("Starting evaluation-only batch with latest detection...")
        evaluation_result = runner.evaluate_batch(
            task_name=task_name,
            model_names=model_names,
            dataset_names=dataset_names,
            template_names=template_names,
            metric_names=metric_names,
            batch_timestamp=None,  # Always use latest detection
        )

        logger.info(
            f"Evaluation mode completed: {evaluation_result.get('successful', 'N/A')}/{evaluation_result.get('total_experiments', 'N/A')} evaluation experiments successful"
        )

    elif args.mode == "both":
        # Both inference and evaluation (2-phase processing)
        logger.info(
            f"Starting both inference and evaluation with {args.num_retry} inference retry round(s)..."
        )

        # Phase 1: Run inference batch with retry
        logger.info("Phase 1: Starting inference batch with retry...")
        inference_result = runner.infer_batch_with_retry(
            task_name=task_name,
            model_names=model_names,
            dataset_names=dataset_names,
            template_names=template_names,
            checkpoint_interval=checkpoint_interval,
            num_retry=num_retry,
        )

        logger.info(
            f"Phase 1 completed: {inference_result.get('total_experiments', 'N/A')} experiments finished, "
            f"{inference_result.get('total_resumed_experiments', 0)} total resumed across "
            f"{inference_result.get('retry_rounds_completed', 0)} round(s)"
        )

        # Phase 2: Run evaluation batch with latest detection (single execution)
        logger.info("Phase 2: Starting evaluation batch with latest detection...")
        evaluation_result = runner.evaluate_batch(
            task_name=task_name,
            model_names=model_names,
            dataset_names=dataset_names,
            template_names=template_names,
            metric_names=metric_names,
            batch_timestamp=None,  # Use latest detection
        )

        logger.info(
            f"Phase 2 completed: {evaluation_result.get('successful', 'N/A')}/{evaluation_result.get('total_experiments', 'N/A')} evaluation experiments successful"
        )
        logger.info("Both-mode batch processing completed successfully")


if __name__ == "__main__":
    main()
