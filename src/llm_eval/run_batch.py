"""Batch processing runner for LLM evaluation and inference."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import sys
import argparse
import json

from loguru import logger

from .load_config import ConfigLoader
from .load_sample import SampleLoader
from .exec_inference import InferenceExecutor
from .exec_evaluation import EvaluationExecutor
from .output.output_saver import OutputSaver


class BatchRunner:
    """Orchestrates LLM batch evaluation and inference workflows."""

    def __init__(self, output_dir: str = "results", log_level: str = "INFO"):
        """Initialize pipeline with core dependencies."""
        # Setup logging first
        self._configure_logging(log_level)

        self.output_dir = Path(output_dir)
        self.output_saver = OutputSaver(output_dir)

        logger.info("Initialized batch processing pipeline")

    def _configure_logging(self, level: str = "INFO") -> None:
        """Configure loguru for console output."""
        logger.remove()
        logger.add(
            sys.stderr,
            level=level.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
        logger.info(f"Logging initialized: level={level}")

    def _start_session(
        self,
        task_name: str,
        model_name: str,
        template_name: str,
        dataset_name: str,
        timestamp: datetime,
        log_file_name: str = "session.log",
    ) -> str:
        """Start session with timestamp and logging."""
        # Start output saver session
        self.output_saver.start_session(
            task_name, model_name, template_name, dataset_name, timestamp
        )

        # Format timestamp and create log directory
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        log_dir = (
            self.output_dir
            / task_name
            / dataset_name
            / model_name
            / template_name
            / timestamp_str
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / log_file_name

        # Start session file logging
        self._session_handler_id = logger.add(
            str(log_file_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            mode="w",
        )
        logger.info(f"Session logging started: {log_file_path}")

        return str(log_file_path)

    def _stop_session(self) -> None:
        """Stop session logging."""
        if (
            hasattr(self, "_session_handler_id")
            and self._session_handler_id is not None
        ):
            logger.remove(self._session_handler_id)
            logger.info("Session logging stopped")
            self._session_handler_id = None

    def _infer_single(
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
    ) -> None:
        """Process a single experiment (dataset x template combination)."""
        # Run inference only
        dataset_config = task_configs["datasets"][dataset_name]
        template_class = task_configs["components"]["template_class"]

        # Load samples and generate responses
        samples = SampleLoader.load_dataset(dataset_config, dataset_name)

        if (
            checkpoint_interval
            and checkpoint_interval > 0
            and len(samples) > checkpoint_interval
        ):
            # Checkpoint mode: process in chunks
            logger.info(
                f"Checkpoint mode enabled: processing {len(samples)} samples in chunks of {checkpoint_interval}"
            )

            all_samples = []
            all_messages_batch = []
            all_outputs = []
            early_terminated = False

            for chunk_start in range(0, len(samples), checkpoint_interval):
                chunk_end = min(chunk_start + checkpoint_interval, len(samples))
                chunk_samples = samples[chunk_start:chunk_end]

                logger.info(
                    f"Processing chunk {chunk_start + 1}-{chunk_end}/{len(samples)}"
                )

                # Process chunk without saving
                _, chunk_messages, chunk_outputs, chunk_early_terminated = (
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

                # Accumulate results
                all_samples.extend(chunk_samples)
                all_messages_batch.extend(chunk_messages)
                all_outputs.extend(chunk_outputs)

                if chunk_early_terminated:
                    early_terminated = True

                # Save checkpoint (cumulative results)
                self.output_saver.save_raw_responses(
                    samples=all_samples,
                    messages_batch=all_messages_batch,
                    outputs=all_outputs,
                    model_config=model_config,
                    dataset_config=dataset_config,
                )

                logger.info(
                    f"Checkpoint saved: {len(all_samples)}/{len(samples)} samples completed"
                )

                if early_terminated:
                    logger.warning(
                        "Early termination detected, stopping chunk processing"
                    )
                    break

            # Final assignment for logging
            outputs_count = len(all_outputs)
        else:
            # Standard mode: process all at once
            samples, _, outputs, early_terminated = (
                inference_executor.generate_responses_for_samples(
                    template_class,
                    samples,
                    template_name,
                    dataset_config,
                    model_config,
                    save=True,
                    **kwargs,
                )
            )
            outputs_count = len(outputs)

        logger.info(f"Inference completed: {outputs_count} responses generated")

    def infer_batch(
        self,
        task_name: str,
        model_names: List[str],
        dataset_names: Optional[List[str]] = None,
        template_names: Optional[List[str]] = None,
        checkpoint_interval: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run batch inference pipeline only (no evaluation)."""
        # Load configurations (no metrics needed for inference)
        task_configs = ConfigLoader.load_task_configs(
            task_name, dataset_names, template_names, []
        )
        model_configs_with_types = ConfigLoader.load_model_configs(model_names)

        logger.info(
            f"[INFERENCE PHASE] Loaded task config and {len(model_configs_with_types)} model configs"
        )

        # Calculate total experiments for progress tracking
        total_experiments = (
            len(model_configs_with_types) * len(dataset_names) * len(template_names)
        )
        experiment_count = 0

        # Generate batch-wide common timestamp
        batch_timestamp = datetime.now()
        logger.info(
            f"[INFERENCE PHASE] Batch started at: {batch_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
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
                            f"[{experiment_count}/{total_experiments}] Starting inference: {task_name}/{model_name}/{dataset_name}/{template_name} (started at: {experiment_start_time.strftime('%H:%M:%S')})"
                        )

                        # Start session for this experiment with batch timestamp
                        self._start_session(
                            task_name,
                            model_name,
                            template_name,
                            dataset_name,
                            batch_timestamp,
                        )

                        # Process single experiment
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

                        logger.info(
                            f"[{experiment_count}/{total_experiments}] Completed inference: {task_name}/{model_name}/{dataset_name}/{template_name}"
                        )

                    finally:
                        # Stop session-specific logging
                        self._stop_session()

        # Batch inference completed
        logger.info(
            f"[INFERENCE PHASE] Batch completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return {
            "total_experiments": total_experiments,
            "batch_timestamp": batch_timestamp,
            "status": "inference_completed",
        }

    def evaluate_batch(
        self,
        task_name: str,
        model_names: List[str],
        dataset_names: List[str],
        template_names: List[str],
        metric_names: Optional[List[str]] = None,
        batch_timestamp: Optional[datetime] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run batch evaluation on specific batch timestamp experiments."""
        if batch_timestamp:
            timestamp_str = batch_timestamp.strftime("%Y%m%d_%H%M%S")
            logger.info(
                f"[EVALUATION PHASE] Starting evaluation for batch timestamp: {timestamp_str}"
            )
        else:
            logger.info("[EVALUATION PHASE] Starting evaluation using latest detection")

        logger.info(
            f"[EVALUATION PHASE] Processing {len(model_names)} models, {len(dataset_names)} datasets, {len(template_names)} templates"
        )

        evaluation_executor = EvaluationExecutor(self.output_saver)

        success_count = 0
        failed_experiments = []

        for model_name in model_names:
            for dataset_name in dataset_names:
                for template_name in template_names:
                    try:
                        # Determine target file and session timestamp
                        if batch_timestamp:
                            session_timestamp = batch_timestamp
                            timestamp_str = batch_timestamp.strftime("%Y%m%d_%H%M%S")
                            target_file = (
                                self.output_dir
                                / task_name
                                / dataset_name
                                / model_name
                                / template_name
                                / timestamp_str
                                / "raw_responses.json"
                            )

                            if not target_file.exists():
                                logger.warning(f"Target file not found: {target_file}")
                                failed_experiments.append(
                                    (
                                        model_name,
                                        dataset_name,
                                        template_name,
                                        "file_not_found",
                                    )
                                )
                                continue

                            logger.info(
                                f"Evaluating batch timestamp: {model_name}/{dataset_name}/{template_name} ({timestamp_str})"
                            )
                        else:
                            latest_result = self._find_latest_raw_responses(
                                task_name, dataset_name, model_name, template_name
                            )
                            if not latest_result:
                                logger.warning(
                                    f"No experiments found for {model_name}/{dataset_name}/{template_name}"
                                )
                                failed_experiments.append(
                                    (
                                        model_name,
                                        dataset_name,
                                        template_name,
                                        "no_experiments",
                                    )
                                )
                                continue

                            target_file = Path(latest_result[0])
                            session_timestamp = latest_result[1]
                            logger.info(
                                f"Evaluating latest: {model_name}/{dataset_name}/{template_name}"
                            )

                        # Load and evaluate the target file
                        with open(target_file, "r") as f:
                            data = json.load(f)

                        # Load task config for evaluation
                        task_configs = ConfigLoader.load_task_configs(
                            task_name, [dataset_name], None, metric_names
                        )

                        # Load raw responses and convert samples
                        sample_class = task_configs["components"]["sample_class"]
                        samples, _, outputs = SampleLoader.load_raw_responses(
                            str(target_file), sample_class
                        )

                        logger.info(
                            f"Evaluating {len(samples)} samples with {len(outputs)} outputs"
                        )

                        # Start session for saving evaluation results
                        metadata = data.get("metadata", {})
                        self.output_saver.start_session(
                            task_name=task_name,
                            model_name=metadata.get("model", model_name),
                            template_name=metadata.get("template", template_name),
                            dataset_name=dataset_name,
                            timestamp=session_timestamp,
                        )

                        # Evaluate outputs and save
                        evaluation_executor.evaluate_outputs(
                            outputs, samples, task_configs["metrics"], save=True
                        )
                        success_count += 1
                        logger.info("Evaluation completed and results saved")

                    except Exception as e:
                        logger.error(
                            f"Evaluation failed for {model_name}/{dataset_name}/{template_name}: {e}"
                        )
                        failed_experiments.append(
                            (model_name, dataset_name, template_name, str(e))
                        )

                    finally:
                        # Stop session-specific logging
                        self._stop_session()

        total_experiments = len(model_names) * len(dataset_names) * len(template_names)
        logger.info(
            f"[EVALUATION PHASE] Completed: {success_count}/{total_experiments} successful"
        )

        return {
            "total_experiments": total_experiments,
            "successful": success_count,
            "failed": len(failed_experiments),
            "failed_experiments": failed_experiments,
            "status": "evaluation_completed",
        }

    def _find_latest_raw_responses(
        self, task_name: str, dataset_name: str, model_name: str, template_name: str
    ) -> Optional[tuple[str, datetime]]:
        """Find the latest raw_responses.json for given experiment parameters.

        Returns:
            Tuple of (file_path, timestamp) if found, None otherwise
        """
        exp_dir = (
            self.output_dir / task_name / dataset_name / model_name / template_name
        )

        if not exp_dir.exists():
            return None

        # Find timestamp directories with raw_responses.json
        timestamp_dirs = []
        for item in exp_dir.iterdir():
            if item.is_dir() and len(item.name) == 15 and item.name[8] == "_":
                try:
                    datetime.strptime(item.name, "%Y%m%d_%H%M%S")
                    raw_file = item / "raw_responses.json"
                    if raw_file.exists():
                        timestamp_dirs.append((item.name, str(raw_file)))
                except ValueError:
                    continue

        if not timestamp_dirs:
            return None

        # Sort by timestamp (descending) and return latest
        timestamp_dirs.sort(key=lambda x: x[0], reverse=True)
        latest_timestamp_str, latest_file = timestamp_dirs[0]
        latest_timestamp = datetime.strptime(latest_timestamp_str, "%Y%m%d_%H%M%S")

        logger.debug(f"Latest experiment timestamp: {latest_timestamp_str}")
        return latest_file, latest_timestamp


def main():
    """CLI entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework - Batch Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Both inference and evaluation (default)
  uv run --env-file .env python -m llm_eval.run_batch --mode both --config configs/batch/test.yaml
  uv run --env-file .env python -m llm_eval.run_batch --config configs/batch/test.yaml  # same as above (both is default)
  
  # Inference only
  uv run --env-file .env python -m llm_eval.run_batch --mode infer --config configs/batch/test.yaml
  
  # Evaluation only (requires existing inference results)
  uv run --env-file .env python -m llm_eval.run_batch --mode evaluate --config configs/batch/test.yaml
  
  # Config with CLI overrides
  uv run --env-file .env python -m llm_eval.run_batch --mode both --config configs/batch/test.yaml --models plamo-2_0-prime
  
  # CLI only (no config file)
  uv run --env-file .env python -m llm_eval.run_batch --mode both --task medec --models gpt-4_1 --datasets small --templates paper_p1

Retry functionality:
  uv run --env-file .env python -m llm_eval.run_batch_with_retry --mode both --config configs/batch/test.yaml
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
        "--batch-timestamp",
        type=str,
        help="Batch timestamp for evaluate mode (YYYYMMDD_HHMMSS format)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N samples (default: 10)",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory path")
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

    runner = BatchRunner(output_dir=output_dir, log_level=args.log_level)

    logger.info(f"Starting {args.mode} mode")

    task_name = args.task or config.get("task")
    model_names = args.models or config.get("models")
    dataset_names = args.datasets or config.get("datasets")
    template_names = args.templates or config.get("templates")
    metric_names = args.metrics or config.get("metrics")
    checkpoint_interval = args.checkpoint_interval or config.get("checkpoint_interval")

    if not task_name or not model_names:
        raise ValueError(
            "--task and --models are required for evaluation/inference (can be provided via --config or CLI args)"
        )

    if args.mode == "infer":
        # Inference only mode
        logger.info("Starting inference-only batch...")
        result = runner.infer_batch(
            task_name=task_name,
            model_names=model_names,
            dataset_names=dataset_names,
            template_names=template_names,
            checkpoint_interval=checkpoint_interval,
        )

        logger.info(
            f"Inference mode completed: {result.get('total_experiments', 'N/A')} experiments finished"
        )

    elif args.mode == "evaluate":
        # Evaluation only mode (requires batch timestamp)
        if not args.batch_timestamp:
            raise ValueError(
                "--batch-timestamp is required for evaluate mode (format: YYYYMMDD_HHMMSS)"
            )

        # Parse batch timestamp
        try:
            batch_timestamp = datetime.strptime(args.batch_timestamp, "%Y%m%d_%H%M%S")
        except ValueError:
            raise ValueError(
                f"Invalid batch timestamp format: {args.batch_timestamp}. Expected format: YYYYMMDD_HHMMSS"
            )

        logger.info(
            f"Starting evaluation-only batch for timestamp: {args.batch_timestamp}"
        )
        evaluation_result = runner.evaluate_batch(
            task_name=task_name,
            model_names=model_names,
            dataset_names=dataset_names,
            template_names=template_names,
            metric_names=metric_names,
            batch_timestamp=batch_timestamp,
        )

        logger.info(
            f"Evaluation mode completed: {evaluation_result.get('successful', 'N/A')}/{evaluation_result.get('total_experiments', 'N/A')} evaluation experiments successful"
        )

    elif args.mode == "both":
        # Both inference and evaluation (2-phase processing)
        logger.info("Starting both inference and evaluation...")

        # Phase 1: Run inference batch
        logger.info("Phase 1: Starting inference batch...")
        inference_result = runner.infer_batch(
            task_name=task_name,
            model_names=model_names,
            dataset_names=dataset_names,
            template_names=template_names,
            checkpoint_interval=checkpoint_interval,
        )

        logger.info(
            f"Phase 1 completed: {inference_result.get('total_experiments', 'N/A')} inference experiments finished"
        )

        # Phase 2: Run evaluation batch
        logger.info("Phase 2: Starting evaluation batch...")
        evaluation_result = runner.evaluate_batch(
            task_name=task_name,
            model_names=model_names,
            dataset_names=dataset_names,
            template_names=template_names,
            metric_names=metric_names,
            batch_timestamp=inference_result.get("batch_timestamp"),
        )

        logger.info(
            f"Phase 2 completed: {evaluation_result.get('successful', 'N/A')}/{evaluation_result.get('total_experiments', 'N/A')} evaluation experiments successful"
        )
        logger.info("Both-mode batch processing completed successfully")


if __name__ == "__main__":
    main()
