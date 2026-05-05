"""
Multi-benchmark runner for CV Agent with advanced filtering and CLI overrides.

Supports running multiple benchmarks sequentially from a single config file,
with task pattern matching, per-task sampling, and config overrides via CLI.
"""

import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import anyio
import orjson
import structlog
import typer
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from langfuse import get_client, propagate_attributes
from langgraph.pregel.main import RunnableConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from cv_agent.benchmark_loaders import get_dataset_loader
from cv_agent.benchmark_loaders.base import BaseDatasetLoader
from cv_agent.core.builder import GraphBuilder
from cv_agent.utils.config import get_invocation_config
from cv_agent.utils.grader import Grade, grade_answer
from cv_agent.utils.logging import setup_logging
from cv_agent.utils.storage import upload_pil_image_to_minio

logger = structlog.get_logger()


# ============================================================================
# CLI Override Parser
# ============================================================================


def _parse_value(value: str) -> Any:
    """Parse string value to appropriate Python type.

    Supports:
    - JSON arrays: [1,2,3] → list
    - Integers: 42 → int
    - Floats: 3.14 → float
    - Booleans: true/false → bool
    - Strings: everything else
    """
    value = value.strip()

    # Try JSON array
    if value.startswith("[") and value.endswith("]"):
        try:
            return orjson.loads(value)
        except orjson.JSONDecodeError:
            pass

    # Try boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Default to string
    return value


def apply_cli_overrides(cfg: DictConfig, override_args: list[str]) -> None:
    """Apply CLI overrides directly to the config using OmegaConf's path-based updates.

    This approach avoids the list/dict merge conflict by applying overrides
    to the already-loaded config structure.

    Args:
        cfg: Loaded OmegaConf config
        override_args: List of 'key.path=value' strings
    """
    for arg in override_args:
        if "=" not in arg:
            raise ValueError(f"Invalid override format: {arg}. Expected 'key=value'")

        key_path, value_str = arg.split("=", 1)
        parsed_value = _parse_value(value_str)

        # Use OmegaConf.update to handle both dicts and lists correctly
        OmegaConf.update(cfg, key_path, parsed_value, merge=True)
        logger.info("config_overriden", key=key_path, value=parsed_value)


# ============================================================================
# Answer Extraction (from anyio version)
# ============================================================================


def extract_answer(message_content: str) -> str:
    """Extracts content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", message_content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: remove thought traces
    thought_action_pattern = r"Thought:.*"
    cleaned_content = re.sub(thought_action_pattern, "", message_content, flags=re.DOTALL).strip()
    if not cleaned_content:
        return "Error: Agent finished without a final answer tag."
    return cleaned_content


# ============================================================================
# Benchmark Tracer (from anyio version)
# ============================================================================


class BenchmarkTracer:
    """
    Encapsulates Langfuse tracing logic for a single benchmark sample.
    """

    def __init__(
        self, client: Any, sample_id: str, task_name: str, question: str, correct_answer: str
    ):
        self.client = client
        self.sample_id = sample_id
        self.task_name = task_name
        self.question = question
        self.correct_answer = correct_answer
        self.span = None
        self._cm = None

    def __enter__(self):
        # Start a trace span for this specific sample
        trace_id = self.client.create_trace_id()
        structlog.contextvars.bind_contextvars(trace_id=trace_id)

        self._cm = self.client.start_as_current_observation(
            name=f"{self.sample_id}_{self.task_name}",
            as_type="span",
            input={"question": self.question},
            trace_context={"trace_id": trace_id},
        )
        self.span = self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cm:
            if exc_type and self.span:
                self.span.update(level="ERROR", status_message=str(exc_val))
            return self._cm.__exit__(exc_type, exc_val, exc_tb)

    def score(self, agent_answer: str) -> str:
        """Grades the answer and records the score in Langfuse."""
        if not self.span:
            return "Error: No active span"

        grade = grade_answer(agent_answer, self.correct_answer)

        self.span.score(
            name="correctness",
            value=grade.name,
            data_type="CATEGORICAL",
            comment=f"Agent: {agent_answer} | Correct: {self.correct_answer}",
        )

        self.span.update(
            output={"agent_answer": agent_answer},
            metadata={
                "task_name": self.task_name,
                "sample_id": self.sample_id,
                "correct_answer": self.correct_answer,
                "agent_answer": agent_answer,
            },
        )
        return grade.name


# ============================================================================
# Agent Execution (from anyio version)
# ============================================================================


async def run_agent_sample(
    question: str,
    initial_image: Image.Image,
    agent_executor,
    sample_id: str,
    invoke_config: dict,
    state_overrides: dict,
    turn_info: str = "Sample",
) -> tuple[str, dict]:
    """
    Runs the CV agent for a single question and image.
    """
    try:
        # Offload blocking I/O to thread
        public_minio_url, img_width, img_height = await anyio.to_thread.run_sync(  # type: ignore[attr-defined]
            upload_pil_image_to_minio, initial_image, sample_id
        )
        if not public_minio_url:
            raise ValueError("Failed to upload initial image to Minio.")
    except Exception as e:
        logger.exception("image_upload_failed", turn_info=turn_info)
        raise ValueError(f"Failed to upload image: {e}") from e

    initial_state = {
        "question": question,
        "original_figure_url": public_minio_url,
        "current_turn": 1,
        "max_turns": 10,
        "messages": [],
        "image_dimensions": {public_minio_url: (img_width, img_height)},
        "prefix": sample_id,
        "url_map": {public_minio_url: public_minio_url},
        "tool_usage": {},
    }
    if state_overrides:
        initial_state.update(state_overrides)

    final_answer = "Error: No final answer found."
    final_tool_usage = {}

    try:
        final_state = await agent_executor.ainvoke(initial_state, config=invoke_config)

        if "tool_usage" in final_state:
            final_tool_usage = final_state["tool_usage"]

        if "messages" in final_state and final_state["messages"]:
            final_content = final_state["messages"][-1].content
            final_answer = extract_answer(final_content)
        else:
            logger.warning("final_messages_not_found", turn_info=turn_info)

    except Exception as e:
        logger.exception("agent_run_failed", turn_info=turn_info)
        return f"Error: Agent run failed: {e}", {}

    return final_answer, final_tool_usage


async def process_sample(
    idx: int,
    dataset_loader: BaseDatasetLoader,
    agent_executor,
    invoke_config: dict,
    agent_config_dict: dict,
    langfuse_client: Any,
    results_list: list,
    intermediate_output: anyio.Path,
    limiter: anyio.CapacityLimiter,
    bench_name: str,
):
    """Processes a single sample with concurrency limit and lazy loading."""
    async with limiter:
        # Load sample lazily (inside limiter block)
        try:
            sample_data = dataset_loader[idx]
        except Exception:
            logger.exception("sample_load_failed", bench_name=bench_name, index=idx)
            results_list.append(
                {
                    "index": idx,
                    "task": "unknown",
                    "question": "unknown",
                    "correct_answer": "unknown",
                    "agent_answer": "Error: Failed to load sample",
                }
            )
            return

        question = sample_data["question"]
        correct_answer = sample_data["correct_answer"]
        initial_image = sample_data["image"]
        task_name = sample_data["task_name"]
        sample_id = sample_data["sample_id"]

        turn_info = f"[{bench_name}] ID {idx} | {sample_id}"

        result_data = {
            "index": idx,
            "task": task_name,
            "question": question,
            "correct_answer": correct_answer,
            "agent_answer": "Error: Skipped",
        }

        try:
            tracer = BenchmarkTracer(
                langfuse_client, sample_id, task_name, question, correct_answer
            )
            with tracer:
                with propagate_attributes(tags=[task_name, sample_id, "multi", bench_name]):
                    cvagent_answer, tool_usage_stats = await run_agent_sample(
                        question=question,
                        initial_image=initial_image,
                        agent_executor=agent_executor,
                        sample_id=sample_id,
                        invoke_config=invoke_config,
                        state_overrides=agent_config_dict,
                        turn_info=turn_info,
                    )

                tracer.score(cvagent_answer)

            result_data["agent_answer"] = cvagent_answer
            result_data["tool_usage"] = tool_usage_stats

            logger.info(
                "answer_received", turn_info=turn_info, pred=cvagent_answer, label=correct_answer
            )

        except Exception as e:
            logger.exception("sample_run_failed", turn_info=turn_info)
            result_data["agent_answer"] = f"Error: {e}"

        results_list.append(result_data)
        async with await intermediate_output.open("ab") as fd:
            await fd.write(orjson.dumps(result_data) + b"\n")


# ============================================================================
# Filtering Engine
# ============================================================================


def apply_filters(dataset_loader: BaseDatasetLoader, filtering_config: dict) -> list[int]:
    """Apply filtering rules sequentially to get final list of sample indices.

    Args:
        dataset_loader: Dataset loader instance
        filtering_config: Dict with optional keys:
            - indices: list[int] - Specific indices to run
            - task_patterns: list[str] - Glob patterns like ['Existence/*']
            - samples_per_task: int - Random sample N per task
            - limit: int - Max total samples

    Order of operations:
    1. Start with indices if provided, else all
    2. Filter by task_patterns (glob matching)
    3. Sample per task (grouped by task_name)
    4. Apply limit (truncate)

    Returns:
        Final list of sample indices to run
    """
    rng = random.Random(int(os.getenv("CV_AGENT_SEED", "0")))

    # Step 1: Initial index set
    indices = filtering_config.get("indices")
    if indices is not None:
        working_indices = indices
        logger.info("samples_filtered", by="working_indices", count=len(working_indices))
    else:
        working_indices = list(range(len(dataset_loader)))

    # Step 2: Filter by task patterns
    task_patterns = filtering_config.get("task_patterns")
    if task_patterns:
        filtered = []
        for idx in working_indices:
            try:
                task_name = dataset_loader.get_metadata(idx)["task_name"]
                if any(fnmatch(task_name, pattern) for pattern in task_patterns):
                    filtered.append(idx)
            except Exception as e:
                logger.warning(
                    "sample_load_failed", idx=idx, error=str(e), context="task_pattern_filter"
                )
        working_indices = filtered
        logger.info(
            "samples_filtered",
            by="task_pattern",
            count=len(working_indices),
            patterns=task_patterns,
        )

    # Step 3: Sample per task
    samples_per_task = filtering_config.get("samples_per_task")
    if samples_per_task:
        task_to_indices = defaultdict(list)
        for idx in working_indices:
            try:
                task_name = dataset_loader.get_metadata(idx)["task_name"]
                task_to_indices[task_name].append(idx)
            except Exception as e:
                logger.warning(
                    "sample_load_failed", idx=idx, error=str(e), context="samples_per_task_filter"
                )

        sampled = []
        for _, task_indices in task_to_indices.items():
            if len(task_indices) > samples_per_task:
                sampled.extend(rng.sample(task_indices, samples_per_task))
            else:
                sampled.extend(task_indices)
        working_indices = sampled
        logger.info(
            "samples_filtered",
            by="samples_per_task",
            count=len(working_indices),
            samples_per_task=samples_per_task,
            num_tasks=len(task_to_indices),
        )

    # Step 4: Apply limit
    limit = filtering_config.get("limit")
    if limit:
        working_indices = working_indices[:limit]
        logger.info("samples_filtered", by="limit", count=len(working_indices), limit=limit)

    if not working_indices:
        logger.warning("no_samples_after_filter")

    return working_indices


# ============================================================================
# Stats Computation
# ============================================================================


def compute_stats(results: list[dict]) -> dict[str, Any]:
    """Compute correctness statistics from results."""
    stats: dict[str, Any] = {
        "total_samples": len(results),
        "correct": 0,
        "wrong": 0,
        "no_answer": 0,
        "error": 0,
    }

    for result in results:
        agent_answer = result.get("agent_answer", "")
        correct_answer = result.get("correct_answer", "")

        grade = grade_answer(agent_answer, correct_answer)

        if grade == Grade.CORRECT:
            stats["correct"] += 1
        elif grade == Grade.WRONG:
            stats["wrong"] += 1
        elif grade == Grade.NO_ANSWER:
            stats["no_answer"] += 1
        elif grade == Grade.ERROR:
            stats["error"] += 1

    if stats["total_samples"] > 0:
        stats["accuracy"] = stats["correct"] / stats["total_samples"]
    else:
        stats["accuracy"] = 0.0

    return stats


# ============================================================================
# Producer-Consumer Helpers
# ============================================================================


async def _produce_indices(
    send_stream: MemoryObjectSendStream[int],
    indices: list[int],
) -> None:
    """Feed sample indices through stream for lazy processing."""
    async with send_stream:
        for idx in indices:
            await send_stream.send(idx)


async def _consume_and_process(
    receive_stream: MemoryObjectReceiveStream[int],
    dataset_loader: BaseDatasetLoader,
    agent_executor,
    invoke_config: RunnableConfig,
    agent_config_dict: dict,
    langfuse_client: Any,
    results: list,
    intermediate_output: anyio.Path,
    limiter: anyio.CapacityLimiter,
    bench_name: str,
) -> None:
    """Receive indices from stream and spawn workers to process them."""
    async with receive_stream, anyio.create_task_group() as tg:
        async for idx in receive_stream:
            tg.start_soon(  # type: ignore[arg-type]
                process_sample,
                idx,
                dataset_loader,
                agent_executor,
                invoke_config,
                agent_config_dict,
                langfuse_client,
                results,
                intermediate_output,
                limiter,
                bench_name,
            )


# ============================================================================
# Single Benchmark Runner
# ============================================================================


async def run_single_benchmark(
    bench_name: str,
    bench_config: DictConfig,
    base_config: DictConfig,
    base_results_dir: anyio.Path,
) -> dict[str, Any]:
    """Execute one benchmark run.

    Returns:
        Summary dict with stats
    """
    start_time = time.time()

    logger.info("benchmark_start", name=bench_name, dataset=bench_config.get("dataset"))
    bench_results_dir = base_results_dir / bench_name
    await bench_results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge config
    merged_config = OmegaConf.merge(base_config, bench_config)
    assert isinstance(merged_config, DictConfig), "Merged config must be DictConfig"

    # Step 2: Build agent executor
    builder = GraphBuilder(merged_config)
    agent_executor = builder.build()
    logger.info("agent_executor_built", bench_name=bench_name)

    # Step 3: Get invocation config
    invoke_config = get_invocation_config(merged_config)
    agent_config_dict = (
        OmegaConf.to_container(merged_config.agent_config, resolve=True)
        if "agent_config" in merged_config
        else {}
    )

    # Step 4: Load dataset
    dataset_name = bench_config.dataset

    # Extract loader arguments if provided (e.g., data_path, image_dir for MME/VStarLoader)
    loader_args: dict[str, Any] = {}
    if "loader_args" in bench_config:
        loader_args_raw = OmegaConf.to_container(bench_config.loader_args, resolve=True)
        if isinstance(loader_args_raw, dict):
            loader_args = loader_args_raw  # type: ignore[assignment]
            logger.info("custom_loader_args", bench_name=bench_name, loader_args=loader_args)

    dataset_loader = get_dataset_loader(dataset_name, **loader_args)
    logger.info(
        "dataset_loaded", bench_name=bench_name, dataset=dataset_name, size=len(dataset_loader)
    )

    # Step 5: Apply filters
    filtering_config = bench_config.get("filtering", {})
    filtering_dict: dict[str, Any] = {}
    if filtering_config:
        filtering_dict = OmegaConf.to_container(filtering_config, resolve=True)  # type: ignore[assignment]
    all_indices = apply_filters(dataset_loader, filtering_dict)

    if not all_indices:
        logger.error("no_samples_after_filter", bench_name=bench_name)
        return {
            "name": bench_name,
            "dataset": dataset_name,
            "total_samples": 0,
            "correct": 0,
            "wrong": 0,
            "no_answer": 0,
            "error": 0,
            "time_elapsed": 0,
            "accuracy": 0.0,
        }

    # Step 6: Run samples with producer-consumer pattern
    concurrency = bench_config.get("concurrency", 2)
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    limiter = anyio.CapacityLimiter(concurrency)
    langfuse_client = get_client()
    results = []

    # Create bounded stream to prevent loading all samples upfront
    # Buffer size decoupled from concurrency: allows small queue without hoarding
    max_buffer_size = min(concurrency * 2, 10)
    send_stream, receive_stream = anyio.create_memory_object_stream[int](
        max_buffer_size=max_buffer_size
    )

    async with anyio.create_task_group() as tg:
        # Producer: feeds indices as capacity becomes available
        tg.start_soon(_produce_indices, send_stream, all_indices)

        # Consumer: spawns workers that load samples lazily
        tg.start_soon(  # type: ignore[arg-type]
            _consume_and_process,
            receive_stream,
            dataset_loader,
            agent_executor,
            invoke_config,
            agent_config_dict,
            langfuse_client,
            results,
            bench_results_dir / "intermediate.jsonl",
            limiter,
            bench_name,
        )

    # Flush traces
    langfuse_client.flush()

    # Step 7: Save results
    results_file = bench_results_dir / "results.json"
    await results_file.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))

    logger.info("results_saved", bench_name=bench_name, path=str(results_file))

    # Step 8: Compute stats
    stats = compute_stats(results)
    stats["name"] = bench_name
    stats["dataset"] = dataset_name
    stats["time_elapsed"] = time.time() - start_time

    logger.info("benchmark_completed", bench_name=bench_name, **stats)

    return stats


# ============================================================================
# Multi-Benchmark Orchestrator
# ============================================================================


async def run_multi_benchmark(cfg: DictConfig, cli_override_args: list[str]) -> None:
    """Top-level orchestrator for running multiple benchmarks."""

    # Step 1: Apply CLI overrides directly to the config
    if cli_override_args:
        apply_cli_overrides(cfg, cli_override_args)

    # Step 2: Setup logging
    if "logging" in cfg:
        setup_logging(cfg.logging)

    # Step 3: Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = anyio.Path(__file__).parent.parent.parent
    results_dir = base_dir / "results" / "multi" / timestamp
    await results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("multi_benchmark_start", results_dir=str(results_dir), timestamp=timestamp)

    # Step 4: Validate benchmarks section
    if "benchmarks" not in cfg:
        raise ValueError("Config must contain 'benchmarks' section")

    benchmarks = cfg.benchmarks
    if not benchmarks:
        raise ValueError("'benchmarks' section is empty")

    # Step 5: Run each benchmark
    summaries = []
    for bench_name, bench_config in benchmarks.items():
        logger.info("benchmark_started", bench_name=bench_name)

        try:
            summary = await run_single_benchmark(
                bench_name=bench_name,
                bench_config=bench_config,
                base_config=cfg,
                base_results_dir=results_dir,
            )
            summaries.append(summary)
        except Exception as e:
            logger.error("benchmark_failed", bench_name=bench_name, error=str(e))
            logger.exception("benchmark_failed")
            # Continue to next benchmark
            summaries.append(
                {
                    "name": bench_name,
                    "dataset": bench_config.get("dataset", "unknown"),
                    "total_samples": 0,
                    "correct": 0,
                    "wrong": 0,
                    "no_answer": 0,
                    "error": 0,
                    "time_elapsed": 0,
                    "accuracy": 0.0,
                    "status": "failed",
                }
            )

    # Step 6: Compute aggregate stats
    aggregate: dict[str, Any] = {
        "total_samples": sum(s["total_samples"] for s in summaries),
        "correct": sum(s["correct"] for s in summaries),
        "wrong": sum(s["wrong"] for s in summaries),
        "no_answer": sum(s["no_answer"] for s in summaries),
        "error": sum(s["error"] for s in summaries),
    }

    if aggregate["total_samples"] > 0:
        aggregate["accuracy"] = aggregate["correct"] / aggregate["total_samples"]
    else:
        aggregate["accuracy"] = 0.0

    # Step 7: Save summary
    summary_data = {
        "timestamp": timestamp,
        "total_benchmarks": len(summaries),
        "benchmarks": summaries,
        "aggregate": aggregate,
    }

    summary_file = results_dir / "summary.json"
    await summary_file.write_bytes(orjson.dumps(summary_data, option=orjson.OPT_INDENT_2))

    logger.info("benchmark_summary", **aggregate, path=f"{summary_file}")


# ============================================================================
# CLI Entry Point
# ============================================================================

app = typer.Typer()


@app.command()
def main(
    config: Path = typer.Argument(..., help="Path to multi-benchmark config YAML"),  # noqa: B008
    set: list[str] = typer.Option([], help="Override config values (e.g., --set key=value)"),  # noqa: B008
):
    """Run multi-benchmark evaluation with advanced filtering."""

    # Load config
    cfg = OmegaConf.load(config.resolve())
    assert isinstance(cfg, DictConfig), "Config must be a valid OmegaConf DictConfig"

    # Run with CLI override args (applied inside run_multi_benchmark)
    anyio.run(run_multi_benchmark, cfg, set)


if __name__ == "__main__":
    app()
