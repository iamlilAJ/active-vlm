# Active Reasoning Vision-Language Models via Sequential Experimental Design

## Abstract

Visual perception in modern Vision-Language Models (VLMs) is constrained by a fundamental **perceptual bandwidth bottleneck**: a broad field-of-view inevitably sacrifices the fine-grained details necessary for complex reasoning.

We frame overcoming this limitation as a sequential decision-making process and formalise it through the lens of **Sequential Bayesian Optimal Experimental Design (S-BOED)**. While exact Bayesian inference is intractable in continuous gigapixel spaces, we derive principled yet tractable approximations that balance spatial coverage against resolution.

To validate this framework, we present a **training-free inference strategy** as a practical instantiation of the S-BOED objective for agents equipped with multiple vision tools. The strategy is designed as a flexible template that accommodates arbitrary optimisation algorithms — from efficient greedy sampling to look-ahead planning. Empirical evaluations on gigapixel-level benchmarks show that our approach significantly outperforms standard baselines and effectively narrows the gap towards human-annotated oracles.

## Code Overview

This repository contains the public implementation for config-driven active reasoning agents for vision-language benchmark evaluation. The code builds LangGraph workflows from YAML configs and runs variants that improve cropping decisions with sequential experimental design.

The main public entrypoint is:

```bash
uv run -m cv_agent.main_benchmark_multi <config.yaml>
```

Included Configs:
- `configs/boed-full.yaml`: primary full benchmark config using BOED cropping over MME, V*Bench, CV-Bench, and HR-Bench.
- `configs/lookahead-mme-remote-sensing.yaml`: look-ahead cropping policy on MME remote-sensing tasks.
- `configs/mcmc-mme-remote-sensing.yaml`: MCMC cropping policy on MME remote-sensing tasks.

## Setup

Install dependencies with `uv`:

```bash
uv sync --extra dev
```

Create a local env file from the template:

```bash
cp .env.example .env
```

Required runtime services:

- An OpenAI-compatible vision-language model endpoint.
- A cropping MCP server exposing `crop_image_tool_crop_image_post`.
- A MinIO/S3-compatible bucket for image hosting.
- Local MME and V*Bench files if running configs that reference those datasets.

Langfuse tracing is optional. If Langfuse credentials are not configured, the Langfuse client is disabled by the SDK.

## Optional Vision Tools

The public example configs keep only cropping enabled because OCR, detection, segmentation, and depth estimation require additional MCP service deployments. The code for those tools is still included and can be enabled in custom configs with tool names `ocr`, `detection`, `detection_small_object`, `segmentation`, and `depth_estimation` once you provide matching `server_url` values. The OCR wrapper defaults to `parse_pdf_file_parse_post`; set `remote_tool_name` in the tool parameters if your OCR MCP server exposes a different operation such as `ocr_image_with_mineru`.

## Environment

Core model and tool settings:

```bash
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
export OPENAI_API_KEY="your-api-key"
export CV_AGENT_MODEL="Qwen3-VL-30B-A3B-Instruct"
export CV_AGENT_CROPPING_MCP_URL="https://your-cropping-mcp-server/mcp"
```

Object storage:

```bash
export CV_AGENT_MINIO_ENDPOINT="host:port"
export CV_AGENT_MINIO_ACCESS_KEY="access-key"
export CV_AGENT_MINIO_SECRET_KEY="secret-key"
export CV_AGENT_MINIO_BUCKET="cv-agent"
export CV_AGENT_MINIO_SECURE="false"
```

Dataset paths:

```bash
export MME_DATA_PATH="/path/to/MME-RealWorld-Lite.json"
export MME_IMAGE_DIR="/path/to/mme/images"
export VSTAR_DATA_PATH="/path/to/vstar/test_questions.jsonl"
export VSTAR_IMAGE_DIR="/path/to/vstar/images"
```

Optional tracing:

```bash
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

These are all included in `.env.example`. You can run with `uv run --env-file .env -m ...` to set these environment variables easily.

## Running

Run the full BOED benchmark config:

```bash
uv run python -m cv_agent.main_benchmark_multi configs/boed-full.yaml
```

Run the MME remote-sensing variants:

```bash
uv run python -m cv_agent.main_benchmark_multi configs/lookahead-mme-remote-sensing.yaml
uv run python -m cv_agent.main_benchmark_multi configs/mcmc-mme-remote-sensing.yaml
```

Use `--set` to override config values without editing YAML:

```bash
uv run python -m cv_agent.main_benchmark_multi configs/boed-full.yaml \
  --set benchmark_concurrency=2 \
  --set benchmarks.mme.filtering.limit=10
```

Results are written under `results/multi/<timestamp>/`.

### Filtering options

```yaml
benchmarks:
  my_benchmark:
    dataset: mme
    concurrency: 2
    filtering:
      # Option 1: Specific indices
      indices: [10, 20, 30, 40, 50]
      # Option 2: Task patterns (glob-style wildcards)
      task_patterns:
        - 'Existence/*'
        - 'Counting/*'
      # Option 3: Sample N items per task (random)
      samples_per_task: 5
      # Option 4: Limit total samples
      limit: 100
      # Can combine multiple filters - they apply sequentially
```

**Filter Order:**
1. Start with `indices` (if provided) OR all indices
2. Filter by `task_patterns` (glob matching)
3. Sample `samples_per_task` (grouped by task_name)
4. Truncate with `limit`

## Development

Run local checks:

```bash
uv run ruff format
uv run ruff check .
uv run pytest -q
```

The tests are offline by default and validate config loading, registry wiring, pure policy helpers, answer grading, and storage env parsing.

## Citation

```bibtex
@inproceedings{liu2026activevlm,
  title     = {Active Reasoning Vision-Language Models via Sequential Experimental Design},
  author    = {Liu, Anjie and Gong, Ziqin and Song, Yan and Chen, Yuxiang and Liu, Xiaolong and Lu, Hengtong and Zhang, Kaike and Wei, Chen},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

This project is released under the MIT License. See `LICENSE`.
