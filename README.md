# Political Leaning Classifier

Local web UI and batch script for [zhezhou1106/political-leaning-classifier-v2](https://huggingface.co/zhezhou1106/political-leaning-classifier-v2) — a Qwen3.5-4B fine-tune that scores English text on a 5-point political spectrum.

| Score | Label      |
|-------|------------|
| 0     | Far Left   |
| 1     | Lean Left  |
| 2     | Center     |
| 3     | Lean Right |
| 4     | Far Right  |

## Quick Start (one command)

Install [uv](https://docs.astral.sh/uv/) if you do not have it, then:

```bash
bash run.sh
```

This runs `uv sync` (creates or updates `.venv` from `uv.lock`) and launches the Gradio web UI at **http://localhost:7860**.

Equivalent manual steps:

```bash
uv sync
uv run python app.py
```

## Requirements

- Python 3.13+ (see `requires-python` in `pyproject.toml`; `.python-version` pins the team default)
- [uv](https://docs.astral.sh/uv/) for installs and runs
- CUDA GPU with >= 6 GB VRAM (4-bit quantization is enabled by default)

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock` (commit both when you change deps).

## Configuration

Create a `.env` file (optional):

```
MODEL_PATH=/path/to/local/model
```

If `MODEL_PATH` is not set, the model is downloaded from Hugging Face Hub automatically.

## Web UI

```bash
uv run python app.py
```

Paste any article or text, click **Classify**, and see the predicted label plus per-class probabilities.

## Batch Classification

Classify a CSV file (must have a `text` column):

```bash
uv run python batch_classify.py --input articles.csv
```

Or pass strings directly:

```bash
uv run python batch_classify.py --texts "First article text" "Second article text"
```

Results are saved to `results/<timestamp>_results.csv`.

### Options

| Flag            | Description                          |
|-----------------|--------------------------------------|
| `--input FILE`  | CSV with a `text` column             |
| `--texts ...`   | One or more quoted strings           |
| `--no-quantize` | Use bf16 instead of 4-bit quantization |

## License

Model weights are Apache-2.0 (see upstream). This wrapper code is MIT.
