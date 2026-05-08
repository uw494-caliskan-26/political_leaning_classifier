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

```bash
bash run.sh
```

This installs dependencies and launches a Gradio web UI at **http://localhost:7860**.

## Requirements

- Python 3.10+
- CUDA GPU with >= 6 GB VRAM (4-bit quantization is enabled by default)
- `pip` (dependencies are listed in `requirements.txt`)

## Configuration

Create a `.env` file (optional):

```
MODEL_PATH=/path/to/local/model
```

If `MODEL_PATH` is not set, the model is downloaded from Hugging Face Hub automatically.

## Web UI

```bash
python app.py
```

Paste any article or text, click **Classify**, and see the predicted label plus per-class probabilities.

## Batch Classification

Classify a CSV file (must have a `text` column):

```bash
python batch_classify.py --input articles.csv
```

Or pass strings directly:

```bash
python batch_classify.py --texts "First article text" "Second article text"
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
