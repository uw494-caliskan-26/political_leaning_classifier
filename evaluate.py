"""Evaluate classifier on a labeled CSV (text + label columns).

Based on batch_classify.py: loads the same model, runs classify_batch, records
per-row predictions, then computes confusion matrix (integer labels 0–4) and MAE.

Usage:
    python evaluate.py
    python evaluate.py --input data/subset_balanced.csv
    python evaluate.py --no-quantize
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from classifier import load_model, classify_batch, LABEL_MAP

RESULTS_DIR = Path("results")
DEFAULT_INPUT = Path("data/subset_balanced.csv")


def confusion_matrix_counts(y_true: list[int], y_pred: list[int], n_classes: int = 5) -> list[list[int]]:
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
    return cm


def main():
    parser = argparse.ArgumentParser(description="Evaluate political-leaning classifier on labeled CSV")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="CSV with 'text' and 'label' columns (0–4)",
    )
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization (use bf16)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    eval_df = df.dropna(subset=["text", "label"]).copy()
    dropped = len(df) - len(eval_df)
    if dropped:
        print(f"Warning: dropped {dropped} row(s) with missing text or label.")

    texts = eval_df["text"].astype(str).tolist()
    y_true = eval_df["label"].astype(int).tolist()

    print(f"Evaluating {len(texts)} row(s) from {input_path}...")
    model, tokenizer, number_token_ids = load_model(quantize_4bit=not args.no_quantize)
    results = classify_batch(texts, model, tokenizer, number_token_ids)

    pred_scores = [int(r["score"]) for r in results]
    pred_labels = [r["label"] for r in results]

    rows = []
    for (_, row), res, ps, pl in zip(eval_df.iterrows(), results, pred_scores, pred_labels):
        r = {
            "row_index": row.name,
            "text": row["text"],
            "label_true": int(row["label"]),
            "label_pred": ps,
            "label_pred_name": pl,
            "score": res["score"],
            "weighted_score": res["weighted_score"],
        }
        for j in range(5):
            r[f"prob_{LABEL_MAP[j]}"] = res["probabilities"][LABEL_MAP[j]]
        rows.append(r)

    out_pred = pd.DataFrame(rows)
    mae = (out_pred["label_true"] - out_pred["label_pred"]).abs().mean()
    cm = confusion_matrix_counts(y_true, pred_scores, n_classes=5)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = RESULTS_DIR / f"{ts}_evaluation_predictions.csv"
    cm_path = RESULTS_DIR / f"{ts}_evaluation_confusion_matrix.csv"
    metrics_path = RESULTS_DIR / f"{ts}_evaluation_metrics.json"

    out_pred.to_csv(pred_path, index=False)

    cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in range(5)], columns=[f"pred_{j}" for j in range(5)])
    cm_df.to_csv(cm_path)

    metrics = {
        "input": str(input_path.resolve()),
        "n_rows": len(texts),
        "n_dropped_missing": dropped,
        "mae_integer_pred": float(mae),
        "confusion_matrix": cm,
        "confusion_matrix_note": "rows are true label 0–4, columns are predicted label 0–4",
        "label_names": {str(i): LABEL_MAP[i] for i in range(5)},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Predictions saved to {pred_path}")
    print(f"Confusion matrix saved to {cm_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"MAE (integer prediction): {mae:.4f}")
    print(cm_df.to_string())


if __name__ == "__main__":
    main()
