"""Batch classification script.

Usage:
    python batch_classify.py --input data.csv          # CSV with a 'text' column
    python batch_classify.py --texts "sentence 1" "sentence 2"
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from classifier import load_model, classify_batch, LABEL_MAP

RESULTS_DIR = Path("results")


def main():
    parser = argparse.ArgumentParser(description="Batch political-leaning classification")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Path to a CSV file with a 'text' column")
    group.add_argument("--texts", nargs="+", type=str, help="One or more text strings to classify")
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization (use bf16)")
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input)
        if "text" not in df.columns:
            raise ValueError("CSV must contain a 'text' column.")
        texts = df["text"].dropna().tolist()
    else:
        texts = args.texts

    print(f"Classifying {len(texts)} text(s)...")
    model, tokenizer, number_token_ids = load_model(quantize_4bit=not args.no_quantize)
    results = classify_batch(texts, model, tokenizer, number_token_ids)

    rows = []
    for text, res in zip(texts, results):
        row = {
            "text": text,
            "score": res["score"],
            "label": res["label"],
        }
        for i in range(5):
            row[f"prob_{LABEL_MAP[i]}"] = res["probabilities"][LABEL_MAP[i]]
        rows.append(row)

    out_df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{timestamp}_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")
    print(out_df[["text", "score", "label"]].to_string(index=False))


if __name__ == "__main__":
    main()
