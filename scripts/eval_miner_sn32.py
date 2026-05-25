#!/usr/bin/env python3
"""Evaluate miner DeBERTa model on ahmadreza13 dataset with SN32 validator reward."""

import argparse
import os
import sys

# Project root (parent of scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "neurons"))

os.environ.setdefault("USE_TF", "0")

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix

from detection.validator.reward import reward as sn32_reward
from miners.deberta_classifier import DebertaClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(ROOT, "datasets/ahmadreza13/data"),
    )
    parser.add_argument(
        "--foundation",
        type=str,
        default=os.path.join(ROOT, "models/deberta-v3-large-hf-weights"),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(ROOT, "models/deberta-large-ls03-ctx1024.pth"),
    )
    parser.add_argument("--batch-size", type=int, default=8, help="unused; deberta uses internal bs=4")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.shuffle(seed=args.seed).select(range(args.n_samples))
    texts = dataset["data"]
    labels = np.array(dataset["generated"], dtype=bool)

    print(f"Samples: {len(texts)} (human={np.sum(~labels)}, ai={np.sum(labels)})")
    print(f"Loading model on {args.device} ...")
    model = DebertaClassifier(
        foundation_model_path=args.foundation,
        model_path=args.weights,
        device=args.device,
    )

    print("Running inference ...")
    y_pred = np.array(model.predict_batch(texts), dtype=np.float64)

    sn32_score, metrics = sn32_reward(y_pred, labels)
    binary_preds = np.round(y_pred).astype(int)
    acc = accuracy_score(labels, binary_preds)
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()

    print("\n=== SN32 miner score (validator reward) ===")
    print(f"  sn32_reward (avg of F1, FP-score, AP): {sn32_score:.4f}")
    print(f"  f1_score:    {metrics['f1_score']:.4f}")
    print(f"  fp_score:    {metrics['fp_score']:.4f}  (1 - FP/n; higher = fewer human→AI errors)")
    print(f"  ap_score:    {metrics['ap_score']:.4f}")
    print(f"  accuracy:    {acc:.4f}")
    print(f"  confusion:   TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  (FP = human wrongly flagged as AI, FN = AI missed)")


if __name__ == "__main__":
    main()
