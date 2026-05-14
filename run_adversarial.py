"""
Adversarial Evaluation.

Load a fine-tuned classifier, run it against adversarial_set.csv, and write
results.csv. Read label names from model.config.id2label — do not hard-code.
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or HF Hub id.

    Defaults to local 'model' (your Lab 7A checkpoint). CI overrides via MODEL_PATH env.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path. Return a DataFrame with all
    original columns plus predicted_label, predicted_probability, correct.

    Read label names from model.config.id2label — do not hard-code class names.
    """
    df = pd.read_csv(adv_csv_path)
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    encoded = tokenizer(
        df["text"].astype(str).tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits

    probabilities = torch.softmax(logits, dim=-1)
    predicted_ids = torch.argmax(probabilities, dim=-1).tolist()
    predicted_labels = [id2label[predicted_id] for predicted_id in predicted_ids]
    predicted_probabilities = [
        float(probabilities[row_idx, predicted_id].cpu().item())
        for row_idx, predicted_id in enumerate(predicted_ids)
    ]

    results = df.copy()
    results["predicted_label"] = predicted_labels
    results["predicted_probability"] = predicted_probabilities
    results["correct"] = [
        _normalize_label(predicted) == _normalize_label(expected)
        for predicted, expected in zip(predicted_labels, results["expected_label"])
    ]
    return results


def _normalize_label(label) -> str:
    """Normalize labels from different model configs for comparison."""
    return str(label).strip().lower()


def main() -> None:
    """Orchestrate; write results.csv."""
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")


if __name__ == "__main__":
    main()
