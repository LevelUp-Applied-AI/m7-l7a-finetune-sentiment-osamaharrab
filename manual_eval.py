"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    num_labels = getattr(getattr(model, "config", None), "num_labels", 0)
    if len(texts) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, num_labels), dtype=np.float32)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    was_training = model.training
    model.eval()

    pred_batches = []
    prob_batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            logits = model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1)
            batch_preds = torch.argmax(batch_probs, dim=-1)

            prob_batches.append(batch_probs.detach().cpu().numpy())
            pred_batches.append(batch_preds.detach().cpu().numpy())

    if was_training:
        model.train()

    return np.concatenate(pred_batches, axis=0), np.concatenate(prob_batches, axis=0)


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.

    Returns:
      {
        "accuracy": float,
        "macro_f1": float,
        "per_class": {label_index: {"precision": ..., "recall": ..., "f1": ...}, ...},
      }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    if y_true.size == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}

    labels = np.unique(np.concatenate([y_true, y_pred]))
    per_class = {}
    f1_scores = []

    for label in labels:
        true_is_label = y_true == label
        pred_is_label = y_pred == label

        tp = int(np.sum(true_is_label & pred_is_label))
        fp = int(np.sum(~true_is_label & pred_is_label))
        fn = int(np.sum(true_is_label & ~pred_is_label))

        precision_denominator = tp + fp
        recall_denominator = tp + fn
        precision = tp / precision_denominator if precision_denominator else 0.0
        recall = tp / recall_denominator if recall_denominator else 0.0
        f1_denominator = precision + recall
        f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0

        label_index = int(label)
        per_class[label_index] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        f1_scores.append(f1)

    accuracy = float(np.sum(y_true == y_pred) / y_true.size)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
