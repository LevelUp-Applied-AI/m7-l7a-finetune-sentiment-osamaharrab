"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    probs, y_true = _validate_inputs(probs, y_true, n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_centers = (edges[:-1] + edges[1:]) / 2.0
    bucket_accuracies = np.zeros(n_bins, dtype=float)
    bucket_counts = np.zeros(n_bins, dtype=int)

    if probs.shape[0] == 0:
        return bucket_centers, bucket_accuracies, bucket_counts

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = predictions == y_true
    bucket_ids = _bucket_indices(confidences, edges)

    for bucket_id in range(n_bins):
        in_bucket = bucket_ids == bucket_id
        bucket_counts[bucket_id] = int(np.sum(in_bucket))
        if bucket_counts[bucket_id] > 0:
            bucket_accuracies[bucket_id] = float(np.mean(correct[in_bucket]))

    return bucket_centers, bucket_accuracies, bucket_counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    probs, y_true = _validate_inputs(probs, y_true, n_bins)
    n_examples = probs.shape[0]
    if n_examples == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = predictions == y_true
    bucket_ids = _bucket_indices(confidences, edges)

    ece = 0.0
    for bucket_id in range(n_bins):
        in_bucket = bucket_ids == bucket_id
        bucket_count = int(np.sum(in_bucket))
        if bucket_count == 0:
            continue

        bucket_accuracy = float(np.mean(correct[in_bucket]))
        bucket_confidence = float(np.mean(confidences[in_bucket]))
        ece += (bucket_count / n_examples) * abs(bucket_accuracy - bucket_confidence)

    return float(ece)


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _validate_inputs(probs: np.ndarray, y_true: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)

    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array of shape (N, num_classes)")
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("probs and y_true must have the same number of rows")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probs must contain only finite values")
    if probs.shape[0] > 0:
        confidences = np.max(probs, axis=1)
        if np.any((confidences < 0.0) | (confidences > 1.0)):
            raise ValueError("max predicted probabilities must be between 0 and 1")

    return probs, y_true


def _bucket_indices(confidences: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bucket_ids = np.searchsorted(edges, confidences, side="right") - 1
    return np.clip(bucket_ids, 0, len(edges) - 2)
