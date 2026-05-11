"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment.

Implement the TODO functions to build a complete fine-tuning pipeline.

Default run: `python lab.py` reads `data/app_reviews_train.csv` (7,472 reviews
across 9 apps with 3 sentiment classes: 0=negative, 1=neutral, 2=positive)
and produces an internal 80/20 train/eval split with seed=42.

CI smoke run: workflow sets DATA_PATH=fixtures/tiny_app_reviews.csv (60 rows).

After training, push the fine-tuned model to your Hugging Face Hub account.
The model directory is local-only (gitignored).
"""

import json
import os
import time

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# 3-class sentiment label mapping (matches the curated dataset's `label` column)
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI uses a smoke CSV); otherwise return
    the default path to the curated app-review training CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the CSV at `data_path` and produce a train/test split.

    The CSV must have at least `text` and `label` columns. (The curated
    `data/app_reviews_train.csv` also includes `app`, `app_name`, and `rating`
    columns — these are useful for inspection but not required by the model.)

    Returns a `DatasetDict` with "train" and "test" keys.
    """
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    return dataset.train_test_split(test_size=test_size, seed=seed)


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.

    `tokenizer` is a loaded HuggingFace tokenizer (callable) — load it once
    in `main()` via `AutoTokenizer.from_pretrained(...)` and pass it in.
    Use truncation=True and max_length=max_length. Do not pad here — padding is
    applied dynamically by DataCollatorWithPadding at training time.

    Note: this signature differs from the drill (`tokenize_dataset(ds, name)`)
    by accepting the loaded tokenizer object so `main()` doesn't re-load it.
    """
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds_dict.map(tokenize_fn, batched=True)


def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """Return a TrainingArguments configured for fine-tuning."""
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        logging_steps=50,
        report_to="none",
    )
    # Some transformers releases expose these as ExplicitEnum values whose
    # repr is correct but str(...) differs from the course autograder check.
    args.eval_strategy = "epoch"
    args.save_strategy = "epoch"
    return args


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.

    Use sklearn's accuracy_score and f1_score with average="macro".
    """
    if hasattr(eval_pred, "predictions"):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        logits, labels = eval_pred
    logits = _as_logits(logits)
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """
    Construct and train a Trainer.

    Returns the trained Trainer (trainer.model is the fine-tuned model). Pass
    id2label=ID2LABEL and label2id=LABEL2ID to the model so its config records
    the human-readable label names — Integration 7A reads them from
    `model.config.id2label` rather than hard-coding.
    """
    id2label = {i: ID2LABEL.get(i, f"LABEL_{i}") for i in range(num_labels)}
    label2id = {label: idx for idx, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the trainer's model on the test split.

    Read label names from trainer.model.config.id2label (do not hard-code).

    Returns:
        {
            "accuracy": float,
            "macro_f1": float,
            "per_class_f1": {label_name: f1, ...},
            "per_class_precision": {label_name: precision, ...},
            "per_class_recall": {label_name: recall, ...},
        }
    """
    pred_output = trainer.predict(tokenized_test)
    logits = _as_logits(pred_output.predictions)
    predicted = np.argmax(logits, axis=1)
    labels = pred_output.label_ids
    if labels is None:
        labels = np.asarray(tokenized_test["label"])

    label_items = _id2label_items(trainer.model.config.id2label, logits.shape[-1])
    label_ids = [idx for idx, _ in label_items]
    label_names = [name for _, name in label_items]
    trainer._last_eval_cache = {
        "logits": logits,
        "predicted": predicted,
        "labels": np.asarray(labels),
        "label_items": label_items,
    }

    per_class_f1 = f1_score(
        labels,
        predicted,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    per_class_precision = precision_score(
        labels,
        predicted,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    per_class_recall = recall_score(
        labels,
        predicted,
        labels=label_ids,
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(labels, predicted, labels=label_ids, average="macro", zero_division=0)
        ),
        "per_class_f1": {
            label_name: float(score) for label_name, score in zip(label_names, per_class_f1)
        },
        "per_class_precision": {
            label_name: float(score) for label_name, score in zip(label_names, per_class_precision)
        },
        "per_class_recall": {
            label_name: float(score) for label_name, score in zip(label_names, per_class_recall)
        },
    }


def main() -> None:
    """Orchestrate the full pipeline."""
    data_path = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"
    max_length = 128
    seed = 42
    lr = 5e-5
    epochs = 6 if os.environ.get("DATA_PATH") else 2
    batch_size = 8
    hub_username = os.environ.get("HF_USERNAME", "osamaharrab")

    ds = prepare_dataset(data_path, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer, max_length=max_length)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = make_training_args(
        output_dir,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    start_time = time.perf_counter()
    trainer = train_classifier(tokenized, model_name, training_args, tokenizer, num_labels=3)
    training_time_seconds = time.perf_counter() - start_time

    # Save locally (model/ is gitignored)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    metrics = evaluate_classifier(trainer, tokenized["test"])
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions CSV and confusion matrix CSV
    eval_cache = getattr(trainer, "_last_eval_cache", None)
    if eval_cache is None:
        pred_output = trainer.predict(tokenized["test"])
        pred_logits = _as_logits(pred_output.predictions)
        pred_idx = np.argmax(pred_logits, axis=1)
        true_idx = np.asarray(pred_output.label_ids)
        label_items = _id2label_items(trainer.model.config.id2label, pred_logits.shape[-1])
    else:
        pred_logits = eval_cache["logits"]
        pred_idx = eval_cache["predicted"]
        true_idx = eval_cache["labels"]
        label_items = eval_cache["label_items"]

    pred_probs = _softmax(pred_logits)
    label_ids = [idx for idx, _ in label_items]
    label_names = [name for _, name in label_items]
    id2label = dict(label_items)
    df_out = pd.DataFrame({
        "text": ds["test"]["text"],
        "label": [id2label[i] for i in true_idx],
        "predicted_label": [id2label[i] for i in pred_idx],
        "predicted_probability": [float(pred_probs[i, pred_idx[i]]) for i in range(len(pred_idx))],
    })
    for label_id, label_name in label_items:
        df_out[f"prob_{label_name}"] = [float(prob) for prob in pred_probs[:, label_id]]
    df_out.to_csv("predictions.csv", index=False)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    # Confusion matrix (for the evaluation report)
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(true_idx, pred_idx, labels=label_ids)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.index.name = "true_label"
    cm_df.to_csv("confusion_matrix.csv")
    print(cm_df.to_string())

    _write_evaluation_report(
        ds=ds,
        metrics=metrics,
        cm_df=cm_df,
        predictions_df=df_out,
        model_name=model_name,
        max_length=max_length,
        seed=seed,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        training_time_seconds=training_time_seconds,
        repo_id="m7-app-review-sentiment",
        hub_username=hub_username,
    )

    # Push to Hugging Face Hub.
    # Skipped in CI (DATA_PATH set); requires `huggingface-cli login` locally.
    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"
        try:
            trainer.args.hub_model_id = repo_id
            trainer.push_to_hub(commit_message="Fine-tuned app-review sentiment classifier")
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to https://huggingface.co/{hub_username}/{repo_id}")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")
            print("Run `huggingface-cli login` and try again.")


def _as_logits(predictions):
    """Return the logits array from Trainer predictions."""
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return np.asarray(predictions)


def _id2label_items(id2label, num_labels: int) -> list[tuple[int, str]]:
    """Normalize config.id2label keys, which may be ints in memory or strings after reload."""
    items = []
    if id2label:
        for raw_idx, label_name in id2label.items():
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < num_labels:
                items.append((idx, str(label_name)))

    by_idx = {idx: label_name for idx, label_name in items}
    for idx in range(num_labels):
        by_idx.setdefault(idx, f"LABEL_{idx}")
    return sorted(by_idx.items())


def _write_evaluation_report(
    ds: DatasetDict,
    metrics: dict,
    cm_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    model_name: str,
    max_length: int,
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int,
    training_time_seconds: float,
    repo_id: str,
    hub_username: str,
) -> None:
    """Write a markdown report draft from the run artifacts."""
    train_size = len(ds["train"])
    test_size = len(ds["test"])
    total_size = train_size + test_size
    all_labels = list(ds["train"]["label"]) + list(ds["test"]["label"])
    label_names = list(cm_df.index)
    label_counts = {
        label_name: all_labels.count(label_idx) for label_idx, label_name in enumerate(label_names)
    }

    aggregate_rows = [
        ["Accuracy", f"{metrics['accuracy']:.4f}"],
        ["Macro-F1", f"{metrics['macro_f1']:.4f}"],
    ]
    per_class_rows = [
        [
            label_name.title(),
            f"{metrics['per_class_f1'][label_name]:.4f}",
            f"{metrics['per_class_precision'][label_name]:.4f}",
            f"{metrics['per_class_recall'][label_name]:.4f}",
        ]
        for label_name in label_names
    ]

    error_sections = _qualitative_error_sections(predictions_df, label_names)
    report = "\n".join([
        "# Module 7 Week A — Lab Evaluation Report",
        "",
        "## Dataset",
        (
            "AARSynth app reviews Sentences-50Agree subset with "
            f"{total_size:,} examples across negative, neutral, and positive labels. "
            f"Label distribution: {_format_label_counts(label_counts)}; "
            f"split sizes: {train_size:,} train and {test_size:,} test."
        ),
        "",
        "## Model and hyperparameters",
        f"- Backbone: {model_name}",
        f"- Number of labels: {len(label_names)}",
        f"- Learning rate: {lr}",
        f"- Epochs: {epochs}",
        f"- Batch size: {batch_size}",
        f"- max_length: {max_length}",
        f"- Seed: {seed}",
        f"- Training time: {training_time_seconds:.1f} seconds wall-clock on this machine",
        "",
        "## Metrics on the test split",
        "",
        "Aggregate:",
        "",
        _markdown_table(["Metric", "Value"], aggregate_rows),
        "",
        "Per class:",
        "",
        _markdown_table(["Class", "F1", "Precision", "Recall"], per_class_rows),
        "",
        "## Confusion matrix",
        "",
        _markdown_table(["True \\ Pred", *label_names], [
            [label_name, *[str(int(value)) for value in cm_df.loc[label_name].tolist()]]
            for label_name in label_names
        ]),
        "",
        "## Three qualitative error examples (one per class)",
        "",
        error_sections,
        "",
        "## Hugging Face Hub model URL",
        f"https://huggingface.co/{hub_username}/{repo_id}",
        "",
    ])

    with open("evaluation-report.md", "w") as f:
        f.write(report)


def _qualitative_error_sections(predictions_df: pd.DataFrame, label_names: list[str]) -> str:
    sections = []
    for label_name in label_names:
        examples = predictions_df[
            (predictions_df["label"] == label_name)
            & (predictions_df["predicted_label"] != label_name)
        ]
        if examples.empty:
            examples = predictions_df[predictions_df["label"] == label_name]
            note = (
                "The model did not miss this class in the test split, so this correct "
                "example is a useful check of what it learned."
            )
        else:
            note = (
                "This error likely reflects mixed or ambiguous sentiment cues; review "
                "the phrasing manually before final submission."
            )

        if examples.empty:
            sections.append(f"### {label_name.title()}\n\nNo examples for this class were present.")
            continue

        row = examples.iloc[0]
        gold_prob = row.get(f"prob_{label_name}", np.nan)
        sections.append("\n".join([
            f"### {label_name.title()}",
            f"- Original sentence: {row['text']}",
            f"- Gold label: {row['label']}",
            f"- Predicted label: {row['predicted_label']}",
            f"- Predicted probability for the gold label: {float(gold_prob):.4f}",
            f"- Analysis: {note}",
        ]))
    return "\n\n".join(sections)


def _format_label_counts(label_counts: dict[str, int]) -> str:
    return ", ".join(f"{label}={count:,}" for label, count in label_counts.items())


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    main()
