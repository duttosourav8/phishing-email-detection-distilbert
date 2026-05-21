
"""
Phishing Email Payload Detection using DistilBERT + Machine Learning
====================================================================

GitHub-ready, structurally cleaned version of the original Colab notebooks/scripts.

Project goal
------------
Detect phishing emails from email text using:
1. Fine-tuned DistilBERT sequence classification.
2. DistilBERT CLS embeddings + classical ML classifiers.


- Uses a public Hugging Face dataset:
    zefang-liu/phishing-email-dataset

"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Project constants
# ============================================================

RANDOM_STATE = 42
DATASET_NAME = "zefang-liu/phishing-email-dataset"
BASE_MODEL_NAME = "distilbert-base-uncased"

LABEL_MAP = {
    "Safe Email": 0,
    "Phishing Email": 1,
}

ID_TO_LABEL = {
    0: "Safe Email",
    1: "Phishing Email",
}

PHISHING_CUE_WORDS = [
    "verify now",
    "click here",
    "urgent",
    "confirm account",
    "limited offer",
    "account suspended",
    "update credentials",
]





# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    dataset_name: str = DATASET_NAME
    base_model_name: str = BASE_MODEL_NAME
    output_dir: str = "results"
    model_dir: str = "models"
    test_size: float = 0.20
    random_state: int = RANDOM_STATE
    max_length: int = 512
    batch_size: int = 16
    embedding_batch_size: int = 32
    epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    use_train_augmentation: bool = True
    synonym_swaps: int = 2
    save_plots: bool = True


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_output_dirs(config: Config) -> None:
    """Create output directories."""
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)


# ============================================================
# Dataset loading and cleaning
# ============================================================

def load_phishing_dataset(dataset_name: str = DATASET_NAME) -> pd.DataFrame:
    """
    Load phishing email dataset from Hugging Face.

    Expected columns in the source dataset:
    - Email Text
    - Email Type

    Returns
    -------
    pd.DataFrame with columns:
    - text
    - label
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    df = dataset["train"].to_pandas()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required_columns = {"Email Text", "Email Type"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")

    df = df.dropna(subset=["Email Text", "Email Type"]).copy()
    df["Email Text"] = df["Email Text"].astype(str).str.lower().str.strip()
    df = df[df["Email Text"] != ""].copy()

    df["label"] = df["Email Type"].map(LABEL_MAP)

    if df["label"].isna().any():
        unknown_labels = df.loc[df["label"].isna(), "Email Type"].unique()
        raise ValueError(f"Unknown labels found: {unknown_labels}")

    df = df.rename(columns={"Email Text": "text"})
    df = df[["text", "label"]].reset_index(drop=True)

    return df


def split_original_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the original dataset before augmentation to prevent leakage.

    Why this matters:
    If we augment first and split later, the original version of an email can
    end up in train while its augmented version goes to test. That makes the
    test set too easy and can inflate performance.
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================
# Training-only adversarial-style augmentation
# ============================================================

def _safe_wordnet_download() -> None:
    """Download NLTK WordNet resources if available."""
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception:
        pass


def synonym_swap(text: str, n: int = 2, seed: int = RANDOM_STATE) -> str:
    """
    Replace up to `n` words with WordNet synonyms.

    This is a simple adversarial-style augmentation method. It is not meant
    to be linguistically perfect; it is a lightweight robustness experiment.
    """
    _safe_wordnet_download()

    try:
        from nltk.corpus import wordnet
    except Exception:
        return text

    rng = random.Random(seed)
    words = str(text).split()
    indices = list(range(len(words)))
    rng.shuffle(indices)

    swaps = 0
    for idx in indices:
        original_word = words[idx]
        synsets = wordnet.synsets(original_word)
        if not synsets:
            continue

        lemmas = [
            lemma.name().replace("_", " ")
            for lemma in synsets[0].lemmas()
            if lemma.name().lower() != original_word.lower()
        ]

        if lemmas:
            words[idx] = rng.choice(lemmas)
            swaps += 1

        if swaps >= n:
            break

    return " ".join(words)


def add_phishing_cue_words(text: str, seed: int = RANDOM_STATE) -> str:
    """Append one phishing-like cue phrase to the text."""
    rng = random.Random(seed)
    return f"{text} {rng.choice(PHISHING_CUE_WORDS)}"


def augment_training_data(
    train_df: pd.DataFrame,
    synonym_swaps: int = 2,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Create augmented training examples only from the training split.

    This prevents train-test leakage.
    """
    augmented_rows = []

    for i, row in train_df.iterrows():
        base_text = str(row["text"])
        swapped_text = synonym_swap(
            base_text,
            n=synonym_swaps,
            seed=random_state + i,
        )
        augmented_text = add_phishing_cue_words(
            swapped_text,
            seed=random_state + i,
        )

        augmented_rows.append({
            "text": augmented_text,
            "label": int(row["label"]),
        })

    augmented_df = pd.DataFrame(augmented_rows)

    combined_train_df = pd.concat(
        [train_df[["text", "label"]], augmented_df],
        ignore_index=True,
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return combined_train_df


def prepare_train_test_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, split, and optionally augment training data."""
    df = load_phishing_dataset(config.dataset_name)
    train_df, test_df = split_original_dataset(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    if config.use_train_augmentation:
        train_df = augment_training_data(
            train_df,
            synonym_swaps=config.synonym_swaps,
            random_state=config.random_state,
        )

    return train_df, test_df


# ============================================================
# Evaluation utilities
# ============================================================

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute binary classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = float("nan")

    return metrics


def print_and_save_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    """Print metrics and save them as JSON."""
    print(json.dumps(metrics, indent=2))
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save sklearn classification report."""
    from sklearn.metrics import classification_report

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Safe Email", "Phishing Email"],
        zero_division=0,
    )
    print(report)
    output_path.write_text(report, encoding="utf-8")


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save confusion matrix plot."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Safe", "Phishing"],
    )

    display.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_roc_curve_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
    title: str = "ROC-AUC Curve",
) -> None:
    """Save ROC curve plot."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc_value = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# Approach 1: Fine-tuned DistilBERT
# ============================================================

class EmailTextDataset:
    """PyTorch Dataset wrapper for tokenized email text."""

    def __init__(self, encodings: Dict, labels: Iterable[int]):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx: int) -> Dict:
        import torch

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def run_finetuned_distilbert(config: Config) -> Dict[str, float]:
    """
    Fine-tune DistilBERT for phishing email classification.

    This is the strongest and most direct transformer-based approach.
    """
    import torch
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        Trainer,
        TrainingArguments,
    )

    set_seed(config.random_state)
    ensure_output_dirs(config)

    train_df, test_df = prepare_train_test_data(config)

    tokenizer = DistilBertTokenizerFast.from_pretrained(config.base_model_name)

    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=config.max_length,
    )

    test_encodings = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=config.max_length,
    )

    train_dataset = EmailTextDataset(train_encodings, train_df["label"].tolist())
    test_dataset = EmailTextDataset(test_encodings, test_df["label"].tolist())

    model = DistilBertForSequenceClassification.from_pretrained(
        config.base_model_name,
        num_labels=2,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        # Softmax probability of phishing class
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        phishing_probs = probs[:, 1]
        return compute_binary_metrics(labels, preds, phishing_probs)

    training_args = TrainingArguments(
        output_dir=str(Path(config.model_dir) / "distilbert_checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(Path(config.output_dir) / "logs"),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        seed=config.random_state,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    # Predictions for plots and error analysis
    outputs = trainer.predict(test_dataset)
    logits = outputs.predictions
    y_true = outputs.label_ids
    y_pred = logits.argmax(axis=1)

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y_proba = probs[:, 1]

    output_dir = Path(config.output_dir)
    print_and_save_metrics(eval_metrics, output_dir / "finetuned_distilbert_metrics.json")
    save_classification_report(y_true, y_pred, output_dir / "finetuned_distilbert_classification_report.txt")

    if config.save_plots:
        save_confusion_matrix_plot(
            y_true,
            y_pred,
            output_dir / "finetuned_distilbert_confusion_matrix.png",
            title="Fine-tuned DistilBERT Confusion Matrix",
        )
        save_roc_curve_plot(
            y_true,
            y_proba,
            output_dir / "finetuned_distilbert_roc_curve.png",
            title="Fine-tuned DistilBERT ROC-AUC Curve",
        )

    trainer.save_model(str(Path(config.model_dir) / "finetuned_distilbert"))

    return eval_metrics


# ============================================================
# Approach 2: DistilBERT embeddings + classical ML
# ============================================================

def get_distilbert_cls_embeddings(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    """
    Extract CLS-token embeddings from DistilBERT.

    These embeddings can be used as dense features for classical ML models.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for start_idx in range(0, len(texts), batch_size):
            batch = [str(x).strip() for x in texts[start_idx:start_idx + batch_size]]
            batch = [x if x else "[EMPTY]" for x in batch]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)
            cls_vectors = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(cls_vectors)

    return np.vstack(embeddings)


def run_embedding_ml_benchmark(config: Config) -> pd.DataFrame:
    """
    Extract DistilBERT CLS embeddings and benchmark classical ML classifiers.
    """
    import joblib
    from sklearn.ensemble import (
        AdaBoostClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from transformers import DistilBertModel, DistilBertTokenizerFast

    set_seed(config.random_state)
    ensure_output_dirs(config)

    train_df, test_df = prepare_train_test_data(config)

    tokenizer = DistilBertTokenizerFast.from_pretrained(config.base_model_name)
    bert_model = DistilBertModel.from_pretrained(config.base_model_name)

    start_time = time.time()

    X_train = get_distilbert_cls_embeddings(
        train_df["text"].tolist(),
        tokenizer,
        bert_model,
        batch_size=config.embedding_batch_size,
        max_length=config.max_length,
    )
    X_test = get_distilbert_cls_embeddings(
        test_df["text"].tolist(),
        tokenizer,
        bert_model,
        batch_size=config.embedding_batch_size,
        max_length=config.max_length,
    )

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"Embedding extraction completed in {elapsed_minutes:.2f} minutes.")

    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=config.random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=config.random_state),
        "AdaBoost": AdaBoostClassifier(random_state=config.random_state),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=config.random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=config.random_state),
    }

    output_dir = Path(config.output_dir)
    model_dir = Path(config.model_dir)
    results = []

    best_name = None
    best_f1 = -1.0
    best_model = None
    best_y_pred = None
    best_y_proba = None

    for model_name, classifier in models.items():
        print(f"\n=== Training {model_name} ===")
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        if hasattr(classifier, "predict_proba"):
            y_proba = classifier.predict_proba(X_test)[:, 1]
        elif hasattr(classifier, "decision_function"):
            scores = classifier.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        else:
            y_proba = y_pred

        metrics = compute_binary_metrics(y_test, y_pred, y_proba)
        metrics["model"] = model_name
        results.append(metrics)

        print(classification_report(y_test, y_pred, target_names=["Safe Email", "Phishing Email"], zero_division=0))
        print("ROC AUC:", metrics.get("roc_auc"))

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = model_name
            best_model = classifier
            best_y_pred = y_pred
            best_y_proba = y_proba

    results_df = pd.DataFrame(results).sort_values(
        by=["f1", "recall", "precision"],
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(output_dir / "distilbert_embeddings_ml_model_comparison.csv", index=False)

    if best_model is not None:
        joblib.dump(best_model, model_dir / "best_embedding_ml_classifier.joblib")
        save_classification_report(
            y_test,
            best_y_pred,
            output_dir / "best_embedding_ml_classification_report.txt",
        )

        if config.save_plots:
            save_confusion_matrix_plot(
                y_test,
                best_y_pred,
                output_dir / "best_embedding_ml_confusion_matrix.png",
                title=f"Best Embedding ML Model: {best_name}",
            )
            save_roc_curve_plot(
                y_test,
                best_y_proba,
                output_dir / "best_embedding_ml_roc_curve.png",
                title=f"Best Embedding ML ROC Curve: {best_name}",
            )

    print("\nModel comparison:")
    print(results_df)

    return results_df


# ============================================================
# Reported results mode
# ============================================================

def print_reported_original_results() -> None:
    """
    Print documented original notebook results without rerunning training.

    Use this when you only need to upload defensible code + documented result
    summary to GitHub and do not have time to rerun the full experiment.
    """
    print("\nReported original notebook results")
    print("=" * 60)
    print(json.dumps(REPORTED_ORIGINAL_NOTEBOOK_RESULTS, indent=2))

    print("\nRecommended README result line:")
    print(
        "Fine-tuned DistilBERT achieved "
        f"{REPORTED_ORIGINAL_NOTEBOOK_RESULTS['accuracy'] * 100:.2f}% accuracy, "
        f"{REPORTED_ORIGINAL_NOTEBOOK_RESULTS['f1'] * 100:.2f}% F1-score, and "
        f"{REPORTED_ORIGINAL_NOTEBOOK_RESULTS['roc_auc']:.4f} ROC-AUC "
        "in the original notebook evaluation."
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phishing email detection using DistilBERT and ML classifiers."
    )

    parser.add_argument(
        "--mode",
        choices=["reported", "embeddings", "finetune", "all"],
        default="reported",
        help=(
            "reported: print documented original results only; "
            "embeddings: run DistilBERT embeddings + classical ML; "
            "finetune: fine-tune DistilBERT; "
            "all: run embeddings and fine-tuning."
        ),
    )
    parser.add_argument("--output-dir", default="results", help="Directory for result files.")
    parser.add_argument("--model-dir", default="models", help="Directory for saved models.")
    parser.add_argument("--epochs", type=int, default=2, help="Fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training/eval batch size.")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Embedding extraction batch size.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum DistilBERT token length.")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training-only augmentation.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot saving.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config(
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        max_length=args.max_length,
        use_train_augmentation=not args.no_augmentation,
        save_plots=not args.no_plots,
    )

    set_seed(config.random_state)
    ensure_output_dirs(config)

    # Save run config for reproducibility.
    Path(config.output_dir, "run_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    if args.mode == "reported":
        print_reported_original_results()

    elif args.mode == "embeddings":
        run_embedding_ml_benchmark(config)

    elif args.mode == "finetune":
        run_finetuned_distilbert(config)

    elif args.mode == "all":
        run_embedding_ml_benchmark(config)
        run_finetuned_distilbert(config)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
