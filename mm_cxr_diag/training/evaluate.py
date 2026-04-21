"""Multi-label evaluation: AUC, optimal thresholds, classification reports."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tqdm
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    coverage_error,
    f1_score,
    fbeta_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    label_ranking_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from mm_cxr_diag.inference.persistence import load_model
from mm_cxr_diag.models import CXRModel

Device = torch.device | Literal["cuda", "cpu"]

logger = logging.getLogger(__name__)


def run_inference(
    model: str | CXRModel,
    test_loader: DataLoader,
    device: Device = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Run a model over a DataLoader and collect sigmoid probabilities + labels."""
    if isinstance(model, str):
        path = Path(model)
        if not path.exists():
            raise FileNotFoundError(f"Model path {model} does not exist.")
        if not path.is_file():
            raise FileNotFoundError(f"Model path {model} is not a file.")
        if path.suffix != ".pth":
            logger.warning("Model path %s does not have a .pth extension.", model)
        logger.info("Loading model from %s", model)
        model = load_model(path).to(device)
    else:
        model = model.to(device)

    all_preds: list = []
    all_labels: list = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(
            test_loader,
            desc="Running inference",
            unit="batch",
            total=len(test_loader),
        )
        for images, tabular, labels in pbar:
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)

            outputs = model(images, tabular)
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    labels: Sequence[str],
    rare_threshold: float = 0.01,
) -> dict[str, float]:
    """Per-class optimal thresholds — F1 for common classes, F2 for rare ones."""
    thresholds: dict[str, float] = {}
    candidates = np.linspace(0.01, 0.99, 99)

    for i, label in enumerate(labels):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred_proba[:, i]
        prevalence = float(np.mean(y_true_i))

        best_threshold = 0.5
        best_metric = 0.0
        if prevalence < rare_threshold:
            for t in candidates:
                score = fbeta_score(y_true_i, (y_pred_i >= t).astype(int), beta=2)
                if score > best_metric:
                    best_metric = score
                    best_threshold = float(t)
        else:
            for t in candidates:
                score = f1_score(y_true_i, (y_pred_i >= t).astype(int))
                if score > best_metric:
                    best_metric = score
                    best_threshold = float(t)

        thresholds[label] = best_threshold

    return thresholds


def evaluate_model(
    preds: np.ndarray,
    labels: np.ndarray,
    class_labels: Sequence[str],
) -> dict:
    """Compute the full multi-label metric suite."""
    thresholds = find_optimal_thresholds(labels, preds, class_labels)

    binary_preds = np.zeros_like(preds)
    for i, label in enumerate(class_labels):
        binary_preds[:, i] = (preds[:, i] >= thresholds[label]).astype(int)

    results: dict = {"thresholds": thresholds}
    results["report"] = classification_report(
        labels, binary_preds, target_names=list(class_labels), output_dict=True
    )

    auc_scores: list[float] = []
    valid_indices: list[int] = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            auc_scores.append(float(auc))
            valid_indices.append(i)
        else:
            auc_scores.append(float("nan"))
    results["auc_scores"] = auc_scores

    if valid_indices:
        results["micro_auc"] = float(
            roc_auc_score(
                labels[:, valid_indices], preds[:, valid_indices], average="micro"
            )
        )
        results["macro_auc"] = float(
            roc_auc_score(
                labels[:, valid_indices], preds[:, valid_indices], average="macro"
            )
        )
        results["weighted_auc"] = float(
            roc_auc_score(
                labels[:, valid_indices], preds[:, valid_indices], average="weighted"
            )
        )
    else:
        results["micro_auc"] = float("nan")
        results["macro_auc"] = float("nan")
        results["weighted_auc"] = float("nan")

    results["hamming_loss"] = float(hamming_loss(labels, binary_preds))
    results["jaccard_similarity"] = float(
        jaccard_score(labels, binary_preds, average="samples")
    )
    results["avg_precision"] = float(
        average_precision_score(labels, preds, average="weighted")
    )

    results["confusion_matrices"] = [
        confusion_matrix(labels[:, i], binary_preds[:, i])
        for i in range(labels.shape[1])
    ]
    results["lrap"] = float(label_ranking_average_precision_score(labels, preds))
    results["coverage_error"] = float(coverage_error(labels, preds))
    results["ranking_loss"] = float(label_ranking_loss(labels, preds))

    return results


def print_evaluation_results(
    results: dict,
    class_labels: Sequence[str],
    save_path: str | Path | None = None,
) -> None:
    """Print (and optionally persist) the evaluation summary."""
    lines: list[str] = []
    lines.append("Optimal Classification Thresholds:\n")
    for class_name, threshold in results["thresholds"].items():
        lines.append(f"{class_name}: {threshold:.4f}")

    lines.append("\nIndividual Class Performance:")
    lines.append(
        "(AUC scores are binary classification metrics that handle class imbalance)\n"
    )
    for class_name, auc in zip(class_labels, results["auc_scores"], strict=True):
        support = (
            results["report"][class_name]["support"]
            if class_name in results["report"]
            else 0
        )
        lines.append(f"{class_name}:")
        if not np.isnan(auc):
            lines.append(f"  AUC Score: {auc:.4f}")
        else:
            lines.append("  AUC Score: Not applicable (only one class present)")
        lines.append(f"  Support: {support} samples")
        lines.append("")

    lines.append("Overall AUC Scores:")
    lines.append(f"Weighted Average AUC: {results['weighted_auc']:.4f}")
    lines.append(f"Macro Average AUC:    {results['macro_auc']:.4f}")
    lines.append(f"Micro Average AUC:    {results['micro_auc']:.4f}")

    lines.append("\nOverall Metrics:")
    lines.append(f"Hamming Loss:                    {results['hamming_loss']:.4f}")
    lines.append(
        f"Jaccard Similarity:              {results['jaccard_similarity']:.4f}"
    )
    lines.append(f"Average Precision:               {results['avg_precision']:.4f}")
    lines.append(f"Label Ranking Average Precision: {results['lrap']:.4f}")
    lines.append(f"Coverage Error:                  {results['coverage_error']:.4f}")
    lines.append(f"Ranking Loss:                    {results['ranking_loss']:.4f}")

    lines.append("\nClassification Report:\n")
    for class_name, metrics in results["report"].items():
        if isinstance(metrics, dict):
            lines.append(f"Class: {class_name}")
            lines.append(f"  Precision: {metrics['precision']:.4f}")
            lines.append(f"  Recall:    {metrics['recall']:.4f}")
            lines.append(f"  F1-score:  {metrics['f1-score']:.4f}")
            lines.append(f"  Support:   {metrics['support']}")
            lines.append("")

    lines.append("\nConfusion Matrices (per class):")
    for class_name, cm in zip(class_labels, results["confusion_matrices"], strict=True):
        lines.append(f"\n{class_name}:")
        lines.append("[[TN FP]")
        lines.append(" [FN TP]]")
        lines.append(str(cm))

    text = "\n".join(lines)
    print(text)

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Results saved to {save_path}")
