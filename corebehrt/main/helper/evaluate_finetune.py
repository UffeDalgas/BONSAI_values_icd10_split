from os.path import join
from typing import List
import torch
import pandas as pd
import numpy as np

from corebehrt.modules.setup.manager import ModelManager
from corebehrt.modules.trainer.inference import EHRInferenceRunner
from corebehrt.modules.preparation.dataset import BinaryOutcomeDataset
from corebehrt.modules.setup.config import instantiate_function


def inference_fold(
    finetune_folder: str,
    cfg: dict,
    test_data: BinaryOutcomeDataset,
    logger,
    fold: int,
) -> None:
    fold_folder = join(finetune_folder, f"fold_{fold}")

    # Load model
    modelmanager_trained = ModelManager(cfg, fold)
    checkpoint = modelmanager_trained.load_checkpoint(checkpoints=True)
    model = modelmanager_trained.initialize_finetune_model(checkpoint, [])
    print(f"Model loaded from {fold_folder}")

    # Run inference
    return_embeddings = cfg.get("return_embeddings", False)
    evaluater = EHRInferenceRunner(
        model=model,
        test_dataset=test_data,  # test only after training
        args=cfg.trainer_args,
        cfg=cfg,
    )
    logits_tensor, targets_tensor, embeddings_tensor = evaluater.inference_loop(
        return_embeddings=return_embeddings
    )
    probas = torch.sigmoid(logits_tensor).numpy()

    return probas, embeddings_tensor


def get_sequence_length(dataset: BinaryOutcomeDataset) -> List[int]:
    lengths = [len(patient.concepts) for patient in dataset.patients]
    return lengths


def compute_metrics(cfg, targets, all_probas, logger):
    """
    Computes and saves metrics for each fold and the average metrics.
    """
    if not hasattr(cfg, "metrics") or not cfg.metrics:
        return

    targets = np.asarray(targets)
    all_probas = np.asarray(all_probas)

    # --- label distribution (same targets across folds) ---
    n_total = int(targets.size)
    n_pos = int((targets == 1).sum())
    n_neg = int((targets == 0).sum())
    prev = n_pos / n_total if n_total else float("nan")
    logger.info(f"Label distribution: total={n_total}  pos(1)={n_pos}  neg(0)={n_neg}  "
                f"prevalence={prev:.4f}")
    pd.DataFrame([{"n_total": n_total, "label_0": n_neg, "label_1": n_pos,
                   "prevalence": prev}]).to_csv(
        join(cfg.paths.predictions, "label_distribution.csv"), index=False)

    metrics = {k: instantiate_function(v) for k, v in cfg.metrics.items()}
    thr = float(getattr(cfg, "decision_threshold", 0.5))
    fold_metrics_list = []

    for n_fold, probas in enumerate(all_probas, start=1):
        fold_metrics = {name: func(targets, probas) for name, func in metrics.items()}
        fold_metrics.update(_threshold_metrics(targets, probas, thr))
        fold_metrics["n_total"] = n_total
        fold_metrics["label_0"] = n_neg
        fold_metrics["label_1"] = n_pos
        fold_metrics["prevalence"] = round(prev, 4)
        fold_metrics["fold"] = f"fold_{n_fold}"
        fold_metrics_list.append(fold_metrics)

    metrics_df = pd.DataFrame(fold_metrics_list)
    avg_metrics = metrics_df.drop(columns=["fold"]).mean().to_dict()
    avg_metrics["fold"] = "average"
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    metrics_df.to_csv(join(cfg.paths.predictions, "metrics.csv"), index=False)

    logger.info("Average metrics:")
    for key in avg_metrics:
        if key != "fold":
            logger.info(f"{key}: {avg_metrics[key]:.4f}")


def _threshold_metrics(targets, probas, thr: float = 0.5) -> dict:
    """Threshold-based + calibration metrics that matter for imbalanced mortality."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, matthews_corrcoef,
        balanced_accuracy_score, brier_score_loss, confusion_matrix,
    )
    preds = (np.asarray(probas) >= thr).astype(int)
    out = {}
    try:
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        out["tp"], out["fp"], out["tn"], out["fn"] = int(tp), int(fp), int(tn), int(fn)
        out["sensitivity_recall"] = recall_score(targets, preds, zero_division=0)
        out["specificity"] = tn / (tn + fp) if (tn + fp) else float("nan")
        out["precision_ppv"] = precision_score(targets, preds, zero_division=0)
        out["f1"] = f1_score(targets, preds, zero_division=0)
        out["balanced_accuracy"] = balanced_accuracy_score(targets, preds)
        out["mcc"] = matthews_corrcoef(targets, preds) if len(set(preds)) > 1 else 0.0
        out["brier"] = brier_score_loss(targets, probas)
    except Exception:
        pass
    return out
