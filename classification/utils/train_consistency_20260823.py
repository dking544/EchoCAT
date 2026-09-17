from __future__ import annotations

import os

import torch
from numpy import mean
from tqdm import tqdm

from core.evaluations import evaluate
from utils.checkpoint import save_checkpoint
from utils.train_utils import (
    AsciiTable,
    SKLEARN_AVAILABLE,
    calculate_detailed_metrics,
    get_lr,
)


_CONSISTENCY_CFG = None


def configure_consistency(config):
    global _CONSISTENCY_CFG
    config = dict(config or {})
    if config.get("type") != "jensen_shannon":
        raise ValueError(f"Unsupported consistency config: {config}")
    weight = float(config.get("weight", 0.5))
    if weight < 0:
        raise ValueError("Consistency weight must be non-negative")
    _CONSISTENCY_CFG = dict(config, weight=weight)


def _jensen_shannon(first, second):
    epsilon = torch.finfo(first.dtype).eps
    first = first.clamp_min(epsilon)
    second = second.clamp_min(epsilon)
    midpoint = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.log() - midpoint.log())).sum(dim=1).mean()
        + (second * (second.log() - midpoint.log())).sum(dim=1).mean()
    )


def train(model, runner, lr_update_func, device, epoch, epoches, test_cfg, meta):

    if _CONSISTENCY_CFG is None:
        raise RuntimeError("configure_consistency must be called before training")

    train_loss = 0.0
    classification_loss_total = 0.0
    consistency_loss_total = 0.0
    pred_list, target_list = [], []
    runner["epoch"] = epoch + 1
    meta["epoch"] = runner["epoch"]
    model.train()

    loader = runner["train_loader"]
    with tqdm(
        total=len(loader),
        desc=f"Train consistency: Epoch {epoch + 1}/{epoches}",
        postfix=dict,
        mininterval=0.3,
    ) as progress:
        for iteration, batch in enumerate(loader):
            paired_images, targets, _ = batch
            if paired_images.ndim != 5 or paired_images.shape[1] != 2:
                raise ValueError(
                    "Consistency input must have shape [B,2,C,H,W], got "
                    f"{tuple(paired_images.shape)}"
                )
            paired_images = paired_images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = paired_images.shape[0]
            images = torch.cat(
                (paired_images[:, 0], paired_images[:, 1]), dim=0,
            )
            repeated_targets = torch.cat((targets, targets), dim=0)

            runner["optimizer"].zero_grad()
            lr_update_func.before_train_iter(runner)
            probabilities, losses = model(
                images,
                targets=repeated_targets,
                return_loss=True,
                train_statu=True,
            )
            clean_probabilities = probabilities[:batch_size]
            styled_probabilities = probabilities[batch_size:]
            consistency_loss = _jensen_shannon(
                clean_probabilities, styled_probabilities,
            )
            classification_loss = losses["loss"]
            total_loss = (
                classification_loss
                + _CONSISTENCY_CFG["weight"] * consistency_loss
            )
            total_loss.backward()
            runner["optimizer"].step()

            averaged_probabilities = 0.5 * (
                clean_probabilities + styled_probabilities
            )
            pred_list.append(averaged_probabilities.detach())
            target_list.append(targets.detach())
            train_loss += float(total_loss.detach().item())
            classification_loss_total += float(
                classification_loss.detach().item()
            )
            consistency_loss_total += float(consistency_loss.detach().item())

            progress.set_postfix(
                Loss=train_loss / (iteration + 1),
                Cls=classification_loss_total / (iteration + 1),
                JSD=consistency_loss_total / (iteration + 1),
                Lr=get_lr(runner["optimizer"]),
            )
            runner["iter"] += 1
            meta["iter"] = runner["iter"]
            progress.update(1)

    if not pred_list:
        raise RuntimeError("Training loader produced no batches")
    average_loss = train_loss / len(loader)
    predictions = torch.cat(pred_list)
    targets = torch.cat(target_list)
    eval_results = evaluate(
        predictions,
        targets,
        test_cfg.get("metrics"),
        test_cfg.get("metric_options"),
    )
    meta["train_info"]["train_loss"].append(average_loss)
    meta["train_info"]["train_acc"].append(eval_results)
    if SKLEARN_AVAILABLE:
        detailed_metrics = calculate_detailed_metrics(predictions, targets)
        detailed_metrics.update(
            avg_loss=average_loss,
            avg_classification_loss=(
                classification_loss_total / len(loader)
            ),
            avg_consistency_loss=consistency_loss_total / len(loader),
            consistency_weight=_CONSISTENCY_CFG["weight"],
        )
        meta.setdefault("train_detailed_metrics", []).append(detailed_metrics)

    table = AsciiTable(
        (
            (
                "Top-1 Acc",
                "Top-5 Acc",
                "Mean Precision",
                "Mean Recall",
                "Mean F1 Score",
                "JSD",
            ),
            (
                f"{eval_results.get('accuracy_top-1', 0.0):.2f}",
                f"{eval_results.get('accuracy_top-5', 100.0):.2f}",
                f"{mean(eval_results.get('precision', 0.0)):.2f}",
                f"{mean(eval_results.get('recall', 0.0)):.2f}",
                f"{mean(eval_results.get('f1_score', 0.0)):.2f}",
                f"{consistency_loss_total / len(loader):.6f}",
            ),
        ),
        "Train Results",
    )
    print(f"\n{table.table}\n")
