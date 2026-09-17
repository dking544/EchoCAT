import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ListDataset(Dataset):
    def __init__(self, list_path, transform):
        self.samples = []
        for raw in Path(list_path).read_text(encoding="utf-8").splitlines():
            if raw.strip():
                path, label = raw.rsplit(" ", 1)
                image_path = Path(path).expanduser()
                if not image_path.is_absolute():
                    image_path = Path(list_path).resolve().parent / image_path
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                self.samples.append((str(image_path.resolve()), int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("L")
            image = self.transform(image)
        return image, label


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = ListDataset(args.train_list, train_tf)
    val_set = ListDataset(args.val_list, val_tf)
    generator = torch.Generator().manual_seed(args.seed)
    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
        generator=generator, **common)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 2, shuffle=False,
        drop_last=False, **common)
    return train_set, val_set, train_loader, val_loader


def class_weights(dataset, device):
    count = np.bincount([label for _, label in dataset.samples], minlength=2)
    weight = count.sum() / (2.0 * np.maximum(count, 1))
    return torch.tensor(weight, dtype=torch.float32, device=device), count.tolist()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels, predictions, scores = [], [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, target).item() * target.size(0)
        probability = logits.softmax(dim=1)
        labels.extend(target.cpu().tolist())
        predictions.extend(probability.argmax(dim=1).cpu().tolist())
        scores.extend(probability[:, 1].cpu().tolist())
    accuracy = float(np.mean(np.asarray(labels) == np.asarray(predictions)))
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy,
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--test-list", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best_resnet18_view_gray.pth").exists():
        raise FileExistsError("Teacher output already contains a checkpoint; use a new directory")
    train_set, val_set, train_loader, val_loader = build_loaders(args)
    if {x[0] for x in train_set.samples} & {x[0] for x in val_set.samples}:
        raise ValueError("Teacher train/validation image overlap")


    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    weights, distribution = class_weights(train_set, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler(enabled=device.type == "cuda")

    history = []
    best_accuracy = -1.0
    print(f"device={device} train={len(train_set)} class_count={distribution}", flush=True)
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for images, target in train_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * target.size(0)
            correct += (logits.argmax(dim=1) == target).sum().item()
            seen += target.size(0)
        scheduler.step()

        val = evaluate(model, val_loader, criterion, device)
        record = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": running_loss / seen,
            "train_accuracy": correct / seen,
            **{f"val_{key}": value for key, value in val.items()},
            "seconds": time.time() - start,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        if val["accuracy"] > best_accuracy:
            best_accuracy = val["accuracy"]
            checkpoint = {
                "arch": "resnet18",
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "metrics": val,
                "class_mapping": {0: "4CH", 1: "non-4CH"},
                "cam_target_class": 0,
                "normalization": {"mean": MEAN, "std": STD},
                "training": "from scratch on grayscale view split",
            }
            torch.save(checkpoint, output / "best_resnet18_view_gray.pth")

    if args.test_list:
        test_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        test_set = ListDataset(args.test_list, test_tf)
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size * 2, shuffle=False,
            drop_last=False, num_workers=args.workers, pin_memory=True,
            persistent_workers=args.workers > 0)
        checkpoint_path = output / "best_resnet18_view_gray.pth"
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        test_metrics = evaluate(model, test_loader, criterion, device)
        (output / "held_out_test_metrics.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(
            "held_out_test=" + json.dumps(test_metrics, ensure_ascii=False),
            flush=True)

    print(f"best_val_accuracy={best_accuracy:.6f}", flush=True)


if __name__ == "__main__":
    main()
