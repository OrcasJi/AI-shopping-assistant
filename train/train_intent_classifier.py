# train/train_intent_classifier.py
"""
Train BERT IntentClassifier with visualization and stability tweaks:
- Focal Loss (optional)
- Class weights
- Tune last N encoder layers
- Cosine w/ warmup (monotonic decay) or linear scheduler
- AMP (torch.amp) + grad clipping + gradient accumulation
- Early stopping by macro-F1
- TensorBoard (optional)
"""
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from utils.paths import DATA_DIR, MODELS_DIR, ensure_dirs
from utils.data_utils import load_intent_data, split_data
from models.intent_classifier import IntentClassifier, IntentDataset

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import ceil

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False


# ===== Helper functions =====
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(data_csv, model_name, max_length, batch_size, val_ratio, num_workers, pin_memory):
    df, label_map, reverse_map = load_intent_data(data_csv)
    train_df, valid_df = split_data(df, test_size=val_ratio, stratify=True)

    train_ds = IntentDataset(train_df["text"].tolist(), train_df["label_id"].tolist(),
                             model_name=model_name, max_length=max_length)
    valid_ds = IntentDataset(valid_df["text"].tolist(), valid_df["label_id"].tolist(),
                             model_name=model_name, max_length=max_length)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
    return train_dl, valid_dl, label_map, reverse_map, train_df["label_id"].tolist(), valid_df["label_id"].tolist()


def maybe_build_class_weights(labels, num_classes, mode):
    if mode != "auto":
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32)


def freeze_encoder_layers(model, tune_last_n=0, freeze_encoder=False):
    """
    freeze_encoder=True: 冻结全部 BERT 编码层（仅训练分类头）
    tune_last_n>0: 只解冻最后 N 层 + pooler + classifier
    """
    if freeze_encoder:
        for p in model.bert.parameters():
            p.requires_grad = False
        return
    if tune_last_n > 0:
        for p in model.bert.parameters():
            p.requires_grad = False
        encoder_layers = getattr(model.bert, "encoder", None)
        if encoder_layers is not None:
            layers = encoder_layers.layer
            for layer in layers[-tune_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(model.bert, "pooler") and model.bert.pooler is not None:
            for p in model.bert.pooler.parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True


def save_curves_png(history, out_dir):
    epochs = list(range(1, len(history["val_loss"]) + 1))
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.plot(epochs, history["val_f1"], label="val_f1 (macro)")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid(True, alpha=0.25)
    path = os.path.join(out_dir, "train_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    print(f"[INFO] saved curves: {path}")


def eval_full(model, dataloader, device, num_intents):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy().tolist()
            logits = model(input_ids, attn)
            pred = logits.argmax(dim=-1).cpu().numpy().tolist()
            y_true.extend(labels)
            y_pred.extend(pred)
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_intents)))
    return acc, f1, cm, y_true, y_pred


# ===== Focal Loss (multi-class) =====
class FocalLoss(torch.nn.Module):
    """
    Focal Loss for multi-class classification.
    Args:
        gamma: focusing parameter
        alpha: class-wise weights tensor (C,) or scalar; if provided, multiplies CE
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        # logits: [B, C], target: [B]
        ce = torch.nn.functional.cross_entropy(
            logits, target, weight=self.alpha, reduction="none"
        )  # per-sample CE
        pt = torch.exp(-ce)  # pt = softmax prob of the true class
        focal = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ===== Main train function =====
def main(args):
    ensure_dirs()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[INFO] device={device}")

    out_dir = str(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Data
    train_dl, valid_dl, label_map, reverse_map, train_labels, _ = build_dataloaders(
        str(args.data), args.model_name, args.max_length, args.batch_size,
        args.val_ratio, args.num_workers, device == "cuda"
    )
    num_intents = len(label_map)
    print(f"[INFO] num_intents={num_intents}")

    # Model
    model = IntentClassifier(num_intents=num_intents,
                             model_name=args.model_name,
                             dropout=args.dropout).to(device)
    freeze_encoder_layers(model, args.tune_last_n, args.freeze_encoder)

    # Optim + Sched
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = ceil(len(train_dl) / max(1, args.grad_accum)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    # Scheduler: cosine (monotonic decay) or linear
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # AMP
    try:
        scaler = torch.amp.GradScaler(
            device_type=("cuda" if device == "cuda" else "cpu"),
            enabled=(device == "cuda" and args.fp16),
        )
    except TypeError:
        from torch.cuda.amp import GradScaler as CudaGradScaler
        scaler = CudaGradScaler(enabled=(device == "cuda" and args.fp16))

    # Loss
    class_weights = maybe_build_class_weights(train_labels, num_intents, args.class_weights)
    if args.loss == "focal":
        alpha = class_weights.to(device) if class_weights is not None else None
        loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=alpha, reduction="mean")
        print(f"[INFO] Using FocalLoss(gamma={args.focal_gamma})")
    else:
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
        print("[INFO] Using CrossEntropyLoss")

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb")) if args.tensorboard and TB_AVAILABLE else None
    best_f1, patience_left = -1.0, args.patience
    history = {"val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_dl, 1):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if scaler.is_enabled():
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(input_ids, attn)
                    loss = loss_fn(logits, labels) / max(1, args.grad_accum)
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids, attn)
                loss = loss_fn(logits, labels) / max(1, args.grad_accum)
                loss.backward()

            if step % max(1, args.grad_accum) == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if args.grad_clip and args.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # ===== Validation =====
        model.eval()
        val_losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for batch in valid_dl:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids, attn)
                loss = loss_fn(logits, labels)
                val_losses.append(float(loss.item()))
                pred = logits.argmax(dim=-1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
        print(f"[Eval {epoch}] loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if writer:
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)
            writer.add_scalar("val/f1_macro", val_f1, epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # Save best (by F1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save(out_dir)
            with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
                json.dump(label_map, f, indent=2, ensure_ascii=False)
            with open(os.path.join(out_dir, "reverse_label_map.json"), "w", encoding="utf-8") as f:
                json.dump({int(v): k for k, v in label_map.items()}, f, indent=2, ensure_ascii=False)

            acc_full, f1_full, cm, y_t, y_p = eval_full(model, valid_dl, device, num_intents)
            with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(classification_report(y_t, y_p, digits=4, zero_division=0))
            pd.DataFrame(cm).to_csv(os.path.join(out_dir, "confusion_matrix.csv"), index=False)
            print(f"[INFO] saved best model to {out_dir} (f1={best_f1:.4f})")
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] no improvement for {args.patience} epochs")
                break

    save_curves_png(history, out_dir)
    if writer:
        writer.close()
    print("[DONE] training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR / "intent_training_data_expanded.csv"))
    parser.add_argument("--out", default=str(MODELS_DIR / "intent_classifier"))
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 微调策略
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--tune-last-n", type=int, default=4, help="unfreeze last N encoder layers (0 = full fine-tune or fully frozen depending on --freeze-encoder)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--scheduler", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--class-weights", choices=["none", "auto"], default="auto")
    parser.add_argument("--loss", choices=["ce", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # 训练效率
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)

    # 其他
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard", action="store_true")

    args = parser.parse_args()
    main(args)
