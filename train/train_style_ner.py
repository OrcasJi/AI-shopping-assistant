# train/train_style_ner.py
# -*- coding: utf-8 -*-
"""
Train BERT Style NER model with robust evaluation.
"""
import os
import sys
from pathlib import Path

# --- 自动切到项目根 ---
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
from transformers import get_linear_schedule_with_warmup

from utils.paths import DATA_DIR, MODELS_DIR, ensure_dirs
from models.style_ner import StyleNER, StyleNERDataset

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import ceil
from collections import Counter

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_curves_png(history, out_dir):
    epochs = list(range(1, len(history["val_loss"]) + 1))
    plt.figure(figsize=(10, 5.2))
    plt.plot(epochs, history["val_loss"], label="val_loss")
    if history.get("val_f1_weighted"):
        plt.plot(epochs, history["val_f1_weighted"], label="val_f1 (weighted)")
    if history.get("val_f1_no_o"):
        plt.plot(epochs, history["val_f1_no_o"], label="val_f1_no_o (macro, B/I)")
    plt.xlabel("epoch"); plt.legend(); plt.grid(True, alpha=0.25)
    path = os.path.join(out_dir, "train_curves.png")
    plt.tight_layout(); plt.savefig(path, dpi=140)
    print(f"[INFO] saved curves: {path}")


def load_style_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_token_labels(dataset: StyleNERDataset) -> Counter:
    cnt = Counter()
    for i in range(len(dataset)):
        y = dataset[i]["labels"].numpy().tolist()
        for t in y:
            if t != -100:
                cnt[t] += 1
    return cnt


def has_positive_tokens(dataset: StyleNERDataset) -> bool:
    for i in range(len(dataset)):
        ys = dataset[i]["labels"].numpy()
        # 非 -100 且 非 O(0) 的 token 视为正样本
        if np.any((ys != -100) & (ys != 0)):
            return True
    return False


def pretty_dist(counter: Counter, label_map: dict) -> str:
    rev = {v: k for k, v in label_map.items()}
    total = sum(counter.values()) or 1
    parts = []
    for lid in [0, 1, 2]:
        parts.append(f"{rev.get(lid, lid)}={counter.get(lid,0)} ({counter.get(lid,0)/total:.2%})")
    return ", ".join(parts)


def main(args):
    ensure_dirs()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[INFO] device={device}")

    out_dir = str(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # ===== Load data =====
    train_data = load_style_data(args.train_data)
    val_data = load_style_data(args.val_data) if args.val_data else None

    # ===== Model & tokenizer =====
    model = StyleNER(model_name=args.model_name, num_labels=3, local_model_path=out_dir)
    tokenizer = model.tokenizer
    label_map = model.label_map

    # （可选）减少高频风格词被切词：按需打开
    # extra_tokens = ["sporty", "minimalist", "athleisure", "preppy", "grunge", "boho", "streetwear"]
    # model.add_extra_tokens(extra_tokens)

    # ===== Datasets & loaders =====
    train_ds = StyleNERDataset(train_data, tokenizer, label_map, max_length=args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_dl = None
    if val_data:
        val_ds = StyleNERDataset(val_data, tokenizer, label_map, max_length=args.max_length)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ===== Print distributions =====
    tr_cnt = count_token_labels(train_ds)
    print("[Train Token Dist]", pretty_dist(tr_cnt, label_map))
    if val_dl:
        va_cnt = count_token_labels(val_ds)
        print("[Val   Token Dist]", pretty_dist(va_cnt, label_map))
        if not has_positive_tokens(val_ds):
            print("[WARN] Validation set has NO B/I tokens (only 'O'). "
                  "Weighted-F1 may be ~1.0 even for trivial models. "
                  "Consider a stratified split ensuring B/I presence.")

    # ===== Optimizer & sched =====
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = ceil(len(train_dl) / max(1, args.grad_accum)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ===== AMP =====
    try:
        scaler = torch.amp.GradScaler(enabled=(device == "cuda" and args.fp16))
        autocast_ctx = torch.amp.autocast if device == "cuda" else torch.cpu.amp.autocast
    except Exception:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler(enabled=(device == "cuda" and args.fp16))
        autocast_ctx = autocast

    # ===== Class weights（基于训练集逆频）=====
    total_tok = max(1, sum(tr_cnt.values()))
    inv = {k: total_tok / (v if v > 0 else 1) for k, v in tr_cnt.items()}
    scale = np.mean(list(inv.values()))
    class_weights = torch.tensor([inv.get(i, 1.0) / scale for i in range(3)], dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    best_score, patience_left = -1.0, args.patience
    history = {"val_loss": [], "val_f1_weighted": [], "val_f1_no_o": []}
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb")) if args.tensorboard and TB_AVAILABLE else None

    for epoch in range(1, args.epochs + 1):
        model.model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_dl, 1):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if scaler.is_enabled():
                with autocast_ctx(device_type="cuda", enabled=True):
                    outputs = model.model(input_ids, attention_mask=attn)
                    loss = loss_fn(outputs.logits.view(-1, 3), labels.view(-1)) / max(1, args.grad_accum)
                scaler.scale(loss).backward()
            else:
                outputs = model.model(input_ids, attention_mask=attn)
                loss = loss_fn(outputs.logits.view(-1, 3), labels.view(-1)) / max(1, args.grad_accum)
                loss.backward()

            if step % max(1, args.grad_accum) == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # ===== Validation =====
        if val_dl:
            model.model.eval()
            total_loss, all_preds, all_labels = 0.0, [], []
            with torch.no_grad():
                for batch in val_dl:
                    input_ids = batch["input_ids"].to(device)
                    attn = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model.model(input_ids, attention_mask=attn, labels=None)
                    # 手动用同一 loss_fn（带 class_weights）
                    loss = loss_fn(outputs.logits.view(-1, 3), labels.view(-1))
                    total_loss += loss.item()

                    preds = torch.argmax(outputs.logits, dim=2).cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    for i in range(preds.shape[0]):
                        valid_idx = labels_np[i] != -100
                        all_preds.extend(preds[i][valid_idx])
                        all_labels.extend(labels_np[i][valid_idx])

            avg_loss = total_loss / max(1, len(val_dl))

            # F1（weighted）+ 只看 B/I 的 macro F1
            report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
            f1_weighted = report["weighted avg"]["f1-score"]
            # labels=[1,2]：只看 B-STYLE / I-STYLE
            try:
                f1_no_o = f1_score(all_labels, all_preds, labels=[1, 2], average="macro", zero_division=0)
            except Exception:
                f1_no_o = 0.0

            history["val_loss"].append(avg_loss)
            history["val_f1_weighted"].append(f1_weighted)
            history["val_f1_no_o"].append(f1_no_o)

            print(f"[Eval {epoch}] loss={avg_loss:.4f}  f1_weighted={f1_weighted:.4f}  f1_no_o={f1_no_o:.4f}")

            if writer:
                writer.add_scalar("val/loss", avg_loss, epoch)
                writer.add_scalar("val/f1_weighted", f1_weighted, epoch)
                writer.add_scalar("val/f1_no_o_macro", f1_no_o, epoch)

            # ===== 选择指标：用 f1_no_o 作为早停与最佳保存 =====
            score_for_selection = f1_no_o

            # Save best
            if score_for_selection > best_score:
                best_score = score_for_selection
                model.save(out_dir)
                # 保存报告/混淆矩阵
                with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
                    f.write(classification_report(all_labels, all_preds, digits=4, zero_division=0))
                    f.write(f"\n\nf1_no_o_macro(B/I only): {f1_no_o:.6f}\n")
                    f.write(f"f1_weighted: {f1_weighted:.6f}\n")
                pd.DataFrame(
                    confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
                ).to_csv(os.path.join(out_dir, "confusion_matrix.csv"), index=False)
                print(f"[INFO] saved best to {out_dir} (f1_no_o={best_score:.4f})")
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[EARLY STOP] no improvement of f1_no_o for {args.patience} epochs")
                    break

    save_curves_png(history, out_dir)
    if writer:
        writer.close()
    print("[DONE] training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=str(DATA_DIR / "style_train.json"))
    parser.add_argument("--val-data", default=str(DATA_DIR / "style_val.json"))
    parser.add_argument("--out", default=str(MODELS_DIR / "style_ner"))
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()
    main(args)
