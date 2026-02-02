# models/intent_classifier.py
import os
import json
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertModel,
    BertTokenizerFast,
    AutoConfig,
)

# 可选：国内镜像（有需要再开启）
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# =========================
# Dataset
# =========================
class IntentDataset(Dataset):
    """
    A simple dataset wrapping texts + label ids and doing on-the-fly tokenization.
    """
    def __init__(
            self,
            texts: List[str],
            label_ids: Optional[List[int]] = None,
            model_name: Optional[str] = None,
            local_model_path: Optional[str] = None,
            max_length: int = 64,
    ):
        assert model_name or local_model_path, "Either model_name or local_model_path must be provided."
        self.texts = [str(t) if t is not None else "" for t in texts]
        self.label_ids = label_ids  # can be None for pure inference
        self.max_length = max_length

        src = local_model_path if local_model_path else model_name
        self.tokenizer = BertTokenizerFast.from_pretrained(src)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.label_ids is not None:
            item["labels"] = torch.tensor(int(self.label_ids[idx]), dtype=torch.long)
        return item


# =========================
# Model
# =========================
class IntentClassifier(nn.Module):
    """
    BERT + Linear head for intent classification.

    Features:
      - init from model_name or local_model_path
      - predict_proba / predict_label / predict_intent
      - train_one_epoch / evaluate helpers
      - save / load full stack (pytorch_model.bin + config + tokenizer)
    """
    def __init__(
            self,
            num_intents: int,
            model_name: Optional[str] = "bert-base-uncased",
            local_model_path: Optional[str] = None,
            dropout: float = 0.1,
    ):
        super().__init__()
        assert model_name or local_model_path, "Either model_name or local_model_path must be provided."
        self.num_intents = int(num_intents)

        src = local_model_path if local_model_path else model_name
        # 配置里写入 num_labels，保证分类头尺寸对齐
        config = AutoConfig.from_pretrained(src)
        config.num_labels = self.num_intents

        self.bert = BertModel.from_pretrained(src, config=config)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, self.num_intents)

        # tokenizer 用于推理
        self.tokenizer = BertTokenizerFast.from_pretrained(src)
        # 保存一下 config 以便 save()
        self.config = config

    # ---------- forward ----------
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output 在部分 BERT 变体里可能为 None，这里统一使用 CLS 表示
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            # 取 CLS token 向量
            pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    # ---------- inference ----------
    @torch.no_grad()
    def predict_proba(
            self,
            texts: Union[str, List[str]],
            device: Optional[str] = None,
            max_length: int = 64,
    ):
        """
        texts: a string or a list of strings
        return: np.ndarray, shape (N, num_intents)
        """
        self.eval()
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        logits = self.forward(input_ids, attn)
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        return prob

    @torch.no_grad()
    def predict_label(
            self,
            text: str,
            device: Optional[str] = None,
            max_length: int = 64,
    ) -> int:
        proba = self.predict_proba(text, device=device, max_length=max_length)[0]
        return int(proba.argmax())

    @torch.no_grad()
    def predict_intent(
            self,
            text: str,
            reverse_label_map: Optional[Dict[int, str]] = None,
            device: Optional[str] = None,
            max_length: int = 64,
    ) -> Union[int, str]:
        lid = self.predict_label(text, device=device, max_length=max_length)
        return reverse_label_map.get(lid, lid) if reverse_label_map else lid

    # ---------- training helpers ----------
    def train_one_epoch(
            self,
            dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            device: Optional[str] = None,
            loss_fn: Optional[nn.Module] = None,
            grad_clip: Optional[float] = 1.0,
    ) -> float:
        """
        Returns: average loss
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()

        loss_fn = loss_fn or nn.CrossEntropyLoss()
        total_loss, steps = 0.0, 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = self.forward(input_ids, attn)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        return total_loss / max(steps, 1)

    @torch.no_grad()
    def evaluate(
            self,
            dataloader: DataLoader,
            device: Optional[str] = None,
            loss_fn: Optional[nn.Module] = None,
    ) -> Tuple[float, float, float]:
        """
        Returns: (avg_loss, accuracy, macro_f1)
        """
        from sklearn.metrics import accuracy_score, f1_score

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        loss_fn = loss_fn or nn.CrossEntropyLoss()
        total_loss, steps = 0.0, 0
        y_true, y_pred = [], []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = self.forward(input_ids, attn)
            loss = loss_fn(logits, labels)

            total_loss += float(loss.item())
            steps += 1

            pred = logits.argmax(dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

        avg_loss = total_loss / max(steps, 1)
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
        return avg_loss, acc, f1

    # ---------- persistence ----------
    def save(self, path: str):
        """
        Save:
          - pytorch_model.bin (state_dict)
          - config.json / tokenizer files
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        # 让 config 中的 num_labels 与当前 head 对齐
        self.config.num_labels = self.num_intents
        self.config.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[IntentClassifier] model saved to: {path}")

    @classmethod
    def load(
            cls,
            path: str,
            num_intents: int,
    ) -> "IntentClassifier":
        """
        Load from a saved directory created by `save()`.
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model directory not found: {path}")

        # 用本地目录初始化 backbone 与 tokenizer，再加载分类头权重
        model = cls(num_intents=num_intents, local_model_path=path)
        state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        return model
