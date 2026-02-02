# test/test_intent_classifier.py
"""
检查：intent 模型加载 & 预测效果
"""
import os
import json
import numpy as np
import torch

import train._bootstrap  # 确保 sys.path 正确
from utils.paths import DATA_DIR, MODELS_DIR
from utils.data_utils import load_intent_data
from models.intent_classifier import IntentClassifier


MODEL_DIR = MODELS_DIR / "intent_classifier"
DATA_CSV  = DATA_DIR / "intent_training_data_extended.csv"

TEST_TEXTS = [
    "I'm looking for a raincoat",
    "What's the price range for boots?",
    "Any elegant jackets?",
    "show me sporty jackets under $100",
    "looking for vintage dresses",
    "Hi, how are you?",
    "Goodbye for now",
]

TOPK = 3  # 显示前 K 个概率最高的意图；设 0 关闭


def load_label_maps():
    """优先用模型目录里的映射；其次用数据文件动态构建；最后降级为空映射。"""
    rev_path = MODEL_DIR / "reverse_label_map.json"
    if rev_path.exists():
        with rev_path.open("r", encoding="utf-8") as f:
            reverse_map = {int(k): v for k, v in json.load(f).items()}
        label_map = {v: k for k, v in reverse_map.items()}
        return label_map, reverse_map, len(label_map)

    try:
        df, label_map, reverse_map = load_intent_data(str(DATA_CSV))
        return label_map, reverse_map, len(label_map)
    except Exception as e:
        print(f"[WARN] 无法从数据构建标签映射：{e}")
        return {}, {}, 5  # 最小可运行降级


def pretty_topk(prob_row, reverse_map, k=3):
    if k <= 0:
        return ""
    idxs = np.argsort(prob_row)[::-1][:k]
    parts = []
    for i in idxs:
        label = reverse_map.get(int(i), str(i)) if reverse_map else str(i)
        parts.append(f"{label}: {prob_row[i]:.3f}")
    return " | " + "  ".join(parts)


def test_intent_classifier():
    # 设备信息
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载标签映射
    label_map, reverse_map, num_intents = load_label_maps()
    print(f"意图类别数: {num_intents}")
    if label_map:
        print(f"意图标签映射: {label_map}")

    # 加载或初始化模型
    ckpt_path = MODEL_DIR / "pytorch_model.bin"
    if ckpt_path.exists():
        print(f"加载已训练模型: {MODEL_DIR}")
        classifier = IntentClassifier.load(str(MODEL_DIR), num_intents)
    else:
        print("未找到已训练模型，初始化新模型（随机权重，仅用于联调）")
        classifier = IntentClassifier(num_intents=num_intents)

    # 批量预测（更快）
    print("\n测试预测：")
    probs = classifier.predict_proba(TEST_TEXTS)  # shape: [N, num_intents]
    for text, prob in zip(TEST_TEXTS, probs):
        top_id = int(np.argmax(prob))
        intent_label = reverse_map.get(top_id, f"label_{top_id}")
        extra = pretty_topk(prob, reverse_map, TOPK)
        print(f"文本: '{text}'")
        print(f"预测意图: {intent_label}{extra}\n")


if __name__ == "__main__":
    test_intent_classifier()
