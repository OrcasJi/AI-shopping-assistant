# test/test_style_ner.py
"""
Style NER 测试：
- 打印设备/模型/标签信息
- 显示验证集标签分布（data/style_val.json）
- 风格抽取（置信度）
- 读取训练时保存的混淆矩阵维度
"""
import os
import json
import numpy as np
import torch

import train._bootstrap  # 确保 sys.path 正确
from utils.paths import DATA_DIR, MODELS_DIR
from models.style_ner import StyleNER


VAL_JSON = DATA_DIR / "style_val.json"
MODEL_DIR = MODELS_DIR / "style_ner"

SAMPLE_SENTENCES = [
    "Looking for a vintage leather jacket for fall.",
    "Need a casual summer dress.",
    "Show me elegant evening gowns and classic coats.",
    "I want sporty sneakers and a minimalist backpack.",
    "Any ideas for office wear?",
]


def _print_val_label_distribution(val_path):
    if not val_path.exists():
        print(f"[INFO] 验证集文件不存在：{val_path}")
        return
    with open(val_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    non_o_count = {}
    samples_with_style = 0
    for s in data:
        has_style = False
        for lab in s.get("labels", []):
            if lab != "O":
                non_o_count[lab] = non_o_count.get(lab, 0) + 1
                has_style = True
        if has_style:
            samples_with_style += 1

    print(f"[VAL] 总样本数: {len(data)}")
    print(f"[VAL] 含 STYLE 的样本数: {samples_with_style}")
    print(f"[VAL] 非 'O' 标签计数: {non_o_count if non_o_count else '<none>'}")


def test_style_ner():
    # 设备信息
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载或初始化模型
    if (MODEL_DIR / "pytorch_model.bin").exists():
        print(f"[Model] 加载已训练模型: {MODEL_DIR}")
        ner = StyleNER.load(str(MODEL_DIR))
    else:
        print("[Model] 未找到训练权重，初始化新模型（随机权重，仅用于联调）")
        ner = StyleNER()

    # 标签信息
    print(f"[Labels] label_map: {ner.label_map}")

    # 验证集标签分布（辅助排查“单标签/全O”）
    _print_val_label_distribution(VAL_JSON)

    # 示例句子抽取（置信度）
    print("\n测试抽取：")
    for s in SAMPLE_SENTENCES:
        results = ner.extract_styles(s, return_confidence=True, threshold=0.0)
        print(f"Text: {s}")
        if results:
            # results: List[Tuple[str, float]]
            for style, prob in results:
                print(f"  - {style} ({prob:.3f})")
        else:
            print("  - <no style found>")

    # 训练阶段若导出了混淆矩阵，这里读一下维度（方便快速判断是否 3x3）
    cm_path = MODEL_DIR / "confusion_matrix.csv"
    if cm_path.exists():
        try:
            import pandas as pd
            cm = pd.read_csv(cm_path)
            print(f"\n[CM] confusion_matrix.csv 形状: {cm.shape}（期望 3x3，对应 O/B/I）")
        except Exception as e:
            print(f"[WARN] 读取混淆矩阵失败：{e}")
    else:
        print("\n[HINT] 未检测到训练阶段导出的混淆矩阵。"
              "若 sklearn 报 'single label' 警告，可在训练时保存时使用："
              "confusion_matrix(all_labels, all_preds, labels=[0,1,2]) 固定维度。")
