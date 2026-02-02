# utils/style_data_prep.py
"""
Build style NER dataset (JSON) from a product catalog CSV.
Outputs:
  - data/style_train.json
  - data/style_val.json

Usage:
  python utils/style_data_prep.py \
      --csv data/Shopping_product_catalog.csv \
      --out-train data/style_train.json \
      --out-val data/style_val.json \
      --test-size 0.2
"""
import os
import sys
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple

# 让脚本无论从哪运行都能定位项目根
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
from sklearn.model_selection import train_test_split
from utils.paths import DATA_DIR

# 基础风格词表（全小写，可自行扩展）
DEFAULT_STYLE_LEXICON = [
    "sporty", "casual", "vintage", "elegant", "formal", "modern",
    "retro", "classic", "minimalist", "chic", "bohemian", "street",
    "preppy", "luxury", "athleisure", "romantic", "punk", "gothic",
    "oversized", "slim", "fitted", "loose", "grunge", "artsy",
    "denim", "leather", "suede", "knit", "silk", "linen", "velvet",
    "summer", "winter", "fall", "spring", "beach", "party", "office"
]

TEXT_COL_CANDIDATES = ["product_name", "name", "title"]
DESC_COL_CANDIDATES = ["description", "desc", "details"]
STYLE_COL_CANDIDATES = ["style", "styles", "style_tags"]


def simple_word_tokenize(text: str) -> Tuple[List[str], List[int]]:
    """极简英文 token 切分并返回每个 token 的结束字符索引（右开区间）。"""
    tokens, ends = [], []
    for m in re.finditer(r"\w+", str(text)):
        tokens.append(m.group(0))
        ends.append(m.end())
    return tokens, ends


def tag_with_lexicon(tokens: List[str], style_terms: List[str]) -> List[str]:
    """用词表对 tokens 打 BIO（B-STYLE/I-STYLE）标签；优先匹配多词短语。"""
    labels = ["O"] * len(tokens)
    norm_terms = sorted([t.strip().lower() for t in style_terms if t and t.strip()],
                        key=lambda x: -len(x.split()))
    low_tokens = [t.lower() for t in tokens]
    i = 0
    while i < len(tokens):
        matched = False
        for term in norm_terms:
            parts = term.split()
            L = len(parts)
            if L == 0 or i + L > len(tokens):
                continue
            if low_tokens[i:i+L] == parts:
                labels[i] = "B-STYLE"
                for j in range(1, L):
                    labels[i + j] = "I-STYLE"
                i += L
                matched = True
                break
        if not matched:
            i += 1
    return labels


def parse_style_terms_from_row(row: pd.Series, style_lexicon: List[str]) -> List[str]:
    """优先从 style 列取；否则从文本中命中词表。"""
    # 1) 显式 style 列
    for col in STYLE_COL_CANDIDATES:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            raw = str(row[col]).lower()
            parts = re.split(r"[,\|;/]+", raw)
            return [p.strip() for p in parts if p and p.strip()]
    # 2) 从标题/描述命中词表
    text_pieces = []
    for col in TEXT_COL_CANDIDATES + DESC_COL_CANDIDATES:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            text_pieces.append(str(row[col]))
    joined = " ".join(text_pieces).lower()
    terms = []
    for t in style_lexicon:
        if re.search(rf"\b{re.escape(t)}\b", joined):
            terms.append(t)
    return terms


def build_sample_from_text(text: str, style_terms: List[str]) -> Dict:
    tokens, ends = simple_word_tokenize(text)
    if not tokens:
        return {}
    labels = tag_with_lexicon(tokens, style_terms)
    return {"text": str(text), "labels": labels, "word_ends": ends}


def mine_style_terms_from_catalog(df: pd.DataFrame, top_k: int = 200) -> List[str]:
    """
    从 product_name/title/description 自动挖风格候选（形容词与常见二词短语）。
    简易启发式，目的是扩展词表覆盖面。
    """
    import collections
    texts = []
    cols = [c for c in ["product_name", "name", "title", "description", "desc", "details"] if c in df.columns]
    for _, row in df.iterrows():
        parts = [str(row[c]) for c in cols if pd.notna(row[c])]
        if parts:
            texts.append(" ".join(parts).lower())
    blob = " ".join(texts)
    adj_candidates = re.findall(r"\b[a-z]{3,}(?:y|ic|al|ish|ed|ist)\b", blob)
    words = re.findall(r"[a-z]+", blob)
    bigrams = [" ".join(words[i:i+2]) for i in range(len(words) - 1)]
    cnt = collections.Counter(adj_candidates + bigrams)
    stop = {"for", "and", "the", "with", "in", "on", "of", "to", "a", "an", "men", "women"}
    terms = []
    for term, _ in cnt.most_common():
        if any(t in stop for t in term.split()):
            continue
        if len(term) < 4:
            continue
        terms.append(term)
        if len(terms) >= top_k:
            break
    return terms


def create_style_dataset_from_catalog(
        csv_path: Path,
        out_train: Path,
        out_val: Path,
        style_lexicon: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        min_chars: int = 12,
        min_style_in_val: int = 30,
):
    """从商品 CSV 生成 style NER 数据集（train/val 两个 JSON 文件）"""
    style_lexicon = style_lexicon or DEFAULT_STYLE_LEXICON

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    def select_text(row: pd.Series) -> str:
        for c in TEXT_COL_CANDIDATES:
            if c in row and pd.notna(row[c]) and str(row[c]).strip():
                return str(row[c])
        for c in DESC_COL_CANDIDATES:
            if c in row and pd.notna(row[c]) and str(row[c]).strip():
                return str(row[c])
        vals = [str(row[c]) for c in (TEXT_COL_CANDIDATES + DESC_COL_CANDIDATES) if c in row and pd.notna(row[c])]
        return " ".join(vals)

    samples = []
    for _, row in df.iterrows():
        text = select_text(row)
        if not text or len(str(text)) < min_chars:
            continue
        terms = parse_style_terms_from_row(row, style_lexicon)
        sample = build_sample_from_text(str(text), terms)
        if sample:
            samples.append(sample)

    # 去重
    uniq = {}
    for s in samples:
        key = s["text"].strip().lower()
        if key not in uniq:
            uniq[key] = s
    samples = list(uniq.values())
    if not samples:
        raise RuntimeError("No valid samples built from CSV. Check columns/lexicon.")

    def has_style(sample): return any(l != "O" for l in sample["labels"])
    stratify_vec = [int(has_style(s)) for s in samples]
    train, val = train_test_split(
        samples, test_size=test_size, random_state=random_state,
        stratify=stratify_vec if len(set(stratify_vec)) > 1 else None
    )

    # 兜底：确保验证集至少有 min_style_in_val 条含 STYLE 的样本
    if sum(has_style(s) for s in val) < min_style_in_val:
        styled = [s for s in train if has_style(s)]
        need = min(min_style_in_val, len(styled))
        val = val + styled[:need]
        train = [s for s in train if s not in styled[:need]]

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(out_val, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    def count_style_tokens(data):
        return sum(sum(1 for l in s["labels"] if l != "O") for s in data)

    print(f"[DONE] samples: total={len(samples)} | train={len(train)} | val={len(val)}")
    print(f"[INFO] style tokens: train={count_style_tokens(train)} | val={count_style_tokens(val)}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DATA_DIR / "Shopping_product_catalog_expanded.csv"))
    parser.add_argument("--out-train", default=str(DATA_DIR / "style_train.json"))
    parser.add_argument("--out-val", default=str(DATA_DIR / "style_val.json"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-mine", action="store_true", help="自动从 CSV 文本中挖词表并合并")
    args = parser.parse_args()

    # 读取 CSV
    df = pd.read_csv(args.csv, encoding="utf-8", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    style_lexicon = set(DEFAULT_STYLE_LEXICON)
    if args.auto_mine:
        auto_terms = set(mine_style_terms_from_catalog(df, top_k=200))
        style_lexicon |= auto_terms
        print(f"[INFO] auto-mined {len(auto_terms)} terms, merged lexicon size={len(style_lexicon)}")

    create_style_dataset_from_catalog(
        csv_path=Path(args.csv),
        out_train=Path(args.out_train),
        out_val=Path(args.out_val),
        style_lexicon=sorted(style_lexicon),
        test_size=args.test_size,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
