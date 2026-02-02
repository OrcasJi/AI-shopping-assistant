import os
import re
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split

# ----------------------------
# Text preprocessing
# ----------------------------
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^0-9a-zA-Z%$€£.,:;!?/'\"&()+\- ]+")

def preprocess_text(text: str) -> str:
    """
    Minimal, English-first normalization:
    - lowercasing
    - unify whitespace
    - strip non-ASCII punctuation/symbols
    - normalize currency forms like `$ 1,299` -> `$1299`
    """
    if text is None:
        return ""
    t = str(text).strip().lower()

    # keep basic ascii & currency/punctuation; drop others
    t = _PUNCT_RE.sub(" ", t)

    # normalize commas in numbers: 1,299 -> 1299
    t = re.sub(r"(?<=\d),(?=\d)", "", t)

    # unify '$ 100' -> '$100', same for £/€
    t = re.sub(r"([$£€])\s+(\d)", r"\1\2", t)

    # collapse whitespaces
    t = _WS_RE.sub(" ", t).strip()
    return t


# ----------------------------------------------------
# Intent dataset loading + label maps (project-aligned)
# ----------------------------------------------------
def load_intent_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Load intent CSV and return:
      - df with columns: ['text', 'intent', 'label_id']
      - label_map: {intent_str -> id_int}
      - reverse_label_map: {id_int -> intent_str}

    Expected CSV columns (minimum): 'text', 'intent'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Intent data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # basic schema check
    required = {"text", "intent"}
    missing = required - set(df.columns.str.lower())
    # try case-insensitive mapping
    lower_map = {c.lower(): c for c in df.columns}
    if missing:
        # maybe columns are 'Text'/'Label' etc. try to adapt
        # map common alternatives
        alt_map = {}
        if "text" not in lower_map and "utterance" in lower_map:
            alt_map["text"] = lower_map["utterance"]
        if "intent" not in lower_map:
            for cand in ["label", "intent_label", "class", "category"]:
                if cand in lower_map:
                    alt_map["intent"] = lower_map[cand]
                    break
        # apply alternative mapping if found
        for std, real in alt_map.items():
            df.rename(columns={real: std}, inplace=True)
        # recompute missing
        missing = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(
                f"CSV must have columns {required}. Missing: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    # enforce exact lowercase column names
    # (keep original names if already exact)
    col_map = {c: c.lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # drop nulls and preprocess
    df = df.dropna(subset=["text", "intent"]).copy()
    df["text"] = df["text"].apply(preprocess_text)

    # build label maps (stable sort)
    intents: List[str] = sorted(df["intent"].astype(str).unique())
    label_map: Dict[str, int] = {lab: i for i, lab in enumerate(intents)}
    reverse_label_map: Dict[int, str] = {i: lab for lab, i in label_map.items()}

    # attach numeric label id (for training)
    df["label_id"] = df["intent"].map(label_map)

    return df, label_map, reverse_label_map


def split_data(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/valid with (optional) stratification by 'label_id'.
    """
    if "label_id" not in df.columns:
        raise ValueError("Dataframe must contain 'label_id' column. Call load_intent_data() first.")

    strat = df["label_id"] if stratify else None
    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
        shuffle=True,
    )
    # reset index for cleanliness
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


# ------------------------------------------------
# Product catalog loader (used by searcher module)
# ------------------------------------------------
def load_product_catalog(file_path: str) -> pd.DataFrame:
    """
    Load product catalog CSV expected by ProductSearcher.

    Required columns: ['name', 'category', 'style', 'price']
    - drop rows with missing essentials
    - coerce price to numeric
    - normalize 'style' as lowercase
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Product catalog file not found: {file_path}")

    df = pd.read_csv(file_path)

    required = {"name", "category", "style", "price"}
    lower_map = {c.lower(): c for c in df.columns}
    missing = required - set(lower_map.keys())
    if missing:
        # try to be forgiving with common alternatives
        alt = {}
        if "price" not in lower_map:
            for cand in ["amount", "sale_price", "usd", "value"]:
                if cand in lower_map:
                    alt["price"] = lower_map[cand]
                    break
        if "style" not in lower_map and "styles" in lower_map:
            alt["style"] = lower_map["styles"]
        if alt:
            df.rename(columns=alt, inplace=True)
            lower_map = {c.lower(): c for c in df.columns}
            missing = required - set(lower_map.keys())
    if missing:
        raise ValueError(
            f"Catalog must have columns {required}. Missing: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # normalize columns to exact names
    df.rename(
        columns={lower_map["name"]: "name",
                 lower_map["category"]: "category",
                 lower_map["style"]: "style",
                 lower_map["price"]: "price"},
        inplace=True
    )

    # clean rows
    df = df.dropna(subset=["name", "category", "style", "price"]).copy()
    # coerce price
    # handle numbers with commas like "1,299"
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"(?<=\d),(?=\d)", "", regex=True)
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0]
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    # normalize style
    df["style"] = df["style"].astype(str).str.lower().str.strip()

    return df
