# test/test_data_utils.py
import pandas as pd
import pytest
from utils.data_utils import (
    preprocess_text,
    load_intent_data,
    split_data,
    load_product_catalog,
)

# ---------------------------
# Unit tests: preprocess_text
# ---------------------------

def test_preprocess_text_basic():
    s = "  Show me  Jackets  UNDER  $ 1,299 !!  "
    got = preprocess_text(s)
    # 期望：小写、去掉多余空格、去除逗号、货币连写
    assert got == "show me jackets under $1299 !!".replace(" !!", " !!").strip().replace("  ", " ")

def test_preprocess_text_currency_spacing():
    s = "around  £  250  and  € 300"
    got = preprocess_text(s)
    # 期望：£、€ 与数字连写
    assert "£250" in got and "€300" in got


# -----------------------------------------
# Unit tests: load_intent_data + split_data
# -----------------------------------------

def test_load_intent_data_and_split(tmp_path):
    # 造一个小的意图数据集（英文环境）
    csv = tmp_path / "intent_small.csv"
    pd.DataFrame(
        {
            "text": [
                "find sporty jackets under $100",
                "show me formal shoes",
                "i want vintage dresses",
                "under $80 casual boots",
                "recommend casual sneakers",
                "looking for vintage skirts",
                "over $200 sporty jackets"
            ],
            "intent": [
                "ask_product",
                "ask_product",
                "ask_style",
                "ask_price",
                "ask_product",
                "ask_style",
                "ask_price"
            ],
        }
    ).to_csv(csv, index=False)

    df, label_map, reverse_map = load_intent_data(str(csv))
    # 基本结构校验
    assert set(["text", "intent", "label_id"]).issubset(df.columns)
    assert len(label_map) == len(set(df["intent"]))
    # 反向映射一致
    for k, v in label_map.items():
        assert reverse_map[v] == k

    # 分层切分
    train_df, valid_df = split_data(df, test_size=0.33, random_state=7, stratify=True)
    assert len(train_df) + len(valid_df) == len(df)
    # 每份数据都应有 label_id
    assert "label_id" in train_df.columns and "label_id" in valid_df.columns
    # 类别分布不至于塌缩（简单检查：两端至少包含一种相同类别）
    assert set(train_df["label_id"]).intersection(set(valid_df["label_id"]))


def test_load_intent_data_with_alt_column_names(tmp_path):
    # 列名不标准时（Label / Utterance），应能自动适配
    csv = tmp_path / "intent_alt_cols.csv"
    pd.DataFrame(
        {
            "Utterance": ["show me dresses", "under $100 sneakers"],
            "Label": ["ask_product", "ask_price"],
        }
    ).to_csv(csv, index=False)

    df, label_map, reverse_map = load_intent_data(str(csv))
    assert set(["text", "intent", "label_id"]).issubset(df.columns)
    assert len(label_map) == 2


# -------------------------------------
# Unit tests: load_product_catalog (CSV)
# -------------------------------------

def test_load_product_catalog_with_commas_and_symbols(tmp_path):
    csv = tmp_path / "catalog_small.csv"
    pd.DataFrame(
        {
            "name": ["Sporty Jacket A", "Vintage Dress B", "Casual Boots C"],
            "category": ["jacket", "dress", "boots"],
            "style": ["sporty", "vintage", "casual"],
            # 各种价格写法：带逗号、小数、带货币符号
            "price": ["1,299", "$129.9", "80"],
        }
    ).to_csv(csv, index=False)

    cat = load_product_catalog(str(csv))
    # 列齐全
    assert set(["name", "category", "style", "price"]).issubset(cat.columns)

    # 价格应为数值，并且按预期解析
    assert pd.api.types.is_numeric_dtype(cat["price"])
    # 1,299 -> 1299
    assert int(cat.loc[cat["name"] == "Sporty Jacket A", "price"].iloc[0]) == 1299
    # $129.9 -> 129.9
    assert pytest.approx(float(cat.loc[cat["name"] == "Vintage Dress B", "price"].iloc[0]), rel=1e-6) == 129.9


def test_load_product_catalog_missing_cols(tmp_path):
    # 缺列时应报错并给出明确提示
    csv = tmp_path / "bad_catalog.csv"
    pd.DataFrame(
        {"title": ["X"], "type": ["dress"], "styles": ["vintage"], "amount": ["100"]}
    ).to_csv(csv, index=False)

    with pytest.raises(ValueError):
        load_product_catalog(str(csv))
