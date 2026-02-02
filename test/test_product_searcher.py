# test/test_product_searcher.py
import pandas as pd
from utils.product_searcher import ProductSearcher

def _build_catalog(tmp_path):
    csv = tmp_path / "catalog.csv"
    pd.DataFrame({
        "name": ["Sporty Jacket A", "Vintage Dress B", "Casual Boots C", "Formal Shoes D"],
        "category": ["jacket", "dress", "boots", "shoes"],
        "style": ["sporty", "vintage", "casual", "formal"],
        "price": ["$120", "1,299", "80", "150"]
    }).to_csv(csv, index=False)
    return str(csv)

def test_extract_price_range(tmp_path):
    ps = ProductSearcher(_build_catalog(tmp_path))
    assert ps.extract_price_range("under $100") == (None, 100.0)
    assert ps.extract_price_range("over 80") == (80.0, None)
    assert ps.extract_price_range("between 50 and 120") == (50.0, 120.0)
    lo, hi = ps.extract_price_range("around 200")
    assert 150 <= lo <= 170 and 230 <= hi <= 250  # ±20% window
    lo2, hi2 = ps.extract_price_range("budget 1k")
    assert lo2 < 1000 < hi2

def test_fuzzy_style_match(tmp_path):
    ps = ProductSearcher(_build_catalog(tmp_path))
    assert ps.fuzzy_style_match("sport") == "sporty"
    assert ps.fuzzy_style_match("vintage") == "vintage"
    assert ps.fuzzy_style_match("FoRmAl") == "formal"

def test_search_products_with_price_and_style(tmp_path):
    ps = ProductSearcher(_build_catalog(tmp_path))
    res = ps.search_products(category="jacket", min_price=100, max_price=130, styles=["sporty"])
    assert len(res) == 1
    assert res.iloc[0]["name"] == "Sporty Jacket A"

def test_search_products_no_price_rank_by_style_then_name(tmp_path):
    ps = ProductSearcher(_build_catalog(tmp_path))
    res = ps.search_products(styles=["vintage"], query_hint="dress")
    assert len(res) >= 1
    assert res.iloc[0]["style"] == "vintage"
