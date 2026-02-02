# utils/product_searcher.py
"""
Product catalog search
"""

import os
import re
from typing import Optional, List, Tuple, Any

import pandas as pd
from rapidfuzz import fuzz


class ProductSearcher:
    """Product catalog search (fuzzy matching + weighted ranking)."""

    def __init__(self, catalog_path: str):
        self.catalog = self.load_catalog(catalog_path)

    # ---------- load & normalize ----------
    def load_catalog(self, catalog_path: str) -> pd.DataFrame:
        if not os.path.isfile(catalog_path):
            raise FileNotFoundError(f"Catalog file not found: {catalog_path}")
        df = pd.read_csv(catalog_path)

        # Basic schema tolerance
        lower = {c.lower(): c for c in df.columns}
        need = {"name", "category", "style", "price"}
        if not need.issubset(set(lower.keys())):
            aliases = {
                "name": ["title", "product", "product_name"],
                "category": ["cate", "type", "class"],
                "style": ["styles", "tag", "tags"],
                "price": ["amount", "usd", "value", "sale_price"],
            }
            mapping = {}
            for k in need:
                if k in lower:
                    mapping[k] = lower[k]
                else:
                    for a in aliases.get(k, []):
                        if a in lower:
                            mapping[k] = lower[a]
                            break
            if set(mapping.keys()) != need:
                raise ValueError(f"Catalog must contain columns {need}. Got: {list(df.columns)}")
            df = df.rename(columns={
                mapping["name"]: "name",
                mapping["category"]: "category",
                mapping["style"]: "style",
                mapping["price"]: "price",
            })
        else:
            df = df.rename(columns={
                lower["name"]: "name",
                lower["category"]: "category",
                lower["style"]: "style",
                lower["price"]: "price",
            })

        # Clean rows
        df = df.dropna(subset=["name", "category", "style", "price"]).copy()

        # Price: remove commas, keep numeric
        df["price"] = (
            df["price"].astype(str)
            .str.replace(r"(?<=\d),(?=\d)", "", regex=True)
            .str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0]
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])

        # Normalize style/name
        df["style"] = df["style"].astype(str).str.lower().str.strip()
        df["name"] = df["name"].astype(str).str.strip()
        df["category"] = df["category"].astype(str).str.strip()
        return df.reset_index(drop=True)

    # ---------- scoring helpers ----------
    def _name_hint_score(self, row_name: str, query: str) -> int:
        if not query:
            return 0
        return fuzz.partial_ratio(query.lower(), (row_name or "").lower())

    def _price_closeness(self, price: float, min_price, max_price) -> float:
        """Return 0..100 closeness score based on budget window."""
        try:
            p = float(price)
        except Exception:
            return 0.0
        if min_price is None and max_price is None:
            return 0.0
        if min_price is not None and max_price is not None:
            # inside window -> high; otherwise distance decay
            if min_price <= p <= max_price:
                return 100.0
            d = min(abs(p - min_price), abs(p - max_price))
            span = max(max_price - min_price, 1.0)
            return max(0.0, 100.0 - 100.0 * d / (span * 2.0))
        if min_price is not None:
            if p >= min_price:
                return 100.0 - min(100.0, (p - min_price) / max(min_price, 1.0) * 50.0)
            return 0.0
        if max_price is not None:
            if p <= max_price:
                return 100.0 - min(100.0, (max_price - p) / max(max_price, 1.0) * 50.0)
            return 0.0
        return 0.0

    def _load_search_config(self) -> Tuple[dict, int]:
        """Load weights config from config/search.yaml if present."""
        import yaml
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "search.yaml")
        weights = {"style": 0.6, "name": 0.2, "price_closeness": 0.2}
        style_thresh = 70
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                weights.update((cfg.get("weights") or {}))
                style_thresh = int((cfg.get("style_match_threshold") or style_thresh))
            except Exception:
                pass
        return weights, style_thresh

    # ---------- main search ----------
    def search(
            self,
            query: str,
            styles: Optional[List[Any]] = None,   # ['sporty', ...] or [('sporty', 0.82), ...]
            min_price: Optional[float] = None,
            max_price: Optional[float] = None,
            top_n: int = 10,
    ) -> pd.DataFrame:
        df = self.catalog.copy()
        query_hint = (query or "").strip()

        # --- price filtering ---
        if min_price is not None:
            df = df[df["price"] >= float(min_price)]
        if max_price is not None:
            df = df[df["price"] <= float(max_price)]

        # --- normalize styles into (phrase, confidence in 0..1) ---
        extracted: List[Tuple[str, float]] = []
        if styles:
            first = styles[0]
            if isinstance(first, (tuple, list)) and len(first) >= 1:
                for item in styles:
                    if len(item) == 1:
                        extracted.append((str(item[0]).lower(), 1.0))
                    else:
                        phrase = str(item[0]).lower()
                        conf = float(item[1])
                        conf = max(0.0, min(1.0, conf))
                        extracted.append((phrase, conf))
            else:
                extracted = [(str(s).lower(), 1.0) for s in styles]

        # --- style scoring (weighted by confidence) ---
        if extracted:
            def wstyle_score(row_style: str) -> float:
                rs = (row_style or "").lower()
                best = 0.0
                for ph, conf in extracted:
                    base = fuzz.token_set_ratio(ph, rs)  # 0..100
                    score = base * conf                  # weight by confidence
                    if score > best:
                        best = score
                return min(100.0, best)

            df = df.copy()
            df["style_score"] = df["style"].apply(wstyle_score)
        else:
            df = df.copy()
            df["style_score"] = 0.0

        # --- name hint score ---
        df["name_score"] = df["name"].apply(lambda n: self._name_hint_score(n, query_hint))

        # --- ranking weights ---
        weights, style_thresh = self._load_search_config()

        if extracted:
            df = df[df["style_score"] >= style_thresh]

        # --- price closeness ---
        df["price_close"] = df["price"].apply(lambda p: self._price_closeness(p, min_price, max_price))

        # --- final weighted score ---
        df["wscore"] = (
                weights.get("style", 0.6) * df["style_score"].fillna(0) +
                weights.get("name", 0.2) * df["name_score"].fillna(0) +
                weights.get("price_closeness", 0.2) * df["price_close"].fillna(0)
        )

        df = df.sort_values(by=["wscore", "style_score", "name_score"], ascending=[False, False, False])
        return df.head(top_n).reset_index(drop=True)
