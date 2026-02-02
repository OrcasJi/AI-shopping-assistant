# models/integrated_system.py
"""
Integrated Shopping Assistant
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# --- ensure project root on sys.path when running directly ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from utils.paths import DATA_DIR, MODELS_DIR
from utils.product_searcher import ProductSearcher
from models.intent_classifier import IntentClassifier
from models.style_ner import StyleNER

try:
    from recommend.generator import generate_recommendation
except Exception:
    generate_recommendation = None


_PRICE_RE = re.compile(
    r"(?:(?:under|below|less than)\s*[$£€]?(\d+(?:\.\d+)?))|"
    r"(?:(?:over|above|more than)\s*[$£€]?(\d+(?:\.\d+)?))|"
    r"(?:between\s*[$£€]?(\d+(?:\.\d+)?)\s*(?:and|-|to)\s*[$£€]?(\d+(?:\.\d+)?))|"
    r"[$£€]?(\d+(?:\.\d+)?)",
    re.IGNORECASE
)

def parse_price_range(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract price range from text; returns (min_price, max_price)."""
    text = text or ""
    m = _PRICE_RE.search(text)
    if not m:
        return None, None

    under, over, lo, hi, single = m.groups()
    if under:
        try:
            return None, float(under)
        except Exception:
            return None, None
    if over:
        try:
            return float(over), None
        except Exception:
            return None, None
    if lo and hi:
        try:
            a, b = float(lo), float(hi)
            return (min(a, b), max(a, b))
        except Exception:
            return None, None
    if single:
        try:
            v = float(single)
            if re.search(r"(around|about|approx(imately)?|~)", text, re.I):
                return max(v * 0.8, 0.0), v * 1.2
            return None, v
        except Exception:
            return None, None
    return None, None


class ShoppingAssistant:
    """Integrated Assistant: intent detection, style extraction, product search."""

    def __init__(
            self,
            catalog_path: Optional[str] = None,
            intent_dir: Optional[str] = None,
            style_dir: Optional[str] = None,
            num_intents: int = 10,
    ):
        self.catalog_path = catalog_path or str(DATA_DIR / "Shopping_product_catalog_expanded.csv")
        self.intent_dir = intent_dir or str(MODELS_DIR / "intent_classifier")
        self.style_dir = style_dir or str(MODELS_DIR / "style_ner")

        self.searcher = ProductSearcher(self.catalog_path)

        if os.path.isdir(self.intent_dir) and os.path.isfile(os.path.join(self.intent_dir, "pytorch_model.bin")):
            self.intent: IntentClassifier = IntentClassifier.load(self.intent_dir, num_intents=num_intents)
        else:
            self.intent = IntentClassifier(num_intents=num_intents, model_name="bert-base-uncased")
            self.intent.eval()

        if os.path.isdir(self.style_dir) and os.path.isfile(os.path.join(self.style_dir, "pytorch_model.bin")):
            self.style: StyleNER = StyleNER.load(self.style_dir)
        else:
            self.style = StyleNER(model_name="bert-base-uncased")

        self.reverse_label_map: Dict[int, str] = {i: f"intent_{i}" for i in range(num_intents)}

    def set_intent_labels(self, reverse_label_map: Dict[int, str]) -> None:
        if reverse_label_map:
            self.reverse_label_map = dict(reverse_label_map)

    # -------- full pipeline --------
    def process_query(
            self,
            text: str,
            top_n: int = 10,
            style_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "query": str,
                "intent": str|int,
                "styles": [str, ...],                 # cleaned phrases
                "style_candidates": [(str, conf), ...],# with confidences (0..1)
                "min_price": float|None,
                "max_price": float|None,
                "products": [ {...}, ... ]
            }
        """
        # 1) intent
        intent_name: Union[int, str] = self.intent.predict_intent(text, reverse_label_map=self.reverse_label_map)

        # 2) styles WITH confidences
        styles_full: List[Tuple[str, float]] = self.style.extract_styles(
            text, return_confidence=True, threshold=style_threshold
        )
        styles_clean = [ph for ph, _ in styles_full]

        # 3) price range
        min_price, max_price = parse_price_range(text)

        # 4) search (first pass: use styles with confidences)
        df: pd.DataFrame = self.searcher.search(
            query=text,
            styles=styles_full,              # pass (phrase, conf)
            min_price=min_price,
            max_price=max_price,
            top_n=top_n,
        )

        # 5) fallback: if empty and styles existed, retry without style filter
        if df.empty and styles_full:
            df = self.searcher.search(
                query=text,
                styles=None,
                min_price=min_price,
                max_price=max_price,
                top_n=top_n,
            )

        return {
            "query": text,
            "intent": intent_name,
            "styles": styles_clean,
            "style_candidates": styles_full,
            "min_price": min_price,
            "max_price": max_price,
            "products": df.to_dict(orient="records"),
        }

    # -------- rendering --------
    def generate_response(self, result: Dict[str, Any]) -> str:
        intent = result.get("intent")
        styles: List[str] = result.get("styles", [])
        min_price = result.get("min_price")
        max_price = result.get("max_price")
        products: List[Dict[str, Any]] = result.get("products", [])

        header_bits: List[str] = []
        if styles:
            header_bits.append(", ".join(styles) + " style")
        if min_price is not None and max_price is not None:
            header_bits.append(f"budget ${min_price:.0f}~${max_price:.0f}")
        elif min_price is not None:
            header_bits.append(f"minimum budget ${min_price:.0f}+")
        elif max_price is not None:
            header_bits.append(f"under ${max_price:.0f}")
        head = "; ".join(header_bits) if header_bits else "(no explicit budget or style detected)"

        if not products:
            return f"Intent: {intent}. {head}. Sorry, no matching products found. Try widening the price range or changing the keywords."

        lines = [f"Intent: {intent}; {head}. Found {len(products)} candidates, top {min(len(products), 3)}:"]
        for i, row in enumerate(products[:3], start=1):
            name = row.get("name", "N/A")
            cat = row.get("category", "N/A")
            st = row.get("style", "N/A")
            price = row.get("price", "N/A")
            try:
                price_txt = f"${float(price):.2f}"
            except Exception:
                price_txt = str(price)
            lines.append(f"{i}. {name} ({cat} / {st}) - {price_txt}")
        return "\n".join(lines)

    # -------- optional LLM --------
    def generate_llm_recommendation(self, result: dict, provider: str = None, lang: str = "en") -> str:
        if generate_recommendation is None:
            return "(LLM disabled: missing dependency or module)"
        try:
            return generate_recommendation(result, provider=provider, lang=lang)
        except Exception as e:
            return f"(LLM generation failed: {e})"


def _demo():
    assistant = ShoppingAssistant()
    tests = [
        "Looking for a sporty jacket under $100",
        "Show me formal shoes around 150 dollars",
        "I want vintage style dresses",
        "Casual boots between $50 and $100",
    ]
    for q in tests:
        r = assistant.process_query(q, top_n=5, style_threshold=0.7)
        print(">>", q)
        print(assistant.generate_response(r))
        print("-" * 60)


if __name__ == "__main__":
    _demo()
