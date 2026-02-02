# recommend/generator.py
"""LLM recommendation generator.

Providers:
- OpenAI: uses 'openai' SDK (Responses API)
- DeepSeek: simple HTTP call via httpx

Environment variables:
- OPENAI_API_KEY
- DEEPSEEK_API_KEY
- LLM_PROVIDER (openai|deepseek), default: openai
"""

import os
from typing import List, Dict, Any, Optional
import json
import httpx

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


SYSTEM_PROMPT = (
    "You are an e-commerce shopping assistant. Be concise, avoid hallucinations, "
    "and never invent prices or products. Use only the provided product list. "
    "Offer helpful next-step filters (style/price/category) when relevant."
)

def _format_products(products: List[Dict[str, Any]], top_n: int = 5) -> str:
    lines = []
    for i, p in enumerate(products[:top_n], start=1):
        name = p.get("name", "N/A")
        cat = p.get("category", "N/A")
        style = p.get("style", "N/A")
        price = p.get("price", "N/A")
        lines.append(f"{i}. {name} | {cat} | {style} | ${price}")
    return "\n".join(lines)


def build_prompt(
    user_query: str,
    intent: Any,
    styles: List[str],
    min_price: Optional[float],
    max_price: Optional[float],
    products: List[Dict[str, Any]],
    lang: str = "zh",
) -> str:
    products_block = _format_products(products, top_n=5)
    price_str = (
        f"预算区间: ${min_price:.0f}~${max_price:.0f}" if min_price is not None and max_price is not None else
        (f"最低预算: ${min_price:.0f}+" if min_price is not None else
         (f"不超过: ${max_price:.0f}" if max_price is not None else ""))
    )
    style_str = "、".join(styles) if styles else "(未检测到风格)"
    if lang == "zh":
        return (f"用户查询: {user_query}\n"f"意图: {intent}\n"f"风格: {style_str}\n"f"{price_str}\n""请基于以下商品给出简洁友好的推荐语(150字内)，不要捏造信息，""可以给出两条筛选建议。商品列表如下:\n"f"{products_block}"        )
    else:
        return (f"User query: {user_query}\n"f"Intent: {intent}\n"f"Styles: {style_str}\n"f"{price_str}\n""Write a short, friendly recommendation under 120 words using ONLY the products below."" Suggest up to two next-step filters.\n"f"{products_block}"        )


def generate_recommendation(
    result: Dict[str, Any],
    provider: Optional[str] = None,
    lang: str = "zh",
    model_openai: str = "gpt-4o-mini",
    model_deepseek: str = "deepseek-chat",
    timeout_s: int = 30,
) -> str:
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    user_query = result.get("query", "")
    intent = result.get("intent")
    styles = result.get("styles", [])
    min_price = result.get("min_price")
    max_price = result.get("max_price")
    products = result.get("products", [])

    prompt = build_prompt(user_query, intent, styles, min_price, max_price, products, lang=lang)

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return "(LLM未启用：缺少 DEEPSEEK_API_KEY)"
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_deepseek,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 256
        }
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()

    # default: openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(LLM未启用：缺少 OPENAI_API_KEY)"
    if OpenAI is None:
        return "(openai SDK 未安装)"

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_openai,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()
