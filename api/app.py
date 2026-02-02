# api/app.py
"""
FastAPI server for AI Shopping Assistant
- CORS enabled for easy frontend integration
- Strong Pydantic schemas for request/response
- Helpful endpoints: /query, /batch_query, /recommend, /reload_catalog, /set_intent_labels, /health
"""

import os
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat

from models.integrated_system import ShoppingAssistant


from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# -------------------------
# App & Middleware
# -------------------------
app = FastAPI(title="AI Shopping Assistant (Integrated)", version="1.1.0")

# CORS (adjust allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # e.g. ["https://your-frontend.example"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("assistant-api")

# Singleton assistant
assistant = ShoppingAssistant()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "web"
INDEX_FILE = FRONTEND_DIR / "index.html"

# 1) 挂载静态目录为 /static（以后有 css/js/img 可放在 web/static）
static_dir = FRONTEND_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 2) 根路径返回 index.html
@app.get("/", include_in_schema=False)
def _index():
    if INDEX_FILE.is_file():
        return FileResponse(str(INDEX_FILE))
    # 如果没放 index.html，也给出友好提示
    return {"msg": "Place your frontend at web/index.html. API docs: /docs"}

# 3) 可选：避免浏览器 404 favicon.ico
@app.get("/favicon.ico", include_in_schema=False)
def _favicon():
    ico = FRONTEND_DIR / "favicon.ico"
    if ico.is_file():
        return FileResponse(str(ico))
    # 返回一个 204，避免红色 404 噪音
    from fastapi import Response
    return Response(status_code=204)







# -------------------------
# Schemas
# -------------------------
class Product(BaseModel):
    name: str
    category: str
    style: str
    price: float

class PipelineResult(BaseModel):
    query: str
    intent: str | int
    styles: List[str]
    style_candidates: Optional[List[tuple[str, float]]] = None
    min_price: Optional[float]
    max_price: Optional[float]
    products: List[Product]

class Query(BaseModel):
    text: str = Field(..., description="User query")
    top_n: conint(ge=1, le=50) = 10
    style_threshold: confloat(ge=0.0, le=1.0) = 0.0
    use_llm: bool = False

class BatchQuery(BaseModel):
    queries: List[Query]

class LabelsPayload(BaseModel):
    mapping: Dict[int, str]  # e.g. {0:"search", 1:"price_query", ...}

class RecommendPayload(BaseModel):
    text: Optional[str] = None        # If omitted, you must provide 'result'
    result: Optional[PipelineResult] = None
    use_llm: bool = True


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/query")
def query(q: Query) -> Dict[str, Any]:
    try:
        result = assistant.process_query(q.text, top_n=q.top_n, style_threshold=q.style_threshold)
        message = assistant.generate_response(result)
        if q.use_llm or os.getenv("USE_LLM", "false").lower() == "true":
            llm_text = assistant.generate_llm_recommendation(result, lang="en")
            return {"result": result, "rule_based": message, "llm": llm_text}
        return {"result": result, "message": message}
    except Exception as e:
        logger.exception("query failed")
        raise HTTPException(status_code=500, detail=f"query failed: {e}")

@app.post("/batch_query")
def batch_query(payload: BatchQuery) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for q in payload.queries:
        try:
            r = assistant.process_query(q.text, top_n=q.top_n, style_threshold=q.style_threshold)
            m = assistant.generate_response(r)
            item: Dict[str, Any] = {"result": r, "message": m}
            if q.use_llm or os.getenv("USE_LLM", "false").lower() == "true":
                item["llm"] = assistant.generate_llm_recommendation(r, lang="en")
                item["rule_based"] = item.pop("message")
            items.append(item)
        except Exception as e:
            logger.exception("batch item failed")
            items.append({"error": str(e), "text": q.text})
    return {"items": items}

@app.post("/set_intent_labels")
def set_intent_labels(p: LabelsPayload) -> Dict[str, Any]:
    try:
        assistant.set_intent_labels(p.mapping)
        return {"ok": True, "size": len(p.mapping)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad labels: {e}")

@app.post("/reload_catalog")
def reload_catalog(path: Optional[str] = None) -> Dict[str, Any]:
    """Hot-reload product catalog CSV without restarting the server."""
    try:
        new_path = path or assistant.catalog_path
        assistant.searcher = ShoppingAssistant(catalog_path=new_path).searcher  # reuse loader logic
        return {"ok": True, "rows": len(assistant.searcher.catalog), "path": new_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"reload failed: {e}")

@app.post("/recommend")
def recommend(p: RecommendPayload) -> Dict[str, Any]:
    try:
        if p.result is None:
            if not p.text:
                raise HTTPException(status_code=400, detail="either 'text' or 'result' must be provided")
            r = assistant.process_query(p.text)
        else:
            # Pydantic model -> dict
            r = p.result.model_dump() if hasattr(p.result, "model_dump") else dict(p.result)  # type: ignore
        if p.use_llm or os.getenv("USE_LLM", "false").lower() == "true":
            return {"llm": assistant.generate_llm_recommendation(r, lang="en")}
        else:
            return {"message": assistant.generate_response(r)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("recommend failed")
        raise HTTPException(status_code=500, detail=f"recommend failed: {e}")
