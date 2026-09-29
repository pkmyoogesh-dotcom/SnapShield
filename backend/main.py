"""SnapShield local API. Submitted URLs are analysed as strings and never fetched."""
from __future__ import annotations

import logging
from functools import lru_cache
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.schemas.models import PredictRequest, PredictResponse
from backend.services.explainer import explain
from backend.services.feature_extractor import InvalidURLError, extract_features, feature_vector
from backend.services.predictor import ModelBundle, classify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("snapshield")

app = FastAPI(title="SnapShield", version="1.0.0",
              description="On-device explainable phishing URL detection.")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://.*|https?://(127\.0\.0\.1|localhost)(:\d+)?)$",
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.middleware("http")
async def limit_body(request: Request, call_next):
    if int(request.headers.get("content-length") or 0) > config.MAX_BODY_BYTES:
        return JSONResponse({"detail": "Request body too large."}, status_code=413)
    return await call_next(request)


@lru_cache(maxsize=1)
def _load() -> ModelBundle:
    return ModelBundle.load()


def get_bundle() -> ModelBundle:
    try:
        return _load()
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from None


@app.get("/api/v1/health")
def health():
    try:
        b, loaded = _load(), True
    except Exception:
        b, loaded = None, False
    return {"status": "ok", "model_loaded": loaded,
            "model_type": b.metadata.get("model_type") if b else None,
            "inference_backend": b.backend_name if b else None,
            "ai_engine": "LOCAL", "cloud_ai_required": False,
            "thresholds": {"suspicious": config.SUSPICIOUS_THRESHOLD, "phishing": config.PHISHING_THRESHOLD}}


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest, bundle: ModelBundle = Depends(get_bundle)):
    try:
        feats = extract_features(req.url)
    except InvalidURLError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    vec = feature_vector(feats)
    score = float(bundle.predict_proba([vec])[0])
    label = classify(score)
    reasons, contribs = explain(vec, bundle)
    log.info("scan host=%s class=%s", urlsplit(req.url).hostname, label)  # host only
    return PredictResponse(url=req.url.strip(), classification=label, risk_score=round(score, 4),
                           risk_percentage=round(score * 100), reasons=reasons,
                           contributions=contribs, features=feats, inference=bundle.backend_name)


if config.DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=config.DASHBOARD_DIR, html=True), name="dashboard")
