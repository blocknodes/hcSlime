"""FastAPI 入口：健康检查 + LLM-first 预测接口。"""

from __future__ import annotations

import json
import logging
import time

from fastapi import FastAPI

from .models import PredictRequest, PredictResponse, Prediction
from .router import predict as run_predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("hcTools.api")

app = FastAPI(title="hcTools LLM-first Planner", version="0.1.0")


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    started = time.perf_counter()
    req_dump = {"query": req.query, "domain": req.domain, "metadata": req.metadata}
    logger.info("PREDICT >> IN  %s", json.dumps(req_dump, ensure_ascii=False))

    prediction: Prediction = await run_predict(req)

    resp = PredictResponse(tool=prediction.tool, params=prediction.params,
                           hit_source=prediction.hit_source)
    logger.info(
        "PREDICT << OUT %s  (%.0fms)",
        json.dumps(
            {"tool": prediction.tool, "params": prediction.params, "error": prediction.error,
             "hit_source": prediction.hit_source},
            ensure_ascii=False,
        ),
        (time.perf_counter() - started) * 1000,
    )
    return resp
