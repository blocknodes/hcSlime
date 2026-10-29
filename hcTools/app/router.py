"""按 domain_key 分发到对应域流水线。

域流水线固定：preprocess → select → fill → postprocess。
内核只负责编排，不包含任何单域业务逻辑；前后处理由各域 Domain 可选提供。
"""

from __future__ import annotations

from .domain import load_domains
from .engine import run as run_engine
from .models import Prediction, PredictRequest


async def predict(req: PredictRequest) -> Prediction:
    domain = load_domains().get(req.domain)
    if domain is None:
        return Prediction(domain=req.domain, error=f"未知域：{req.domain}")

    # 前处理：只允许改写 query / 注入提示，不得替模型决定工具或参数
    if domain.preprocess is not None:
        try:
            processed = domain.preprocess(req)
            if isinstance(processed, PredictRequest):
                req = processed
        except Exception as exc:  # noqa: BLE001 前处理失败回退原始请求
            req = req

    result = await run_engine(req, domain)
    return result