"""运行时配置：LLM 网关走环境变量，服务参数走常量默认。载入路径固定到本文件所在包的上级目录，避免受 cwd 影响。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HC_",
        env_file=str(_BASE / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 网关：与 hcAgent 对齐（juagent 真实网关）；可用 HC_API_BASE / HC_API_KEY 覆盖
    api_base: str = "http://10.19.96.219:4003/v1"
    api_key: str = ""
    model: str = "baseline"    # 与 hcAgent 默认一致

    # select 阶段动态 few-shot 数量（0 表示禁用动态检索）
    select_shots: int = 6

    @property
    def chat_url(self) -> str:
        return f"{self.api_base.rstrip('/')}/chat/completions"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
