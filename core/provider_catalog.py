"""受支持的模型提供商目录。

单独成模块是为了打破依赖环：`config.py` 需要这些目录来解析环境变量默认值，
而 `core/config_validation.py` 也需要它们来校验"提供商是否需要 API Key"。
`config.py` 会重新导出这两个字典，因此 `from config import LLM_PROVIDERS`
的既有写法保持不变。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# 支持的 LLM 提供商及推荐模型
LLM_PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-4o",
        "needs_api_key": True,
        "needs_base_url": False,
        "default_base_url": "",
    },
    "ollama": {
        "label": "Ollama (本地)",
        "default_model": "gemma4:E4B",
        "needs_api_key": False,
        "needs_base_url": True,
        "default_base_url": "http://localhost:11434",
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "default_model": "claude-sonnet-4-20250514",
        "needs_api_key": True,
        "needs_base_url": False,
        "default_base_url": "",
    },
    "dashscope": {
        "label": "DashScope (通义千问)",
        "default_model": "qwen-max",
        "needs_api_key": True,
        "needs_base_url": True,
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "deepseek": {
        "label": "DeepSeek (深度求索)",
        "default_model": "deepseek-chat",
        "needs_api_key": True,
        "needs_base_url": True,
        "default_base_url": "https://api.deepseek.com/v1",
    },
    "zhipu": {
        "label": "智谱 AI",
        "default_model": "glm-4-flash",
        "needs_api_key": True,
        "needs_base_url": True,
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
    "azure": {
        "label": "Azure OpenAI",
        "default_model": "gpt-4o",
        "needs_api_key": True,
        "needs_base_url": True,
        "default_base_url": "",
    },
}

# 支持的 Embedding 模型提供商
EMBEDDING_PROVIDERS: dict[str, dict[str, Any]] = {
    "dashscope": {
        "label": "DashScope (通义千问)",
        "default_model": "text-embedding-v3",
        "dimensions": 1024,
        "needs_api_key": True,
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "text-embedding-3-large",
        "dimensions": 3072,
        "needs_api_key": True,
    },
    "ollama": {
        "label": "Ollama (本地)",
        "default_model": "bge-m3:latest",
        "dimensions": 1024,
        "needs_api_key": False,
    },
}


def provider_requires_api_key(catalog: Mapping[str, Any], provider: Any) -> bool:
    """目录内已知且声明需要密钥的提供商才返回 True。

    未知提供商刻意返回 False：`core/llm_factory.create_llm` 对未知 provider 是
    容错回退（记一条 warning 后按 OpenAI 兼容处理），`reload_llm` 的原子性测试
    也依赖这一宽容语义。收紧这里会同时破坏两者。
    """
    if not isinstance(provider, str):
        return False
    entry = catalog.get(provider)
    return isinstance(entry, Mapping) and bool(entry.get("needs_api_key"))


__all__ = [
    "EMBEDDING_PROVIDERS",
    "LLM_PROVIDERS",
    "provider_requires_api_key",
]
