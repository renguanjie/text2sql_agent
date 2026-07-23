"""
LLM 工厂模块
创建和配置各种 LLM 实例
"""
from typing import Optional, Dict, Any
from loguru import logger
import os


# OpenAI 兼容 API 的默认 Base URL（DashScope / DeepSeek / 智谱均使用 OpenAI 兼容接口）
_OPENAI_COMPATIBLE_BASE_URLS = {
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
}


def create_llm(provider: str = "openai", model: str = "gpt-4",
               api_key: Optional[str] = None, base_url: Optional[str] = None,
               temperature: float = 0.1, **kwargs) -> Any:
    """
    创建 LLM 实例

    Args:
        provider: LLM 提供商 (openai, azure, ollama, anthropic, dashscope, deepseek, zhipu 等)
        model: 模型名称
        api_key: API 密钥
        base_url: 基础 URL（可选，DashScope/DeepSeek/智谱有默认值）
        temperature: 温度参数

    Returns:
        LLM 实例
    """
    if provider == "openai":
        return _create_openai_llm(model, api_key, base_url, temperature, **kwargs)
    elif provider == "azure":
        return _create_azure_llm(model, api_key, base_url, temperature, **kwargs)
    elif provider in ("local", "ollama"):
        return _create_local_llm(model, base_url, temperature, **kwargs)
    elif provider == "anthropic":
        return _create_anthropic_llm(model, api_key, temperature, **kwargs)
    elif provider in ("dashscope", "deepseek", "zhipu"):
        # 这些厂商均使用 OpenAI 兼容 API，复用 ChatOpenAI
        resolved_base_url = base_url or _OPENAI_COMPATIBLE_BASE_URLS.get(provider, "")
        return _create_openai_compatible_llm(provider, model, api_key, resolved_base_url, temperature, **kwargs)
    else:
        logger.warning(f"未知的 LLM 提供商：{provider}，使用 OpenAI 默认配置")
        return _create_openai_llm(model, api_key, base_url, temperature, **kwargs)


def _create_openai_compatible_llm(provider: str, model: str,
                                  api_key: Optional[str], base_url: str,
                                  temperature: float, **kwargs) -> Any:
    """创建 OpenAI 兼容 API 的 LLM（DashScope / DeepSeek / 智谱等）"""
    try:
        from langchain_openai import ChatOpenAI

        if not api_key:
            raise ValueError(f"{provider} API Key 未配置")

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            **kwargs,
        )
        logger.info(f"创建 {provider} LLM: {model} @ {base_url}")
        return llm

    except ImportError:
        logger.error("需要安装 langchain-openai: pip install langchain-openai")
        raise


def _create_openai_llm(model: str, api_key: Optional[str],
                      base_url: Optional[str], temperature: float,
                      **kwargs) -> Any:
    """创建 OpenAI LLM"""
    try:
        from langchain_openai import ChatOpenAI

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API Key 未配置")
            raise ValueError("OpenAI API Key 未配置")

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
        logger.info(f"创建 OpenAI LLM: {model}")
        return llm

    except ImportError:
        logger.error("需要安装 langchain-openai: pip install langchain-openai")
        raise


def _create_azure_llm(model: str, api_key: Optional[str],
                     base_url: Optional[str], temperature: float,
                     **kwargs) -> Any:
    """创建 Azure OpenAI LLM"""
    try:
        from langchain_openai import AzureChatOpenAI

        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            logger.error("Azure OpenAI API Key 未配置")
            raise ValueError("Azure OpenAI API Key 未配置")

        llm = AzureChatOpenAI(
            azure_deployment=model,
            api_key=api_key,
            azure_endpoint=base_url,
            temperature=temperature,
            **kwargs
        )
        logger.info(f"创建 Azure OpenAI LLM: {model}")
        return llm

    except ImportError:
        logger.error("需要安装 langchain-openai: pip install langchain-openai")
        raise


def _create_local_llm(model: str, base_url: Optional[str],
                     temperature: float, **kwargs) -> Any:
    """创建本地 LLM (Ollama 等)"""
    try:
        from langchain_community.chat_models import ChatOllama

        host = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")

        llm = ChatOllama(
            model=model,
            base_url=host,
            temperature=temperature,
            **kwargs
        )
        logger.info(f"创建本地 LLM: {model} @ {host}")
        return llm

    except ImportError:
        logger.error("需要安装 langchain-community: pip install langchain-community")
        raise


def _create_anthropic_llm(model: str, api_key: Optional[str],
                         temperature: float, **kwargs) -> Any:
    """创建 Anthropic Claude LLM"""
    try:
        from langchain_anthropic import ChatAnthropic

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("Anthropic API Key 未配置")
            raise ValueError("Anthropic API Key 未配置")

        llm = ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )
        logger.info(f"创建 Anthropic LLM: {model}")
        return llm

    except ImportError:
        logger.error("需要安装 langchain-anthropic: pip install langchain-anthropic")
        raise


def get_llm_from_config(config: Optional[Dict] = None) -> Any:
    """
    从配置创建 LLM

    Args:
        config: 配置字典

    Returns:
        LLM 实例
    """
    from config import (
        LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
    )

    config = config or {}

    return create_llm(
        provider=config.get('provider', LLM_PROVIDER),
        model=config.get('model', LLM_MODEL),
        api_key=config.get('api_key', LLM_API_KEY),
        base_url=config.get('base_url', LLM_BASE_URL),
        temperature=config.get('temperature', 0.1)
    )
