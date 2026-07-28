"""
Embedding 工厂模块
支持多种 Embedding 模型提供商
"""
from typing import List, Optional, Dict, Any
from collections import OrderedDict
import numpy as np
from loguru import logger

try:
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logger.warning("dashscope 未安装，通义千问 Embedding 不可用。安装：pip install dashscope")


class EmbeddingError(Exception):
    """Embedding 相关错误的基类"""
    pass


class EmbeddingProviderError(EmbeddingError):
    """Embedding 提供商 API 错误"""
    pass


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Embedding 维度不匹配错误"""
    pass


class EmbeddingFactory:
    """Embedding 模型工厂"""

    def __init__(self, provider: str = "dashscope", model: str = "text-embedding-v1",
                 api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        初始化 Embedding 工厂

        Args:
            provider: 提供商 (dashscope, openai, local)
            model: 模型名称
            api_key: API 密钥
            api_base: API 基础 URL
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_max_size: int = 10000  # LRU 上限，避免内存无限增长

        # 客户端缓存（避免重复创建）
        self._openai_client = None
        self._ollama_session = None

        if provider == "dashscope" and not DASHSCOPE_AVAILABLE:
            logger.error("dashscope 未安装，无法使用通义千问 Embedding")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的 Embedding 向量

        Args:
            text: 输入文本

        Returns:
            np.ndarray: Embedding 向量
        """
        # 检查缓存（命中时移到末尾，保持 LRU 顺序）
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]

        if self.provider == "dashscope":
            embedding = self._get_dashscope_embedding(text)
        elif self.provider == "openai":
            embedding = self._get_openai_embedding(text)
        elif self.provider == "ollama":
            embedding = self._get_ollama_embedding(text)
        else:
            logger.warning(f"未知的 Embedding 提供商：{self.provider}，使用 TF-IDF 降级方案")
            embedding = self._get_tfidf_fallback(text)

        # 缓存结果（LRU 淘汰）
        self._cache[text] = embedding
        if len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)
        return embedding

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量获取 Embedding 向量

        Args:
            texts: 文本列表

        Returns:
            np.ndarray: Embedding 矩阵 (N x D)
        """
        # 部分从缓存获取
        cached = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self._cache:
                cached.append((i, self._cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 获取未缓存的
        if uncached_texts:
            if self.provider == "dashscope":
                new_embeddings = self._get_dashscope_embeddings_batch(uncached_texts)
            elif self.provider == "openai":
                new_embeddings = self._get_openai_embeddings_batch(uncached_texts)
            elif self.provider == "ollama":
                new_embeddings = self._get_ollama_embeddings_batch(uncached_texts)
            else:
                new_embeddings = np.array([self._get_tfidf_fallback(t) for t in uncached_texts])

            # 缓存（LRU 淘汰）
            for idx, embedding in zip(uncached_indices, new_embeddings):
                self._cache[texts[idx]] = embedding
                if len(self._cache) > self._cache_max_size:
                    self._cache.popitem(last=False)

        # 合并结果
        result = np.zeros((len(texts), self._get_embedding_dim()))
        for i, emb in cached:
            result[i] = emb
        for idx, emb in zip(uncached_indices, new_embeddings):
            result[idx] = emb

        return result

    def _get_dashscope_embedding(self, text: str) -> np.ndarray:
        """获取通义千问 Embedding"""
        if not DASHSCOPE_AVAILABLE:
            logger.warning("dashscope 不可用，使用 TF-IDF 降级")
            return self._get_tfidf_fallback(text)

        try:
            import dashscope
            from dashscope import TextEmbedding

            # 设置 API Key
            if self.api_key:
                dashscope.api_key = self.api_key

            response = TextEmbedding.call(
                model=self.model,
                input=text
            )

            if response.status_code == 200:
                embedding = response.output['embeddings'][0]['embedding']
                return np.array(embedding, dtype=np.float32)
            else:
                logger.error(f"DashScope API 错误：{response.code} - {response.message}")
                return self._get_tfidf_fallback(text)

        except Exception as e:
            logger.error(f"获取 DashScope Embedding 失败：{e}")
            return self._get_tfidf_fallback(text)

    def _get_dashscope_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """批量获取通义千问 Embedding（DashScope 限制 batch size <= 25）"""
        if not DASHSCOPE_AVAILABLE:
            return np.array([self._get_tfidf_fallback(t) for t in texts])

        try:
            import dashscope
            from dashscope import TextEmbedding

            if self.api_key:
                dashscope.api_key = self.api_key

            # DashScope 限制 batch size 不能超过 25，需要分批处理
            MAX_BATCH_SIZE = 25
            all_embeddings = []

            for i in range(0, len(texts), MAX_BATCH_SIZE):
                batch_texts = texts[i:i + MAX_BATCH_SIZE]
                response = TextEmbedding.call(
                    model=self.model,
                    input=batch_texts  # 支持列表输入
                )

                if response.status_code == 200:
                    for item in response.output['embeddings']:
                        all_embeddings.append(item['embedding'])
                else:
                    logger.error(f"DashScope API 错误：{response.code} - {response.message}")
                    # 失败时使用单个请求降级
                    for text in batch_texts:
                        all_embeddings.append(self._get_dashscope_embedding(text).tolist())

            return np.array(all_embeddings, dtype=np.float32)

        except Exception as e:
            logger.error(f"批量获取 DashScope Embedding 失败：{e}")
            return np.array([self._get_tfidf_fallback(t) for t in texts])

    def _get_openai_client(self):
        """获取或创建缓存的 OpenAI 客户端"""
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
        return self._openai_client

    def _get_openai_embedding(self, text: str) -> np.ndarray:
        """获取 OpenAI Embedding"""
        try:
            client = self._get_openai_client()

            response = client.embeddings.create(
                model=self.model or "text-embedding-ada-002",
                input=text
            )

            return np.array(response.data[0].embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"获取 OpenAI Embedding 失败：{e}")
            return self._get_tfidf_fallback(text)

    def _get_openai_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """批量获取 OpenAI Embedding"""
        try:
            client = self._get_openai_client()

            # OpenAI API 支持批量输入
            response = client.embeddings.create(
                model=self.model or "text-embedding-ada-002",
                input=texts
            )

            embeddings = []
            for item in sorted(response.data, key=lambda x: x.index):
                embeddings.append(item.embedding)

            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            logger.error(f"批量获取 OpenAI Embedding 失败：{e}")
            return np.array([self._get_tfidf_fallback(t) for t in texts])

    def _get_ollama_session(self):
        """获取或创建缓存的 Ollama HTTP 会话（连接池）"""
        if self._ollama_session is None:
            import requests
            self._ollama_session = requests.Session()
            # 配置连接池
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=10,
                max_retries=3
            )
            self._ollama_session.mount('http://', adapter)
            self._ollama_session.mount('https://', adapter)
        return self._ollama_session

    def _get_ollama_embedding(self, text: str) -> np.ndarray:
        """通过 Ollama REST API 获取 Embedding（使用连接池）"""
        base_url = self.api_base or "http://localhost:11434"
        url = f"{base_url}/api/embeddings"
        try:
            session = self._get_ollama_session()
            response = session.post(
                url,
                json={
                    "model": self.model,
                    "prompt": text,
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding", [])
            if not embedding:
                raise ValueError(f"Ollama 返回空 embedding: {data}")
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"获取 Ollama Embedding 失败：{e}")
            return self._get_tfidf_fallback(text)

    def _get_ollama_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """批量获取 Ollama Embedding（逐条调用，Ollama REST API 不支持批量）"""
        return np.array([self._get_ollama_embedding(t) for t in texts])

    def _get_tfidf_fallback(self, text: str) -> np.ndarray:
        """TF-IDF 降级方案（仅用于测试）"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=384)
        # 需要至少一个文档来 fit
        vectorizer.fit([text])
        tfidf = vectorizer.transform([text]).toarray()[0]
        # 填充到 384 维
        if len(tfidf) < 384:
            padded = np.zeros(384, dtype=np.float32)
            padded[:len(tfidf)] = tfidf
            return padded
        return tfidf[:384].astype(np.float32)

    def _get_embedding_dim(self) -> int:
        """获取 Embedding 维度"""
        if self.provider == "dashscope":
            if "v2" in self.model:
                return 1536
            elif "v3" in self.model:
                return 1024
            else:  # v1
                return 1536
        elif self.provider == "openai":
            if "large" in self.model:
                return 3072
            elif "small" in self.model:
                return 1536
            else:
                return 1536
        elif self.provider == "ollama":
            # Ollama 模型维度各不相同，尝试从配置读取，否则返回常见默认值
            from config import EMBEDDING_PROVIDERS
            dim = EMBEDDING_PROVIDERS.get("ollama", {}).get("dimensions", 768)
            return dim
        return 384

    def clear_cache(self):
        """清除缓存并关闭 HTTP 会话"""
        self._cache.clear()
        # 关闭 HTTP 会话和客户端
        if self._ollama_session is not None:
            try:
                self._ollama_session.close()
            except Exception:
                pass
            self._ollama_session = None
        if self._openai_client is not None:
            try:
                self._openai_client.close()
            except Exception:
                pass
            self._openai_client = None

    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)


# ==================== 向后兼容（已废弃，请使用 ApplicationContext） ====================
# 保留全局单例以兼容旧代码，新功能应使用 ApplicationContext

_embedding_factory: Optional[EmbeddingFactory] = None


def get_embedding_factory(provider: str = "dashscope",
                          model: str = "text-embedding-v1",
                          api_key: Optional[str] = None,
                          api_base: Optional[str] = None) -> EmbeddingFactory:
    """
    获取或创建 Embedding 工厂单例

    .. deprecated:: 1.4
        此函数已废弃，请使用 `ApplicationContext` 管理组件生命周期。
        保留此函数仅为向后兼容，新功能不应使用。

    Args:
        provider: 提供商
        model: 模型名称
        api_key: API 密钥
        api_base: API 基础 URL

    Returns:
        EmbeddingFactory: 工厂实例
    """
    import warnings
    warnings.warn(
        "get_embedding_factory() 已废弃，请使用 ApplicationContext 代替。"
        "此函数将在未来版本中移除。",
        DeprecationWarning,
        stacklevel=2
    )
    global _embedding_factory
    if _embedding_factory is None:
        _embedding_factory = EmbeddingFactory(provider, model, api_key, api_base)
    return _embedding_factory
