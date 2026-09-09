"""
文本检索模块
实现 BM25 + FAISS 向量检索的两路混合检索算法
支持真实 Embedding 模型和动态 Few-Shot 检索
"""
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS 未安装，向量检索功能将不可用。安装：pip install faiss-cpu")

# 导入 Embedding 工厂
from ..embedding_factory import EmbeddingFactory
from ..history.identity import GovernedHistoryIdentity


def _default_embedding_factory() -> EmbeddingFactory:
    """按当前配置创建独立实例，避免核心代码依赖旧的全局单例。"""
    from config import (
        EMBEDDING_API_BASE,
        EMBEDDING_API_KEY,
        EMBEDDING_MODEL,
        EMBEDDING_PROVIDER,
    )

    return EmbeddingFactory(
        provider=EMBEDDING_PROVIDER,
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        api_base=EMBEDDING_API_BASE,
    )


class TextPreprocessor:
    """文本预处理器 - 支持中英文混合分词"""

    _CHINESE_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

    def __init__(self):
        """初始化预处理器"""
        # 中英文标点符号
        self.punctuation = r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~，。！？；：""''、·…—（）《》「」『』]'

    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词 - 支持中英文混合

        Args:
            text: 输入文本

        Returns:
            List[str]: 分词结果
        """
        text = text.lower()
        text = re.sub(self.punctuation, ' ', text)
        words = text.split()
        result = []

        for w in words:
            if not w:
                continue
            if len(w) > 1 or w.isdigit():
                # 如果包含中文字符，进行字符级分词
                if self._CHINESE_RE.search(w):
                    result.extend(list(w))
                else:
                    result.append(w)

        return result

    def preprocess(self, text: str) -> str:
        """
        预处理文本，返回标准化文本

        Args:
            text: 输入文本

        Returns:
            str: 预处理后的文本
        """
        tokens = self.tokenize(text)
        return ' '.join(tokens)


class DenseVectorEmbedder:
    """稠密向量嵌入器 - 使用真实 Embedding 模型 (通义千问) 和 FAISS"""

    def __init__(self, embedding_factory: Optional[EmbeddingFactory] = None,
                 embedding_dim: int = 1536, index_type: Optional[str] = "flat"):
        """
        初始化向量嵌入器

        Args:
            embedding_factory: Embedding 工厂实例
            embedding_dim: 向量维度 (通义千问 v1=1536, v2=1536, v3=1024)
            index_type: FAISS 索引类型 ("flat", "ivf", "hnsw")
        """
        self.embedding_factory = embedding_factory or _default_embedding_factory()
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.embedding_matrix = None
        self.documents: List[str] = []

    def build_index(self, documents: List[str]):
        """
        构建向量索引 - 使用真实 Embedding 模型

        Args:
            documents: 文档列表
        """
        self.documents = list(documents)
        self.index = None
        self.embedding_matrix = None

        if not documents:
            return

        if not self.index_type:
            logger.info("FAISS 向量检索已禁用")
            return

        if not FAISS_AVAILABLE:
            logger.warning("FAISS 不可用，跳过向量索引构建")
            return

        logger.info(f"构建 FAISS 向量索引：{len(documents)} 个文档，使用真实 Embedding 模型")

        # 使用真实 Embedding 模型计算向量
        logger.info(f"调用 {self.embedding_factory.provider}/{self.embedding_factory.model} 计算 Embedding...")
        self.embedding_matrix = self._compute_embeddings(documents)
        self.embedding_dim = self.embedding_matrix.shape[1]

        # 构建 FAISS 索引
        emb_dim = self.embedding_matrix.shape[1]
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(emb_dim)  # 内积相似度
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(emb_dim, 32, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "ivf":
            cluster_count = max(1, min(100, int(np.sqrt(len(documents)))))
            quantizer = faiss.IndexFlatIP(emb_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, emb_dim, cluster_count, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(self.embedding_matrix)
        else:
            logger.warning(f"未知 FAISS 索引类型 {self.index_type!r}，回退到 flat")
            self.index = faiss.IndexFlatIP(emb_dim)

        # 添加向量到索引
        self.index.add(self.embedding_matrix)

        logger.info(f"FAISS 向量索引构建完成，维度={emb_dim}, 文档数={len(documents)}")

    def _compute_embeddings(self, documents: List[str]) -> np.ndarray:
        """
        使用真实 Embedding 模型计算文档向量

        Args:
            documents: 文档列表

        Returns:
            np.ndarray: 嵌入矩阵
        """
        # 批量获取 Embedding
        embeddings = self.embedding_factory.get_embeddings_batch(documents)

        # L2 归一化 (FAISS Inner Product 需要归一化向量)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def search(self, query: str, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        向量相似度搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            Tuple: (索引数组，分数数组)
        """
        if top_k <= 0 or not FAISS_AVAILABLE or self.index is None:
            return np.array([]), np.array([])

        # 计算查询向量
        query_embedding = self._compute_single_embedding(query)

        # 搜索
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

        return indices[0], scores[0]

    def _compute_single_embedding(self, text: str) -> np.ndarray:
        """计算单个文本的 Embedding 向量"""
        embedding = self.embedding_factory.get_embedding(text)
        target_dim = self.index.d if self.index is not None and hasattr(self.index, "d") else self.embedding_dim
        embedding = self.embedding_factory._coerce_embedding_dim(embedding, target_dim)
        # L2 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)


class HybridRetriever:
    """BM25 + FAISS 两路混合检索器"""

    def __init__(self, bm25_k1: float = 1.5, bm25_b: float = 0.75,
                 bm25_weight: float = 0.6,
                 dense_weight: float = 0.4,
                 embedding_factory: Optional[EmbeddingFactory] = None,
                 embedding_dim: int = 1536,
                 index_type: Optional[str] = "flat"):
        """
        初始化混合检索器

        Args:
            bm25_k1: BM25 k1 参数
            bm25_b: BM25 b 参数
            bm25_weight: BM25 权重（0-1）
            dense_weight: 稠密向量权重（0-1）
            embedding_factory: Embedding 工厂实例
            embedding_dim: 向量嵌入维度
        """
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        # 归一化权重（确保总和为 1）
        total_weight = bm25_weight + dense_weight
        if total_weight > 0:
            self.bm25_weight /= total_weight
            self.dense_weight /= total_weight

        self.preprocessor = TextPreprocessor()
        self.bm25_model = None
        self.documents: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.doc_metadata: List[Dict] = []

        # FAISS 向量检索器 - 使用真实 Embedding 模型
        self.dense_embedder = DenseVectorEmbedder(
            embedding_factory=embedding_factory,
            embedding_dim=embedding_dim,
            index_type=index_type,
        )

    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        索引文档

        Args:
            documents: 文档列表
            metadata: 文档元数据列表
        """
        logger.info(f"索引 {len(documents)} 个文档")

        if metadata is not None and len(metadata) != len(documents):
            raise ValueError("metadata 数量必须与 documents 数量一致")

        self.documents = list(documents)
        self.doc_metadata = [dict(item) for item in metadata] if metadata is not None else [{} for _ in documents]

        if not self.documents:
            self.tokenized_docs = []
            self.bm25_model = None
            self.dense_embedder.build_index([])
            return

        # 预处理和分词
        self.tokenized_docs = [self.preprocessor.tokenize(doc) for doc in documents]

        # 构建 BM25 模型
        self.bm25_model = BM25Okapi(self.tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)
        logger.info("BM25 模型构建完成")

        # 构建 FAISS 向量索引
        self.dense_embedder.build_index(documents)
        logger.info("两路索引构建完成 (BM25 + FAISS)")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        搜索文档 - 两路混合检索 (BM25 + FAISS 向量)

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[Tuple[Dict, float]]: (文档元数据，相似度分数) 列表
        """
        if top_k <= 0 or not query or not query.strip():
            return []
        if not self.documents or not self.bm25_model:
            logger.warning("检索器未初始化，返回空结果")
            return []

        # 预处理查询
        tokenized_query = self.preprocessor.tokenize(query)

        # 1. BM25 分数
        bm25_scores = self.bm25_model.get_scores(tokenized_query)

        # 2. FAISS 向量相似度分数（限制搜索上限以提升性能）
        if FAISS_AVAILABLE and self.dense_embedder.index is not None:
            faiss_k = min(len(self.documents), 100)  # FAISS 搜索上限 100
            dense_indices, dense_scores = self.dense_embedder.search(query, top_k=faiss_k)
            # 将向量分数映射到文档索引（稀疏映射）
            dense_vec = np.zeros(len(self.documents), dtype=np.float32)
            for idx, score in zip(dense_indices, dense_scores):
                if 0 <= idx < len(self.documents) and score > 0:
                    dense_vec[idx] = score
            dense_scores = dense_vec
        else:
            dense_scores = np.zeros(len(self.documents))

        # 归一化分数到 [0, 1] 范围
        bm25_norm = self._normalize_scores(bm25_scores)
        dense_norm = self._normalize_scores(dense_scores)

        # 两路分数融合
        final_scores = (self.bm25_weight * bm25_norm +
                       self.dense_weight * dense_norm)

        # 获取 Top-K
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if final_scores[idx] > 0:
                metadata = self.doc_metadata[idx].copy()
                metadata['content'] = self.documents[idx]
                metadata['index'] = int(idx)
                # 添加各路分数详情
                metadata['scores'] = {
                    'bm25': float(bm25_norm[idx]),
                    'dense': float(dense_norm[idx]),
                    'final': float(final_scores[idx])
                }
                results.append((metadata, float(final_scores[idx])))

        logger.info(f"检索到 {len(results)} 个相关文档 (BM25+FAISS)")
        return results

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        归一化分数到 [0, 1] 范围

        Args:
            scores: 原始分数

        Returns:
            np.ndarray: 归一化后的分数
        """
        if len(scores) == 0:
            return np.asarray(scores)

        min_score = np.min(scores)
        max_score = np.max(scores)

        if max_score == min_score:
            return np.ones_like(scores) if max_score > 0 else np.zeros_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def add_document(self, document: str, metadata: Optional[Dict] = None):
        """
        添加单个文档

        Args:
            document: 文档内容
            metadata: 文档元数据
        """
        documents = [*self.documents, document]
        metadata_items = [*self.doc_metadata, metadata or {}]
        self.index_documents(documents, metadata_items)

    def remove_document(self, index: int):
        """
        移除文档

        Args:
            index: 文档索引
        """
        if 0 <= index < len(self.documents):
            documents = [doc for i, doc in enumerate(self.documents) if i != index]
            metadata_items = [item for i, item in enumerate(self.doc_metadata) if i != index]
            self.index_documents(documents, metadata_items)

    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)

    def save_index(self, directory: str) -> bool:
        """
        持久化索引到磁盘（BM25 模型 + 文档 + 元数据 + FAISS 索引）

        Args:
            directory: 保存目录路径

        Returns:
            bool: 是否成功
        """
        import json
        import os

        try:
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "documents.json"), "w", encoding="utf-8") as f:
                json.dump({"documents": self.documents, "metadata": self.doc_metadata}, f, ensure_ascii=False)
            if self.dense_embedder.index is not None:
                import faiss
                faiss.write_index(self.dense_embedder.index, os.path.join(directory, "faiss.index"))
                if self.dense_embedder.embedding_matrix is not None:
                    np.save(os.path.join(directory, "embedding_matrix.npy"), self.dense_embedder.embedding_matrix)
            logger.info(f"索引已持久化到 {directory}")
            return True
        except Exception as e:
            logger.error(f"索引持久化失败：{e}")
            return False

    def load_index(self, directory: str) -> bool:
        """
        从磁盘加载持久化的索引

        Args:
            directory: 索引目录路径

        Returns:
            bool: 是否成功加载
        """
        import json
        import os

        try:
            docs_path = os.path.join(directory, "documents.json")
            faiss_path = os.path.join(directory, "faiss.index")
            if not os.path.exists(docs_path):
                return False
            with open(docs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            documents = data.get("documents")
            metadata = data.get("metadata")
            if not isinstance(documents, list) or not isinstance(metadata, list):
                raise ValueError("索引缓存格式无效")
            if len(documents) != len(metadata):
                raise ValueError("索引缓存中的文档与元数据数量不一致")
            self.documents = documents
            self.doc_metadata = metadata
            self.tokenized_docs = [self.preprocessor.tokenize(doc) for doc in self.documents]
            self.bm25_model = (
                BM25Okapi(self.tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)
                if self.tokenized_docs else None
            )
            self.dense_embedder.index = None
            self.dense_embedder.embedding_matrix = None
            self.dense_embedder.documents = self.documents
            if os.path.exists(faiss_path) and FAISS_AVAILABLE:
                import faiss
                self.dense_embedder.index = faiss.read_index(faiss_path)
                emb_path = os.path.join(directory, "embedding_matrix.npy")
                if os.path.exists(emb_path):
                    self.dense_embedder.embedding_matrix = np.load(emb_path)
                    self.dense_embedder.embedding_dim = self.dense_embedder.embedding_matrix.shape[1]
                if self.dense_embedder.index.ntotal != len(self.documents):
                    raise ValueError("FAISS 索引与文档数量不一致")
            logger.info(f"索引已从 {directory} 加载（{len(self.documents)} 个文档）")
            return True
        except Exception as e:
            logger.error(f"索引加载失败：{e}")
            return False


class KnowledgeRetriever:
    """知识库检索器 - 封装 HybridRetriever 用于知识库检索
    支持真实 Embedding 模型和动态 Few-Shot 示例检索
    """

    def __init__(self, config: Optional[Dict] = None,
                 embedding_factory: Optional[EmbeddingFactory] = None):
        """
        初始化知识库检索器

        Args:
            config: 配置字典
            embedding_factory: Embedding 工厂实例
        """
        config = config or {}
        self.embedding_factory = embedding_factory or _default_embedding_factory()
        self.retriever = HybridRetriever(
            bm25_k1=config.get('bm25_k1', 1.5),
            bm25_b=config.get('bm25_b', 0.75),
            bm25_weight=config.get('bm25_weight', 0.6),
            dense_weight=config.get('dense_weight', 0.4),
            embedding_factory=self.embedding_factory,
            embedding_dim=config.get('embedding_dim', 1536),
            index_type=config.get('index_type', 'flat'),
        )
        self.node_index: Dict[int, Dict] = {}  # 节点索引

        # Few-Shot 示例检索配置
        self.few_shot_top_k = config.get('few_shot_top_k', 3)
        self.few_shot_min_similarity = config.get('few_shot_min_similarity', 0.5)
        self.history_index: Optional[faiss.IndexFlatIP] = None
        self.history_embeddings: List[np.ndarray] = []
        self.history_records: List[Dict] = []  # 存储成功历史记录

    def build_index(self, knowledge_nodes: List[Dict]):
        """
        从知识节点构建检索索引

        Args:
            knowledge_nodes: 知识节点列表
        """
        documents = []
        metadata = []

        for node in knowledge_nodes:
            # 构建检索文本：组合名称、类型、描述等
            doc_parts = [
                node.get('name', ''),
                node.get('type', ''),
                node.get('description', ''),
                node.get('table_name', ''),
                node.get('column_name', '')
            ]
            doc_text = ' '.join(filter(None, doc_parts))

            if doc_text.strip():
                documents.append(doc_text)
                meta = {
                    'node_id': node.get('id'),
                    'node_type': node.get('node_type', 'unknown'),
                    'name': node.get('name', ''),
                    'description': node.get('description', '')
                }
                meta.update({k: v for k, v in node.items()
                            if k not in ['name', 'description', 'node_type', 'id']})
                metadata.append(meta)

        self.retriever.index_documents(documents, metadata)
        logger.info(f"知识库索引构建完成，共 {len(documents)} 个节点")

    def build_few_shot_index(
        self,
        history_records: List[Dict],
        *,
        governance_identity: GovernedHistoryIdentity | None = None,
    ):
        """
        从历史成功记录构建 Few-Shot 示例索引

        Args:
            history_records: 历史记录列表，每条记录包含 user_query 和 generated_sql
        """
        self.history_index = None
        self.history_embeddings = []
        self.history_records = []

        if not FAISS_AVAILABLE:
            logger.warning("FAISS 不可用，无法构建 Few-Shot 索引")
            return

        # 筛选成功的记录
        successful_records = [
            r for r in history_records
            if r.get('validation_status') == 'success'
            and r.get('execution_status') == 'success'
            and (
                governance_identity is None
                or governance_identity.matches(r)
            )
        ]

        if len(successful_records) < 1:
            logger.warning("没有成功的历史记录，无法构建 Few-Shot 索引")
            return

        logger.info(f"构建 Few-Shot 索引：{len(successful_records)} 条成功历史记录")

        self.history_records = successful_records
        queries = [r.get('user_query', '') for r in successful_records]

        # 计算查询的 Embedding
        import faiss
        embeddings = self.embedding_factory.get_embeddings_batch(queries)

        # L2 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        self.history_embeddings = embeddings

        # 构建 FAISS 索引
        emb_dim = embeddings.shape[1]
        self.history_index = faiss.IndexFlatIP(emb_dim)
        self.history_index.add(embeddings)

        logger.info(f"Few-Shot 索引构建完成，维度={emb_dim}")

    def retrieve_few_shot_examples(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        检索最相似的 Few-Shot 示例

        Args:
            query: 当前查询
            top_k: 返回数量，默认使用配置的 few_shot_top_k

        Returns:
            List[Dict]: 历史 Q&A 示例列表
        """
        if not FAISS_AVAILABLE or self.history_index is None:
            logger.warning("Few-Shot 索引未构建，返回空示例")
            return []

        k = self.few_shot_top_k if top_k is None else top_k
        if k <= 0 or not query or not query.strip():
            return []
        k = min(k, len(self.history_records))

        # 计算查询向量
        query_embedding = self.embedding_factory._coerce_embedding_dim(
            self.embedding_factory.get_embedding(query), self.history_index.d
        )
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # 搜索
        scores, indices = self.history_index.search(
            query_embedding.reshape(1, -1),
            k
        )

        examples = []
        for idx, score in zip(indices[0], scores[0]):
            if (
                0 <= idx < len(self.history_records)
                and score >= self.few_shot_min_similarity
            ):
                record = self.history_records[idx].copy()
                record['similarity_score'] = float(score)
                examples.append(record)

        logger.info(f"检索到 {len(examples)} 个 Few-Shot 示例")
        return examples

    def add_history_record(
        self,
        record: Dict,
        *,
        governance_identity: GovernedHistoryIdentity | None = None,
    ):
        """
        添加单条历史记录到 Few-Shot 索引

        Args:
            record: 历史记录，包含 user_query 和 generated_sql
        """
        if not FAISS_AVAILABLE:
            return

        # 只添加成功的记录
        if (
            record.get('validation_status') != 'success'
            or record.get('execution_status') != 'success'
            or (
                governance_identity is not None
                and not governance_identity.matches(record)
            )
        ):
            return

        record_id = record.get('id')
        if record_id is not None and any(
            existing.get('id') == record_id for existing in self.history_records
        ):
            return

        # 计算 Embedding
        query = record.get('user_query', '')
        if not query:
            return
        embedding = self.embedding_factory.get_embedding(query)
        if self.history_index is not None:
            embedding = self.embedding_factory._coerce_embedding_dim(
                embedding, self.history_index.d
            )
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # 添加到索引
        if self.history_index is None:
            import faiss
            emb_dim = embedding.shape[0]
            self.history_index = faiss.IndexFlatIP(emb_dim)
            self.history_embeddings = []
            self.history_records = []
        self.history_index.add(embedding.reshape(1, -1))
        # 兼容 numpy 数组和列表
        if isinstance(self.history_embeddings, np.ndarray):
            self.history_embeddings = np.vstack([self.history_embeddings, embedding])
        else:
            self.history_embeddings.append(embedding)
        self.history_records.append(record)

        logger.debug(f"添加历史记录到 Few-Shot 索引，当前共 {len(self.history_records)} 条")

    def retrieve(self, query: str, top_k: int = 5,
                node_type_filter: Optional[str] = None,
                database_id_filter: Optional[int] = None) -> List[Tuple[Dict, float]]:
        """
        检索知识库

        Args:
            query: 查询文本
            top_k: 返回数量
            node_type_filter: 节点类型过滤
            database_id_filter: 数据库 ID 过滤

        Returns:
            List[Tuple[Dict, float]]: 检索结果
        """
        if top_k <= 0:
            return []
        results = self.retriever.search(query, top_k * 2)  # 先多取一些

        # 过滤
        if node_type_filter:
            results = [(m, s) for m, s in results
                      if m.get('node_type') == node_type_filter]

        # 按数据库 ID 过滤
        if database_id_filter is not None:
            results = [(m, s) for m, s in results
                      if m.get('database_id') == database_id_filter or m.get('db_id') == database_id_filter]

        # 返回 top_k
        return results[:top_k]

    def retrieve_tables(self, query: str, top_k: int = 5, database_id: Optional[int] = None) -> List[Tuple[Dict, float]]:
        """检索相关的表"""
        return self.retrieve(query, top_k, node_type_filter='table', database_id_filter=database_id)

    def retrieve_columns(self, query: str, top_k: int = 5, database_id: Optional[int] = None) -> List[Tuple[Dict, float]]:
        """检索相关的字段"""
        return self.retrieve(query, top_k, node_type_filter='column', database_id_filter=database_id)

    def retrieve_all(self, query: str, top_k: int = 5, database_id: Optional[int] = None) -> List[Tuple[Dict, float]]:
        """检索所有类型的节点"""
        return self.retrieve(query, top_k, database_id_filter=database_id)

    def get_few_shot_examples_formatted(self, query: str, top_k: int = 3) -> str:
        """
        获取格式化的 Few-Shot 示例字符串，用于注入 Prompt

        Args:
            query: 当前查询
            top_k: 示例数量

        Returns:
            str: 格式化的示例字符串
        """
        examples = self.retrieve_few_shot_examples(query, top_k)

        if not examples:
            return "暂无历史示例"

        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"""示例 {i}:
  用户查询：{ex.get('user_query', '')}
  SQL: {ex.get('generated_sql', '')}""")

        return "\n\n".join(formatted)

    # ==================== 索引持久化 ====================

    def save_index(self, directory: str) -> bool:
        """保存知识库索引到磁盘"""
        return self.retriever.save_index(directory)

    def load_index(self, directory: str) -> bool:
        """从磁盘加载知识库索引"""
        return self.retriever.load_index(directory)
