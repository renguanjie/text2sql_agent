"""
文本检索模块
实现 BM25 + TF-IDF + FAISS 向量检索的三路混合检索算法
支持真实 Embedding 模型和动态 Few-Shot 检索
"""
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import math
import re
from loguru import logger
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS 未安装，向量检索功能将不可用。安装：pip install faiss-cpu")

# 导入 Embedding 工厂
from ..embedding_factory import EmbeddingFactory, get_embedding_factory


class TextPreprocessor:
    """文本预处理器"""

    def __init__(self):
        """初始化预处理器"""
        # 中英文标点符号
        self.punctuation = r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~，。！？；：""''、·…—（）《》「」『』]'

    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词

        Args:
            text: 输入文本

        Returns:
            List[str]: 分词结果
        """
        # 转小写
        text = text.lower()

        # 移除标点符号
        text = re.sub(self.punctuation, ' ', text)

        # 按空格和特殊字符分割
        # 对于中文，这里使用简单的字符级分词
        # 可以集成 jieba 等分词工具来提升效果
        words = text.split()

        # 过滤空字符串和单个字符（保留数字）
        words = [w for w in words if w and (len(w) > 1 or w.isdigit())]

        return words

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
                 embedding_dim: int = 1536, index_type: str = "flat"):
        """
        初始化向量嵌入器

        Args:
            embedding_factory: Embedding 工厂实例
            embedding_dim: 向量维度 (通义千问 v1=1536, v2=1536, v3=1024)
            index_type: FAISS 索引类型 ("flat", "ivf", "hnsw")
        """
        self.embedding_factory = embedding_factory or get_embedding_factory()
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
        if not FAISS_AVAILABLE:
            logger.warning("FAISS 不可用，跳过向量索引构建")
            return

        logger.info(f"构建 FAISS 向量索引：{len(documents)} 个文档，使用真实 Embedding 模型")

        self.documents = documents

        # 使用真实 Embedding 模型计算向量
        logger.info(f"调用 {self.embedding_factory.provider}/{self.embedding_factory.model} 计算 Embedding...")
        self.embedding_matrix = self._compute_embeddings(documents)

        # 构建 FAISS 索引
        emb_dim = self.embedding_matrix.shape[1]
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(emb_dim)  # 内积相似度
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(emb_dim, 32)
        else:
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
        if not FAISS_AVAILABLE or self.index is None:
            return np.array([]), np.array([])

        # 计算查询向量
        query_embedding = self._compute_single_embedding(query)

        # 搜索
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

        return indices[0], scores[0]

    def _compute_single_embedding(self, text: str) -> np.ndarray:
        """计算单个文本的 Embedding 向量"""
        embedding = self.embedding_factory.get_embedding(text)
        # L2 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)


class HybridRetriever:
    """BM25 + TF-IDF + FAISS 三路混合检索器"""

    def __init__(self, bm25_k1: float = 1.5, bm25_b: float = 0.75,
                 tfidf_max_features: int = 5000,
                 bm25_weight: float = 0.4, tfidf_weight: float = 0.3,
                 dense_weight: float = 0.3,
                 embedding_factory: Optional[EmbeddingFactory] = None,
                 embedding_dim: int = 1536):
        """
        初始化混合检索器

        Args:
            bm25_k1: BM25 k1 参数
            bm25_b: BM25 b 参数
            tfidf_max_features: TF-IDF 最大特征数
            bm25_weight: BM25 权重（0-1）
            tfidf_weight: TF-IDF 权重（0-1）
            dense_weight: 稠密向量权重（0-1）
            embedding_factory: Embedding 工厂实例
            embedding_dim: 向量嵌入维度
        """
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.tfidf_max_features = tfidf_max_features
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight
        self.dense_weight = dense_weight

        # 归一化权重（确保总和为 1）
        total_weight = bm25_weight + tfidf_weight + dense_weight
        if total_weight > 0:
            self.bm25_weight /= total_weight
            self.tfidf_weight /= total_weight
            self.dense_weight /= total_weight

        self.preprocessor = TextPreprocessor()
        self.bm25_model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.documents: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.doc_metadata: List[Dict] = []

        # FAISS 向量检索器 - 使用真实 Embedding 模型
        self.dense_embedder = DenseVectorEmbedder(
            embedding_factory=embedding_factory,
            embedding_dim=embedding_dim,
            index_type="flat"
        )

    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        索引文档

        Args:
            documents: 文档列表
            metadata: 文档元数据列表
        """
        logger.info(f"索引 {len(documents)} 个文档")

        self.documents = documents
        self.doc_metadata = metadata or [{} for _ in documents]

        # 预处理和分词
        self.tokenized_docs = [self.preprocessor.tokenize(doc) for doc in documents]

        # 构建 BM25 模型
        self.bm25_model = BM25Okapi(self.tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)
        logger.info("BM25 模型构建完成")

        # 构建 TF-IDF 向量
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            token_pattern=r'(?u)\b\w+\b',
            stop_words=None
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        logger.info("TF-IDF 矩阵构建完成")

        # 构建 FAISS 向量索引
        self.dense_embedder.build_index(documents)
        logger.info("三路索引构建完成 (BM25 + TF-IDF + FAISS)")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        搜索文档 - 三路混合检索 (BM25 + TF-IDF + FAISS 向量)

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[Tuple[Dict, float]]: (文档元数据，相似度分数) 列表
        """
        if not self.documents or not self.bm25_model:
            logger.warning("检索器未初始化，返回空结果")
            return []

        # 预处理查询
        tokenized_query = self.preprocessor.tokenize(query)

        # 1. BM25 分数
        bm25_scores = self.bm25_model.get_scores(tokenized_query)

        # 2. TF-IDF 分数
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = np.array(query_tfidf.dot(self.tfidf_matrix.T).todense()).flatten()

        # 3. FAISS 向量相似度分数
        if FAISS_AVAILABLE and self.dense_embedder.index is not None:
            dense_indices, dense_scores = self.dense_embedder.search(query, top_k=len(self.documents))
            # 将向量分数映射到文档索引
            dense_vec = np.zeros(len(self.documents), dtype=np.float32)
            for idx, score in zip(dense_indices, dense_scores):
                if idx < len(self.documents):
                    dense_vec[idx] = score
            dense_scores = dense_vec
        else:
            dense_scores = np.zeros(len(self.documents))

        # 归一化分数到 [0, 1] 范围
        bm25_norm = self._normalize_scores(bm25_scores)
        tfidf_norm = self._normalize_scores(tfidf_scores)
        dense_norm = self._normalize_scores(dense_scores)

        # 三路分数融合
        final_scores = (self.bm25_weight * bm25_norm +
                       self.tfidf_weight * tfidf_norm +
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
                    'tfidf': float(tfidf_norm[idx]),
                    'dense': float(dense_norm[idx]),
                    'final': float(final_scores[idx])
                }
                results.append((metadata, float(final_scores[idx])))

        logger.info(f"检索到 {len(results)} 个相关文档 (BM25+TF-IDF+FAISS)")
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
        min_score = np.min(scores)
        max_score = np.max(scores)

        if max_score == min_score:
            return np.zeros_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def add_document(self, document: str, metadata: Optional[Dict] = None):
        """
        添加单个文档

        Args:
            document: 文档内容
            metadata: 文档元数据
        """
        self.documents.append(document)
        self.doc_metadata.append(metadata or {})
        self.tokenized_docs.append(self.preprocessor.tokenize(document))

        # 重建索引
        self.bm25_model = BM25Okapi(self.tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)

    def remove_document(self, index: int):
        """
        移除文档

        Args:
            index: 文档索引
        """
        if 0 <= index < len(self.documents):
            self.documents.pop(index)
            self.doc_metadata.pop(index)
            self.tokenized_docs.pop(index)

            # 重建索引
            if self.documents:
                self.bm25_model = BM25Okapi(self.tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
            else:
                self.bm25_model = None
                self.tfidf_matrix = None

    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)


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
        self.embedding_factory = embedding_factory or get_embedding_factory()
        self.retriever = HybridRetriever(
            bm25_k1=config.get('bm25_k1', 1.5),
            bm25_b=config.get('bm25_b', 0.75),
            tfidf_max_features=config.get('tfidf_max_features', 5000),
            bm25_weight=config.get('bm25_weight', 0.4),
            tfidf_weight=config.get('tfidf_weight', 0.3),
            dense_weight=config.get('dense_weight', 0.3),
            embedding_factory=self.embedding_factory,
            embedding_dim=config.get('embedding_dim', 1536)  # 通义千问 v1 默认 1536 维
        )
        self.node_index: Dict[int, Dict] = {}  # 节点索引

        # Few-Shot 示例检索配置
        self.few_shot_top_k = config.get('few_shot_top_k', 3)
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

    def build_few_shot_index(self, history_records: List[Dict]):
        """
        从历史成功记录构建 Few-Shot 示例索引

        Args:
            history_records: 历史记录列表，每条记录包含 user_query 和 generated_sql
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS 不可用，无法构建 Few-Shot 索引")
            return

        # 筛选成功的记录
        successful_records = [
            r for r in history_records
            if r.get('execution_status') == 'success' or r.get('validation_status') == 'success'
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

        k = top_k or self.few_shot_top_k
        k = min(k, len(self.history_records))

        # 计算查询向量
        query_embedding = self.embedding_factory.get_embedding(query)
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
            if idx < len(self.history_records) and score > 0:
                record = self.history_records[idx].copy()
                record['similarity_score'] = float(score)
                examples.append(record)

        logger.info(f"检索到 {len(examples)} 个 Few-Shot 示例")
        return examples

    def add_history_record(self, record: Dict):
        """
        添加单条历史记录到 Few-Shot 索引

        Args:
            record: 历史记录，包含 user_query 和 generated_sql
        """
        if not FAISS_AVAILABLE:
            return

        # 只添加成功的记录
        if record.get('execution_status') != 'success' and record.get('validation_status') != 'success':
            return

        # 计算 Embedding
        embedding = self.embedding_factory.get_embedding(record.get('user_query', ''))
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
                node_type_filter: Optional[str] = None) -> List[Tuple[Dict, float]]:
        """
        检索知识库

        Args:
            query: 查询文本
            top_k: 返回数量
            node_type_filter: 节点类型过滤

        Returns:
            List[Tuple[Dict, float]]: 检索结果
        """
        results = self.retriever.search(query, top_k * 2)  # 先多取一些

        # 过滤
        if node_type_filter:
            results = [(m, s) for m, s in results
                      if m.get('node_type') == node_type_filter]

        # 返回 top_k
        return results[:top_k]

    def retrieve_tables(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """检索相关的表"""
        return self.retrieve(query, top_k, node_type_filter='table')

    def retrieve_columns(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """检索相关的字段"""
        return self.retrieve(query, top_k, node_type_filter='column')

    def retrieve_all(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """检索所有类型的节点"""
        return self.retrieve(query, top_k)

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
