"""
SQL 生成器模块
整合检索、链式处理，提供统一的 SQL 生成接口（支持异步）
支持多数据库方言动态检测
"""
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from hmac import compare_digest
from hmac import digest as hmac_digest
from secrets import token_bytes
from threading import Lock
from time import monotonic
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..chain.sql_chain import QueryRewriteChain, SQLGenerationChain
from ..config_validation import RuntimeConfigurationError, validate_history_max_records
from ..history.identity import GovernedHistoryIdentity
from ..history.mysql_client import MySQLClient
from ..history.proof import (
    GovernedHistoryFinalization,
    GovernedHistorySave,
    HistoryProofIssuer,
)
from ..retrieval.bm25_tfidf import KnowledgeRetriever
from ..utils.log_sanitizer import log_safe_exception, text_fingerprint
from .dialect_detector import SQLDialectDetector
from .error_codes import (
    EXECUTION_CONTEXT_REQUIRED,
    GENERATION_FAILED,
    HISTORY_CONTEXT_MISMATCH,
)
from .validator import (
    constrain_read_query,
    is_safe_read_query,
    validate_query_scope,
    validate_sql,
)


@dataclass(frozen=True, slots=True)
class _GovernedHistoryBinding:
    session_sha256: str
    generated_sql_sha256: str
    user_query_sha256: str
    identity: GovernedHistoryIdentity
    validation_succeeded: bool
    expires_at: float


class SQLGenerator:
    """SQL 生成器 - 核心业务逻辑封装"""

    def __init__(self, llm, knowledge_retriever: KnowledgeRetriever,
                 sql_chain: SQLGenerationChain,
                 mysql_client: Optional[MySQLClient] = None,
                 config: Optional[Dict] = None,
                 neo4j_client=None,
                 correctness_pipeline=None,
                 history_proof_issuer: HistoryProofIssuer | None = None):
        self.llm = llm
        self.knowledge_retriever = knowledge_retriever
        self.sql_chain = sql_chain
        self.mysql_client = mysql_client
        self.config = config or {}
        self.neo4j_client = neo4j_client
        self.correctness_pipeline = correctness_pipeline
        self._history_proof_issuer = history_proof_issuer
        history_capacity = validate_history_max_records(
            self.config.get('history_max_records', 1000),
            key="history_max_records",
        )
        binding_ttl = self.config.get(
            'governed_history_binding_ttl_seconds', 900
        )
        if (
            isinstance(binding_ttl, bool)
            or not isinstance(binding_ttl, (int, float))
            or not 0 < binding_ttl <= 3600
        ):
            raise ValueError(
                "governed_history_binding_ttl_seconds 必须大于 0 且不超过 3600"
            )
        self._governed_history_binding_capacity = min(history_capacity, 4096)
        self._governed_history_binding_ttl_seconds = float(binding_ttl)
        self._governed_history_digest_key = token_bytes(32)
        self._clock = monotonic
        self._governed_history_bindings: OrderedDict[
            int, _GovernedHistoryBinding
        ] = OrderedDict()
        self._unavailable_governed_history_ids: OrderedDict[int, None] = (
            OrderedDict()
        )
        self._governed_history_lock = Lock()

        self.rewrite_chain = QueryRewriteChain(
            llm, prompt_max_length=self.config.get('prompt_max_length', 4000)
        ) if self.config.get('enable_rewrite', False) else None
        self.dialect_detector = SQLDialectDetector(neo4j_client) if neo4j_client else None
        logger.info(f"SQL 生成器初始化完成，方言检测器：{'已启用' if self.dialect_detector else '未启用'}")

    # ==================== 核心共享逻辑 ====================

    def _prepare_query_sync(self, user_query: str) -> str:
        """查询改写（同步版本，避免 asyncio.run 事件循环冲突）"""
        if self.rewrite_chain:
            return self.rewrite_chain.rewrite(user_query)
        return user_query

    async def _prepare_query_async(self, user_query: str) -> str:
        """异步改写查询。"""
        if self.rewrite_chain:
            return await self.rewrite_chain.rewrite_async(user_query)
        return user_query

    def _detect_dialect_and_db(self, user_query: str,
                                 target_dialect: Optional[str],
                                 target_database_id: Optional[int]) -> Tuple[str, Optional[int]]:
        """检测方言和目标数据库"""
        if target_dialect:
            return target_dialect, target_database_id
        if self.dialect_detector:
            return self.dialect_detector.detect_from_query(user_query)
        return self.config.get('sql_dialect', 'mysql'), None

    def _validate_governed_dialect(self, dialect: Any) -> str:
        if self.correctness_pipeline is None:
            return dialect
        if not isinstance(dialect, str) or dialect.casefold() != "mysql":
            raise RuntimeConfigurationError(
                "SQL_DIALECT 在启用 SEMANTIC_GOVERNANCE_ENABLED 时仅支持 mysql"
            )
        return "mysql"

    def _switch_dialect_if_needed(self, dialect: str):
        """按需切换方言（避免重建整个 SQLChain）"""
        if dialect and dialect != self.sql_chain.dialect:
            logger.info(f"切换方言：{self.sql_chain.dialect} -> {dialect}")
            self.sql_chain.dialect = dialect
            from ..chain.prompts import SQLPromptTemplates
            self.sql_chain.prompt_templates = SQLPromptTemplates(
                dialect=dialect,
                max_length=getattr(self.sql_chain, 'prompt_max_length', 4000),
            )

    def _get_few_shot_examples(self, query: str) -> str:
        """获取 Few-Shot 示例"""
        if self.knowledge_retriever.history_index is None:
            return ""
        return self.knowledge_retriever.get_few_shot_examples_formatted(
            query, top_k=self.config.get('few_shot_top_k', 3)
        )

    def _retrieve_knowledge(self, query: str, database_id: Optional[int]) -> List:
        """检索知识库"""
        results = self.knowledge_retriever.retrieve_all(
            query, top_k=self.sql_chain.retrieval_top_k, database_id=database_id
        )
        logger.info(f"检索到 {len(results)} 个相关知识节点")
        return results

    def _save_history_and_few_shot(
        self,
        user_query: str,
        result: Dict,
        retrieval_query: str,
        session_id: Optional[str],
        governance_identity: GovernedHistoryIdentity | None = None,
    ):
        """保存待执行历史；仅执行成功的记录才会进入 Few-Shot。"""
        if not self.mysql_client or not result.get('sql'):
            return
        validation = result.get('validation')
        if validation:
            validation_status = 'success' if validation.get('valid') else 'fail'
            issues = validation.get('issues')
            if isinstance(issues, (list, tuple)):
                validation_error = '; '.join(
                    issue.get('code')
                    for issue in issues
                    if isinstance(issue, dict) and isinstance(issue.get('code'), str)
                ) or None
            else:
                validation_error = '; '.join(validation.get('errors', [])) or None
        else:
            validation_status = 'pending'
            validation_error = result.get('error')
        if governance_identity is None:
            history_id = self.mysql_client.save_sql_history(
                user_query=user_query,
                generated_sql=result['sql'],
                session_id=session_id,
                matched_knowledge=result.get('matched_knowledge'),
                retrieval_query=retrieval_query,
                validation_status=validation_status,
                validation_error=validation_error,
            )
        elif self._history_proof_issuer is not None:
            command = GovernedHistorySave(
                session_id=session_id,
                user_query=user_query,
                generated_sql=result['sql'],
                matched_knowledge_json=(
                    json.dumps(
                        result.get('matched_knowledge'),
                        ensure_ascii=False,
                    )
                    if result.get('matched_knowledge')
                    else None
                ),
                retrieval_query=retrieval_query,
                validation_status=validation_status,
                validation_error=validation_error,
                semantic_datasource=governance_identity.datasource,
                semantic_dialect=governance_identity.dialect,
                semantic_version=governance_identity.semantic_version,
                semantic_content_sha256=governance_identity.content_sha256,
            )
            proof = self._history_proof_issuer.issue(command)
            history_id = self.mysql_client.save_governed_sql_history(command, proof)
        else:
            logger.warning("governed_history_save_rejected reason=missing_proof_issuer")
            history_id = None
        if history_id:
            result['history_id'] = history_id
            if (
                self.correctness_pipeline is not None
                and validation_status == 'success'
                and result.get('executable') is True
                and isinstance(history_id, int)
                and not isinstance(history_id, bool)
                and isinstance(session_id, str)
                and session_id
                and isinstance(governance_identity, GovernedHistoryIdentity)
            ):
                self._remember_governed_history_binding(
                    history_id,
                    session_id=session_id,
                    generated_sql=result['sql'],
                    user_query=user_query,
                    identity=governance_identity,
                )

    def _identity_digest(self, domain: str, value: str) -> str:
        payload = f"{domain}\0{value}".encode("utf-8")
        return hmac_digest(
            self._governed_history_digest_key, payload, "sha256"
        ).hex()

    def _canonical_sql(self, sql: str) -> str | None:
        try:
            import sqlglot

            dialect = self.sql_chain.dialect
            return sqlglot.parse_one(sql, read=dialect).sql(dialect=dialect)
        except Exception:
            return None

    def _remember_governed_history_binding(
        self,
        history_id: int,
        *,
        session_id: str,
        generated_sql: str,
        user_query: str,
        identity: GovernedHistoryIdentity,
    ) -> None:
        canonical_sql = self._canonical_sql(generated_sql)
        if canonical_sql is None:
            return
        now = self._clock()
        binding = _GovernedHistoryBinding(
            session_sha256=self._identity_digest("session", session_id),
            generated_sql_sha256=self._identity_digest("sql", canonical_sql),
            user_query_sha256=self._identity_digest("question", user_query),
            identity=identity,
            validation_succeeded=True,
            expires_at=now + self._governed_history_binding_ttl_seconds,
        )
        with self._governed_history_lock:
            self._prune_governed_history_bindings_locked(now)
            self._governed_history_bindings.pop(history_id, None)
            self._unavailable_governed_history_ids.pop(history_id, None)
            while (
                len(self._governed_history_bindings)
                >= self._governed_history_binding_capacity
            ):
                evicted_history_id, _binding = (
                    self._governed_history_bindings.popitem(last=False)
                )
                self._mark_governed_history_unavailable_locked(evicted_history_id)
            self._governed_history_bindings[history_id] = binding

    def _mark_governed_history_unavailable_locked(self, history_id: int) -> None:
        self._unavailable_governed_history_ids.pop(history_id, None)
        self._unavailable_governed_history_ids[history_id] = None
        while len(self._unavailable_governed_history_ids) > 4096:
            self._unavailable_governed_history_ids.popitem(last=False)

    def _prune_governed_history_bindings_locked(self, now: float) -> None:
        expired = [
            history_id
            for history_id, binding in self._governed_history_bindings.items()
            if binding.expires_at <= now
        ]
        for history_id in expired:
            self._governed_history_bindings.pop(history_id, None)
            self._mark_governed_history_unavailable_locked(history_id)

    def _consume_governed_history_binding(
        self,
        history_id: int,
        *,
        session_id: str,
        sql: str,
        user_query: str | None,
        semantic_version: str | None,
        content_sha256: str | None,
        governance_identity: GovernedHistoryIdentity,
    ) -> bool:
        """Consume a matching local capability; absence always fails closed."""
        canonical_sql = self._canonical_sql(sql)
        if canonical_sql is None or not isinstance(user_query, str):
            return False
        expected_session = self._identity_digest("session", session_id)
        expected_sql = self._identity_digest("sql", canonical_sql)
        expected_question = self._identity_digest("question", user_query)
        now = self._clock()
        with self._governed_history_lock:
            self._prune_governed_history_bindings_locked(now)
            binding = self._governed_history_bindings.get(history_id)
            if binding is None:
                return False
            if (
                not binding.validation_succeeded
                or not compare_digest(binding.session_sha256, expected_session)
                or not compare_digest(binding.generated_sql_sha256, expected_sql)
                or not compare_digest(binding.user_query_sha256, expected_question)
                or binding.identity != governance_identity
                or binding.identity.semantic_version != semantic_version
                or binding.identity.content_sha256 != content_sha256
            ):
                return False
            self._governed_history_bindings.pop(history_id, None)
            self._mark_governed_history_unavailable_locked(history_id)
            return True

    def _promote_history_to_few_shot(
        self,
        history_id: int,
        *,
        expected_history: Mapping[str, Any],
        governance_identity: GovernedHistoryIdentity | None,
    ) -> None:
        """将真实执行成功的历史记录加入 Few-Shot 索引。"""
        if governance_identity is not None:
            promoted = dict(expected_history)
            promoted.update(
                {
                    'id': history_id,
                    'execution_status': 'success',
                    'governance_validated': True,
                }
            )
            self.knowledge_retriever.add_history_record(
                promoted,
                governance_identity=governance_identity,
            )
            return
        history = self.mysql_client.get_history_detail(history_id)
        if (
            isinstance(history, Mapping)
            and history.get('session_id') == expected_history.get('session_id')
            and history.get('user_query') == expected_history.get('user_query')
            and isinstance(history.get('generated_sql'), str)
            and isinstance(expected_history.get('generated_sql'), str)
            and self._same_sql(
                history['generated_sql'], expected_history['generated_sql']
            )
            and history.get('validation_status') == 'success'
            and history.get('execution_status') == 'success'
        ):
            self.knowledge_retriever.add_history_record(history)

    def _bound_history(
        self,
        history_id: int,
        *,
        session_id: str | None,
        sql: str,
        user_query: str | None,
        semantic_version: str | None,
        content_sha256: str | None,
        governance_identity: GovernedHistoryIdentity | None = None,
    ) -> tuple[
        Mapping[str, Any] | None,
        GovernedHistoryIdentity | None,
        str | None,
    ]:
        """Resolve a history row only when all execution identity fields match."""
        if not session_id:
            return None, None, '执行历史与当前会话或 SQL 不匹配'
        history = self.mysql_client.get_history_detail(history_id)
        if not isinstance(history, Mapping):
            return None, None, '执行历史与当前会话或 SQL 不匹配'
        if history.get('session_id') != session_id:
            return None, None, '执行历史与当前会话或 SQL 不匹配'
        if history.get('validation_status') != 'success':
            return None, None, '执行历史未通过校验'
        stored_sql = history.get('generated_sql')
        if not isinstance(stored_sql, str) or not self._same_sql(stored_sql, sql):
            return None, None, '执行历史与当前会话或 SQL 不匹配'

        identity = None
        if self.correctness_pipeline is not None:
            identity = governance_identity
            if identity is None:
                return None, None, '执行历史与当前治理上下文不匹配'
            governance_flag = history.get('governance_validated')
            is_unvalidated = governance_flag is False or (
                type(governance_flag) is int and governance_flag == 0
            )
            if (
                not identity.matches_audit_identity(history)
                or semantic_version != identity.semantic_version
                or content_sha256 != identity.content_sha256
                or history.get('user_query') != user_query
                or history.get('execution_status') != 'pending'
                or not is_unvalidated
            ):
                return None, None, '执行历史与当前治理上下文不匹配'
            local_binding = self._consume_governed_history_binding(
                history_id,
                session_id=session_id,
                sql=sql,
                user_query=user_query,
                semantic_version=semantic_version,
                content_sha256=content_sha256,
                governance_identity=identity,
            )
            if local_binding is not True:
                return None, None, '执行历史与当前治理上下文不匹配'
        return history, identity, None

    def _same_sql(self, left: str, right: str) -> bool:
        left_canonical = self._canonical_sql(left)
        right_canonical = self._canonical_sql(right)
        return left_canonical is not None and left_canonical == right_canonical

    def _history_context_error(
        self,
        message: str,
        *,
        sql: str,
        semantic_version: str | None,
        content_sha256: str | None,
    ) -> Dict[str, Any]:
        if self.correctness_pipeline is not None:
            return self._governed_error_payload(
                code=HISTORY_CONTEXT_MISMATCH,
                message=message,
                semantic_version=semantic_version,
                content_sha256=content_sha256,
                sql=sql,
                stage='safety',
            ) | {'data': None}
        return {'status': 'error', 'error': message, 'data': None}

    def _update_bound_history(
        self,
        history_id: int,
        history: Mapping[str, Any],
        *,
        execution_status: str,
        execution_error: str | None = None,
        execution_result: Any = None,
    ) -> bool:
        if self.correctness_pipeline is not None:
            if self._history_proof_issuer is None:
                return False
            command = GovernedHistoryFinalization(
                history_id=history_id,
                session_id=history['session_id'],
                user_query=history['user_query'],
                generated_sql=history['generated_sql'],
                execution_status=execution_status,
                execution_error=execution_error,
                execution_result_json=(
                    json.dumps(
                        execution_result,
                        ensure_ascii=False,
                        default=str,
                    )
                    if execution_result is not None
                    else None
                ),
                semantic_datasource=history['semantic_datasource'],
                semantic_dialect=history['semantic_dialect'],
                semantic_version=history['semantic_version'],
                semantic_content_sha256=history['semantic_content_sha256'],
            )
            proof = self._history_proof_issuer.issue(command)
            return self.mysql_client.finalize_governed_sql_execution(command, proof)
        return self.mysql_client.update_sql_execution(
            history_id=history_id,
            execution_status=execution_status,
            execution_error=execution_error,
            execution_result=execution_result,
            session_id=history['session_id'],
            generated_sql=history['generated_sql'],
            expected_user_query=history.get('user_query'),
            require_validation_success=True,
        )

    def _validate_generation_result(self, result: Dict, dialect: str,
                                    allowed_tables: Optional[List[str]] = None,
                                    allowed_schemas: Optional[List[str]] = None) -> Dict:
        """为提取成功的 SQL 附加真实校验结果，供历史与 UI 复用。"""
        if result.get('status') != 'success' or not result.get('sql'):
            return result
        if not self.config.get('sql_validation_enabled', True):
            return result

        sql = result['sql']
        max_length = self.config.get('sql_max_length', 2000)
        if max_length and len(sql) > max_length:
            result['validation'] = {
                'valid': False,
                'syntax_valid': False,
                'execution_valid': False,
                'errors': [f'SQL 长度超过限制（{max_length} 字符）'],
                'warnings': [],
                'formatted_sql': sql,
            }
        else:
            result['validation'] = validate_sql(sql, dialect=dialect)
            if result['validation'].get('valid') and self.config.get(
                'sql_scope_enforcement', True
            ):
                scope = validate_query_scope(
                    sql,
                    allowed_tables=allowed_tables,
                    allowed_schemas=allowed_schemas,
                    blocked_tables=self.config.get('sql_blocked_tables'),
                    dialect=dialect,
                )
                if not scope['valid']:
                    result['validation']['valid'] = False
                    result['validation']['execution_valid'] = False
                    result['validation']['errors'].extend(scope['errors'])
        return result

    def _attach_execution_scope(self, result: Dict, ctx: Dict[str, Any]) -> None:
        """将本次生成使用的 Schema 边界附在结果上，供执行阶段复核。"""
        tables = {
            table.get('table_name') or table.get('name')
            for table in ctx['schema_info'].get('tables', [])
        }
        result['allowed_tables'] = sorted(name for name in tables if name)
        database = getattr(self.mysql_client, 'database', None) if self.mysql_client else None
        result['allowed_schemas'] = [database] if database else []
        result['target_database_id'] = ctx.get('detected_database_id')

    def _prepare_governed_context(
        self, user_query: str
    ) -> tuple[
        Any | None,
        str,
        str | None,
        str | None,
        GovernedHistoryIdentity | None,
        Dict[str, Any] | None,
    ]:
        """Pin the immutable correctness request before invoking the LLM."""
        if self.correctness_pipeline is None:
            return None, "", None, None, None, None
        request = None
        try:
            request = self.correctness_pipeline.prepare_generation(user_query)
            semantic_context, semantic_version, content_sha256 = (
                self.correctness_pipeline.describe_prepared(request)
            )
            identity = self.correctness_pipeline.describe_prepared_history_identity(
                request
            )
            return (
                request,
                semantic_context,
                semantic_version,
                content_sha256,
                identity,
                None,
            )
        except Exception as exc:
            if request is not None:
                self._discard_prepared_safely(request)
            log_safe_exception("governed_generation_context", exc)
            return None, "", None, None, None, self._governed_error_payload(
                code="CORRECTNESS_UNAVAILABLE",
                message="SQL 正确性校验暂时不可用",
                semantic_version=None,
                content_sha256=None,
                sql=None,
            )

    def _discard_prepared_safely(self, request: Any) -> None:
        if self.correctness_pipeline is None or request is None:
            return
        try:
            self.correctness_pipeline.discard_prepared(request)
        except Exception as exc:
            log_safe_exception("prepared_request_discard", exc)

    def _generation_failure_payload(
        self,
        user_query: str,
        exc: Exception,
        *,
        stage: str,
        semantic_version: str | None = None,
        content_sha256: str | None = None,
    ) -> Dict[str, Any]:
        """Return a fixed public failure while recording only safe diagnostics."""
        log_safe_exception(stage, exc, fingerprint_source=user_query)
        message = "SQL 生成暂时不可用，请稍后重试"
        if self.correctness_pipeline is None:
            return {
                'sql': None,
                'status': 'error',
                'error': message,
                'raw_response': None,
            }
        result = self._governed_error_payload(
            code=GENERATION_FAILED,
            message=message,
            semantic_version=semantic_version,
            content_sha256=content_sha256,
            sql=None,
            stage='syntax',
        )
        result['user_query'] = user_query
        return result

    @staticmethod
    def _governed_error_payload(
        *,
        code: str,
        message: str,
        semantic_version: str | None,
        content_sha256: str | None = None,
        sql: str | None,
        stage: str = "semantic",
    ) -> Dict[str, Any]:
        return {
            'sql': sql,
            'status': 'error',
            'error': message,
            'semantic_version': semantic_version,
            'content_sha256': content_sha256,
            'validation': {
                'valid': False,
                'issues': [{
                    'code': code,
                    'message': message,
                    'objects': [],
                    'stage': stage,
                }],
            },
            'repair_attempts': 0,
            'executable': False,
        }

    @staticmethod
    def _merge_correctness_result(
        result: Dict[str, Any], correctness, user_query: str
    ) -> None:
        public = {
            'sql': correctness.sql,
            'status': correctness.status,
            'semantic_version': correctness.semantic_version,
            'content_sha256': correctness.content_sha256,
            'validation': correctness.validation.model_dump(mode='json'),
            'repair_attempts': correctness.repair_attempts,
            'executable': correctness.executable,
            'user_query': user_query,
        }
        if correctness.status == 'error':
            public['error'] = (
                correctness.validation.issues[0].message
                if correctness.validation.issues
                else 'SQL 正确性校验暂时不可用'
            )
        result.update(public)

    def _govern_candidate_sync(
        self,
        result: Dict[str, Any],
        user_query: str,
        request: Any,
        semantic_version: str,
        content_sha256: str,
    ) -> None:
        if result.get('status') != 'success' or not result.get('sql'):
            failure = self._governed_error_payload(
                code=GENERATION_FAILED,
                message='未生成可供正确性校验的 SQL',
                semantic_version=semantic_version,
                content_sha256=content_sha256,
                sql=result.get('sql'),
                stage='syntax',
            )
            failure['error'] = result.get('error') or failure['error']
            result.update(failure)
            result['user_query'] = user_query
            return
        correctness = self.correctness_pipeline.process_prepared(
            request=request,
            candidate_sql=result['sql'],
            repair=self.sql_chain.fix_sql_governed,
            allowed_tables=tuple(result.get('allowed_tables') or ()),
            allowed_schemas=tuple(result.get('allowed_schemas') or ()),
        )
        self._merge_correctness_result(result, correctness, user_query)

    async def _govern_candidate_async(
        self,
        result: Dict[str, Any],
        user_query: str,
        request: Any,
        semantic_version: str,
        content_sha256: str,
    ) -> None:
        if result.get('status') != 'success' or not result.get('sql'):
            failure = self._governed_error_payload(
                code=GENERATION_FAILED,
                message='未生成可供正确性校验的 SQL',
                semantic_version=semantic_version,
                content_sha256=content_sha256,
                sql=result.get('sql'),
                stage='syntax',
            )
            failure['error'] = result.get('error') or failure['error']
            result.update(failure)
            result['user_query'] = user_query
            return
        correctness = await self.correctness_pipeline.process_async_prepared(
            request=request,
            candidate_sql=result['sql'],
            repair=self.sql_chain.fix_sql_governed_async,
            allowed_tables=tuple(result.get('allowed_tables') or ()),
            allowed_schemas=tuple(result.get('allowed_schemas') or ()),
        )
        self._merge_correctness_result(result, correctness, user_query)

    # ==================== 公共方法 ====================

    def _build_pipeline_context(self, final_query: str, detected_dialect: str,
                                detected_database_id: Optional[int],
                                use_few_shot: bool) -> Dict[str, Any]:
        """在查询改写完成后组装检索、Schema 与方言上下文。"""
        knowledge_results = self._retrieve_knowledge(final_query, detected_database_id)
        few_shot_examples = self._get_few_shot_examples(final_query) if use_few_shot else ""
        schema_info = self._get_schema_info(database_id=detected_database_id)
        self._switch_dialect_if_needed(detected_dialect)
        return {
            'final_query': final_query,
            'detected_dialect': detected_dialect,
            'detected_database_id': detected_database_id,
            'knowledge_results': knowledge_results,
            'few_shot_examples': few_shot_examples,
            'schema_info': schema_info,
        }

    def _prepare_pipeline(self, user_query: str, target_dialect: Optional[str],
                          target_database_id: Optional[int],
                          use_few_shot: bool) -> Dict[str, Any]:
        """同步准备生成上下文。"""
        use_few_shot = use_few_shot and self.config.get('few_shot_enabled', True)
        detected_dialect, detected_database_id = self._detect_dialect_and_db(
            user_query, target_dialect, target_database_id
        )
        detected_dialect = self._validate_governed_dialect(detected_dialect)
        logger.info(f"方言检测结果：dialect={detected_dialect}, database_id={detected_database_id}")
        final_query = self._prepare_query_sync(user_query)
        if self.rewrite_chain:
            logger.info(
                f"查询改写完成：query_id={text_fingerprint(user_query)}, "
                f"input_chars={len(user_query)}, output_chars={len(final_query)}"
            )
        return self._build_pipeline_context(
            final_query, detected_dialect, detected_database_id, use_few_shot
        )

    async def _prepare_pipeline_async(self, user_query: str, target_dialect: Optional[str],
                                      target_database_id: Optional[int],
                                      use_few_shot: bool) -> Dict[str, Any]:
        """异步准备生成上下文，避免阻塞异步 LLM 查询改写。"""
        use_few_shot = use_few_shot and self.config.get('few_shot_enabled', True)
        detected_dialect, detected_database_id = self._detect_dialect_and_db(
            user_query, target_dialect, target_database_id
        )
        detected_dialect = self._validate_governed_dialect(detected_dialect)
        logger.info(f"方言检测结果：dialect={detected_dialect}, database_id={detected_database_id}")
        final_query = await self._prepare_query_async(user_query)
        if self.rewrite_chain:
            logger.info(
                f"异步查询改写完成：query_id={text_fingerprint(user_query)}, "
                f"input_chars={len(user_query)}, output_chars={len(final_query)}"
            )
        return self._build_pipeline_context(
            final_query, detected_dialect, detected_database_id, use_few_shot
        )

    def generate_sql(self, user_query: str, session_id: Optional[str] = None,
                    auto_save_history: bool = True,
                    use_few_shot: bool = True,
                    target_dialect: Optional[str] = None,
                    target_database_id: Optional[int] = None) -> Dict[str, Any]:
        """生成 SQL 的主入口（同步）"""
        logger.info(
            f"开始生成 SQL：query_id={text_fingerprint(user_query)}, chars={len(user_query)}"
        )

        try:
            ctx = self._prepare_pipeline(
                user_query, target_dialect, target_database_id, use_few_shot
            )
        except Exception as exc:
            return self._generation_failure_payload(
                user_query, exc, stage="sql_generator"
            )

        (
            correctness_request,
            semantic_context,
            semantic_version,
            content_sha256,
            governance_identity,
            governance_error,
        ) = self._prepare_governed_context(user_query)
        if governance_error is not None:
            governance_error['user_query'] = user_query
            return governance_error

        try:
            # 7. 生成 SQL
            generation_kwargs = dict(
                user_query=ctx['final_query'],
                schema_info=ctx['schema_info'],
                knowledge_results=ctx['knowledge_results'],
                session_id=session_id,
                few_shot_examples=ctx['few_shot_examples']
            )
            if self.correctness_pipeline is not None:
                generation_kwargs['semantic_context'] = semantic_context
            result = self.sql_chain.generate(**generation_kwargs)
            self._attach_execution_scope(result, ctx)
            if self.correctness_pipeline is not None:
                self._govern_candidate_sync(
                    result,
                    user_query,
                    correctness_request,
                    semantic_version or '',
                    content_sha256 or '',
                )
            else:
                self._validate_generation_result(
                    result,
                    ctx['detected_dialect'],
                    result['allowed_tables'],
                    result['allowed_schemas'],
                )
        except Exception as exc:
            result = self._generation_failure_payload(
                user_query,
                exc,
                stage="sql_generator",
                semantic_version=semantic_version,
                content_sha256=content_sha256,
            )
        finally:
            self._discard_prepared_safely(correctness_request)

        # 8. 保存历史
        if auto_save_history:
            self._save_history_and_few_shot(
                user_query,
                result,
                ctx['final_query'],
                session_id,
                governance_identity,
            )

        return result

    async def generate_sql_async(self, user_query: str, session_id: Optional[str] = None,
                                 auto_save_history: bool = True,
                                 use_few_shot: bool = True,
                                 target_dialect: Optional[str] = None,
                                 target_database_id: Optional[int] = None) -> Dict[str, Any]:
        """生成 SQL 的异步入口"""
        logger.info(
            f"开始异步生成 SQL：query_id={text_fingerprint(user_query)}, chars={len(user_query)}"
        )

        try:
            ctx = await self._prepare_pipeline_async(
                user_query, target_dialect, target_database_id, use_few_shot
            )
        except Exception as exc:
            return self._generation_failure_payload(
                user_query, exc, stage="sql_generator_async"
            )

        (
            correctness_request,
            semantic_context,
            semantic_version,
            content_sha256,
            governance_identity,
            governance_error,
        ) = self._prepare_governed_context(user_query)
        if governance_error is not None:
            governance_error['user_query'] = user_query
            return governance_error

        try:
            # 7. 异步生成 SQL
            generation_kwargs = dict(
                user_query=ctx['final_query'],
                schema_info=ctx['schema_info'],
                knowledge_results=ctx['knowledge_results'],
                session_id=session_id,
                few_shot_examples=ctx['few_shot_examples']
            )
            if self.correctness_pipeline is not None:
                generation_kwargs['semantic_context'] = semantic_context
            result = await self.sql_chain.generate_async(**generation_kwargs)
            self._attach_execution_scope(result, ctx)
            if self.correctness_pipeline is not None:
                await self._govern_candidate_async(
                    result,
                    user_query,
                    correctness_request,
                    semantic_version or '',
                    content_sha256 or '',
                )
            else:
                self._validate_generation_result(
                    result,
                    ctx['detected_dialect'],
                    result['allowed_tables'],
                    result['allowed_schemas'],
                )
        except Exception as exc:
            result = self._generation_failure_payload(
                user_query,
                exc,
                stage="sql_generator_async",
                semantic_version=semantic_version,
                content_sha256=content_sha256,
            )
        finally:
            self._discard_prepared_safely(correctness_request)

        # 8. 保存历史
        if auto_save_history:
            self._save_history_and_few_shot(
                user_query,
                result,
                ctx['final_query'],
                session_id,
                governance_identity,
            )

        return result

    def _get_schema_info(self, database_id: Optional[int] = None) -> Dict:
        """获取 Schema 信息"""
        if self.neo4j_client and getattr(self.neo4j_client, 'driver', None):
            return self._get_schema_from_neo4j(database_id)
        if self.mysql_client:
            return self._get_schema_from_mysql(database_id)
        return {'tables': [], 'columns': [], 'relationships': []}

    def _get_schema_from_neo4j(self, database_id: Optional[int] = None) -> Dict:
        """从 Neo4j 获取完整 Schema（表 + 列 + 关系）"""
        try:
            if database_id is not None:
                tables_result = self.neo4j_client.execute_query("""
                    MATCH (d:Database {database_id: $database_id})-[:HAS_TABLE]->(t:Table)
                    OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                    RETURN t.name as table_name, t.description as table_comment,
                           collect({
                               column_name: c.name,
                               column_type: c.type,
                               column_comment: c.description,
                               is_primary_key: c.is_primary_key,
                               is_nullable: c.is_nullable,
                               table_name: t.name
                           }) as columns
                """, {"database_id": database_id})

                tables = []
                all_columns = []
                for row in tables_result:
                    tables.append({
                        'table_name': row['table_name'],
                        'table_comment': row.get('table_comment', '')
                    })
                    for col in row.get('columns', []):
                        if col.get('column_name'):
                            all_columns.append(dict(col))

                # 获取关系（CONNECTS 类型，带权重信息）
                rel_result = self.neo4j_client.execute_query("""
                    MATCH (t1:Table)-[r:CONNECTS]->(t2:Table)
                    WHERE t1.database_id = $database_id OR t2.database_id = $database_id
                    RETURN t1.name as from_table, t2.name as to_table,
                           r.relationship_type as type, r.join_type as join_type,
                           r.join_sql as join_sql, r.from_column as from_column,
                           r.to_column as to_column, r.weight as weight,
                           r.extra_condition as extra_condition
                """, {"database_id": database_id})
            else:
                tables_result = self.neo4j_client.execute_query("""
                    MATCH (t:Table)
                    OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                    RETURN t.name as table_name, t.description as table_comment,
                           collect({
                               column_name: c.name,
                               column_type: c.type,
                               column_comment: c.description,
                               is_primary_key: c.is_primary_key,
                               is_nullable: c.is_nullable,
                               table_name: t.name
                           }) as columns
                """)

                tables = []
                all_columns = []
                for row in tables_result:
                    tables.append({
                        'table_name': row['table_name'],
                        'table_comment': row.get('table_comment', '')
                    })
                    for col in row.get('columns', []):
                        if col.get('column_name'):
                            all_columns.append(dict(col))

                rel_result = self.neo4j_client.execute_query("""
                    MATCH (t1:Table)-[r:CONNECTS]->(t2:Table)
                    RETURN t1.name as from_table, t2.name as to_table,
                           r.relationship_type as type, r.join_type as join_type,
                           r.join_sql as join_sql, r.from_column as from_column,
                           r.to_column as to_column, r.weight as weight,
                           r.extra_condition as extra_condition
                """)

            return {
                'tables': tables,
                'columns': all_columns,
                'relationships': rel_result or []
            }
        except Exception as e:
            logger.warning(f"从 Neo4j 获取 Schema 失败：{e}，回退到 MySQL")
            return self._get_schema_from_mysql(database_id)

    def _get_schema_from_mysql(self, database_id: Optional[int] = None) -> Dict:
        """从 MySQL 获取 Schema 信息"""
        if not self.mysql_client:
            return {'tables': [], 'columns': [], 'relationships': []}

        tables = self.mysql_client.get_table_schema()
        if database_id is not None and self.neo4j_client:
            try:
                result = self.neo4j_client.execute_query("""
                    MATCH (d:Database {database_id: $database_id})-[:HAS_TABLE]->(t:Table)
                    RETURN t.name as table_name
                """, {"database_id": database_id})
                table_names = [r["table_name"] for r in result]
                tables = [t for t in tables if t['table_name'] in table_names]
            except Exception as e:
                logger.warning(f"从 Neo4j 过滤表失败：{e}")

        all_columns = []
        for table in tables:
            columns = self.mysql_client.get_column_schema(table['table_name'])
            all_columns.extend(columns)

        return {'tables': tables, 'columns': all_columns, 'relationships': []}

    # ==================== 其他公共方法 ====================

    def execute_sql(self, sql: str, history_id: Optional[int] = None,
                    allowed_tables: Optional[List[str]] = None,
                    allowed_schemas: Optional[List[str]] = None,
                    user_query: Optional[str] = None,
                    semantic_version: Optional[str] = None,
                    session_id: Optional[str] = None,
                    content_sha256: Optional[str] = None) -> Dict[str, Any]:
        """执行 SQL"""
        if self.correctness_pipeline is not None:
            if not user_query or not semantic_version or not content_sha256:
                return self._governed_error_payload(
                    code=EXECUTION_CONTEXT_REQUIRED,
                    message='执行前需要原问题、语义版本和内容摘要',
                    semantic_version=semantic_version,
                    content_sha256=content_sha256,
                    sql=sql,
                ) | {'data': None}
            authorization = self.correctness_pipeline.authorize_execution(
                question=user_query,
                sql=sql,
                expected_semantic_version=semantic_version,
                expected_content_sha256=content_sha256,
                allowed_tables=tuple(allowed_tables or ()),
                allowed_schemas=tuple(allowed_schemas or ()),
            )
            if not authorization.executable:
                public = {
                    'status': 'error',
                    'error': (
                        authorization.validation.issues[0].message
                        if authorization.validation.issues
                        else 'SQL 执行授权失败'
                    ),
                    'data': None,
                    'sql': authorization.sql,
                    'semantic_version': authorization.semantic_version,
                    'content_sha256': authorization.content_sha256,
                    'validation': authorization.validation.model_dump(mode='json'),
                    'repair_attempts': authorization.repair_attempts,
                    'executable': False,
                }
                return public
            take_identity = getattr(
                self.correctness_pipeline,
                'take_authorization_history_identity',
                None,
            )
            candidate_identity = (
                take_identity(authorization) if callable(take_identity) else None
            )
            governance_identity = (
                candidate_identity
                if isinstance(candidate_identity, GovernedHistoryIdentity)
                else None
            )
        else:
            governance_identity = None

        if not self.mysql_client:
            return {'status': 'error', 'error': 'MySQL 客户端未连接', 'data': None}

        history: Mapping[str, Any] | None = None
        if history_id is not None:
            if self.correctness_pipeline is None and not session_id:
                logger.warning("history_update_skipped reason=missing_session")
            else:
                history, _history_identity, history_error = self._bound_history(
                    history_id,
                    session_id=session_id,
                    sql=sql,
                    user_query=user_query,
                    semantic_version=semantic_version,
                    content_sha256=content_sha256,
                    governance_identity=governance_identity,
                )
                if history_error is not None:
                    return self._history_context_error(
                        history_error,
                        sql=sql,
                        semantic_version=semantic_version,
                        content_sha256=content_sha256,
                    )

        logger.info("准备执行 SQL：sql_id={}", text_fingerprint(sql))

        try:
            safety_error = self._get_query_safety_error(sql, allowed_tables, allowed_schemas)
            if safety_error:
                if history_id is not None and history is not None:
                    self._update_bound_history(
                        history_id,
                        history,
                        execution_status='fail',
                        execution_error=safety_error,
                    )
                return {'status': 'error', 'error': safety_error, 'data': None}

            max_rows = int(self.config.get('sql_query_max_rows', 500))
            constrained_sql = constrain_read_query(
                sql,
                max_rows=max_rows,
                timeout_ms=int(self.config.get('sql_query_timeout_ms', 10000)),
                dialect=self.sql_chain.dialect,
            )
            results, truncated = self.mysql_client.execute_query_limited(
                constrained_sql, max_rows=max_rows
            )

            if history_id is not None and history is not None:
                updated = self._update_bound_history(
                    history_id,
                    history,
                    execution_status='success',
                    execution_result={
                        'row_count': len(results),
                        'data_preview': results[:5],
                        'truncated': truncated,
                    }
                )
                if not updated:
                    return {
                        'status': 'error',
                        'error': '执行记录状态更新失败',
                        'data': None,
                    }
                self._promote_history_to_few_shot(
                    history_id,
                    expected_history=history,
                    governance_identity=_history_identity,
                )

            return {
                'status': 'success', 'error': None,
                'data': results, 'row_count': len(results), 'truncated': truncated
            }
        except Exception as exc:
            error_id = log_safe_exception("sql_execute", exc, sql=sql)
            public_error = f"查询执行失败，请联系管理员并提供错误编号 {error_id}"
            if history_id is not None and history is not None:
                self._update_bound_history(
                    history_id,
                    history,
                    execution_status='fail',
                    execution_error=public_error,
                )
            return {'status': 'fail', 'error': public_error, 'data': None}

    def _is_safe_query(self, sql: str) -> bool:
        """检查 SQL 是否为单条、无副作用的只读查询。"""
        max_length = self.config.get('sql_max_length', 2000)
        if max_length and len(sql) > max_length:
            return False
        return is_safe_read_query(sql, dialect=self.sql_chain.dialect)

    def _get_query_safety_error(self, sql: str,
                                allowed_tables: Optional[List[str]],
                                allowed_schemas: Optional[List[str]]) -> Optional[str]:
        if not self._is_safe_query(sql):
            return '仅允许执行符合长度限制的单条只读 SELECT 查询'
        if not self.config.get('sql_scope_enforcement', True):
            return None
        if allowed_tables is None:
            schema_rows = self.mysql_client.get_table_schema()
            allowed_tables = [
                row.get('table_name') for row in schema_rows if row.get('table_name')
            ]
        if allowed_schemas is None:
            database = getattr(self.mysql_client, 'database', None)
            allowed_schemas = [database] if database else []
        scope = validate_query_scope(
            sql,
            allowed_tables=allowed_tables,
            allowed_schemas=allowed_schemas,
            blocked_tables=self.config.get('sql_blocked_tables'),
            dialect=self.sql_chain.dialect,
        )
        return '；'.join(scope['errors']) if not scope['valid'] else None

    def fix_and_retry(self, history_id: int, error: str) -> Dict[str, Any]:
        """修复 SQL 并重试"""
        if self.correctness_pipeline is not None:
            return {
                'status': 'error',
                'error': '语义治理模式下请重新生成 SQL',
                'data': None,
            }
        if not self.mysql_client:
            return {'status': 'error', 'error': 'MySQL 客户端未连接'}
        history = self.mysql_client.get_history_detail(history_id)
        if not history:
            return {'status': 'error', 'error': '历史记录不存在'}

        original_sql = history.get('generated_sql', '')
        schema_info = self._get_schema_info()

        fix_result = self.sql_chain.fix_sql(sql=original_sql, error=error, schema_info=schema_info)
        if fix_result['status'] != 'success':
            return fix_result

        self.mysql_client.execute_update(
            """UPDATE sql_generation_history
               SET generated_sql = :generated_sql, validation_status = 'pending'
               WHERE id = :history_id""",
            {"generated_sql": fix_result['sql'], "history_id": history_id},
            raise_on_error=True,
        )
        return self.execute_sql(fix_result['sql'], history_id)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'chain_stats': self.sql_chain.get_stats(),
            'retriever_stats': {
                'document_count': self.knowledge_retriever.retriever.get_document_count()
            }
        }
        if self.mysql_client:
            stats['history_stats'] = self.mysql_client.get_statistics()
        return stats

    def detect_dialect(self, user_query: str) -> Tuple[str, Optional[int]]:
        """检测用户查询的目标数据库方言"""
        if self.dialect_detector:
            return self.dialect_detector.detect_from_query(user_query)
        return self.config.get('sql_dialect', 'mysql'), None


def create_sql_generator(
    llm,
    neo4j_client,
    mysql_client,
    config: Optional[Dict] = None,
    embedding_factory: Optional = None,
    correctness_pipeline=None,
    history_proof_issuer: HistoryProofIssuer | None = None,
) -> SQLGenerator:
    """创建 SQL 生成器工厂函数"""
    config = config or {}

    # 1. 从 Neo4j 同步知识到检索器
    knowledge_nodes = neo4j_client.get_schema_metadata() if neo4j_client else {'tables': [], 'columns': []}

    # 2. 构建检索器
    knowledge_retriever = KnowledgeRetriever(config=config, embedding_factory=embedding_factory)

    documents = []
    metadata = []
    for table in knowledge_nodes.get('tables', []):
        doc = f"{table.get('name', '')} {table.get('description', '')}"
        if doc.strip():
            documents.append(doc)
            metadata.append({
                'node_type': 'table',
                'name': table.get('name', ''),
                'description': table.get('description', ''),
                'database_id': table.get('database_id'),
            })

    # 3. 创建 SQL 生成链
    sql_chain = SQLGenerationChain(
        llm=llm,
        dialect=config.get('sql_dialect', 'mysql'),
        retrieval_top_k=config.get('retrieval_top_k', 5),
        prompt_max_length=config.get('prompt_max_length', 4000),
    )

    # 4. 创建 SQL 生成器
    generator = SQLGenerator(
        llm=llm,
        knowledge_retriever=knowledge_retriever,
        sql_chain=sql_chain,
        mysql_client=mysql_client,
        config=config,
        correctness_pipeline=correctness_pipeline,
        history_proof_issuer=history_proof_issuer,
    )

    # 5. 构建知识库索引
    if documents:
        knowledge_retriever.retriever.index_documents(documents, metadata)

    # 6. 构建 Few-Shot 索引
    if mysql_client and config.get('few_shot_enabled', True):
        try:
            if correctness_pipeline is not None:
                history_records = []
                knowledge_retriever.build_few_shot_index([])
            else:
                history_records = mysql_client.get_successful_history_for_few_shot(
                    limit=100,
                )
                knowledge_retriever.build_few_shot_index(history_records)
            logger.info(f"Few-Shot 索引构建完成，共 {len(knowledge_retriever.history_records)} 条成功记录")
        except Exception as e:
            logger.warning(f"构建 Few-Shot 索引失败：{e}")

    return generator
