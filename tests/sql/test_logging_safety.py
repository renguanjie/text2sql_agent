from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import re
from unittest.mock import AsyncMock, Mock

from loguru import logger
import pytest


SECRET = "bearer-runtime-secret-8491"


class AuthenticationError(RuntimeError):
    pass


@contextmanager
def capture_loguru(caplog):
    caplog.set_level(logging.DEBUG)
    sink_id = logger.add(caplog.handler, format="{message}", level="DEBUG")
    try:
        yield
    finally:
        logger.remove(sink_id)


def _assert_safe_failure_log(text: str, stage: str) -> None:
    assert SECRET not in text
    assert "Authorization" not in text
    assert "Bearer" not in text
    assert "Traceback" not in text
    assert f"stage={stage}" in text
    assert "exception_type=AuthenticationError" in text
    assert re.search(r"error_id=[0-9a-f]{32}", text)
    assert re.search(r"fingerprint=[0-9a-f]{12}", text)


def test_sync_generation_exception_uses_only_safe_metadata(caplog):
    from core.chain.sql_chain import SQLGenerationChain

    llm = Mock()
    llm.invoke.side_effect = AuthenticationError(
        f"Authorization: Bearer {SECRET} SELECT private FROM hidden"
    )
    chain = SQLGenerationChain(llm)

    with capture_loguru(caplog):
        result = chain.generate("查询订单", {"tables": [], "columns": []})

    assert result == {
        "sql": None,
        "status": "error",
        "error": "LLM 服务调用失败，请检查模型配置后重试",
        "raw_response": None,
    }
    _assert_safe_failure_log(caplog.text, "sql_generation")


@pytest.mark.asyncio
async def test_async_generation_exception_uses_only_safe_metadata(caplog):
    from core.chain.sql_chain import SQLGenerationChain

    llm = Mock()
    llm.ainvoke = AsyncMock(
        side_effect=AuthenticationError(f"Authorization=Bearer {SECRET}")
    )
    chain = SQLGenerationChain(llm)

    with capture_loguru(caplog):
        result = await chain.generate_async(
            "查询订单", {"tables": [], "columns": []}
        )

    assert result["error"] == "LLM 服务调用失败，请检查模型配置后重试"
    _assert_safe_failure_log(caplog.text, "sql_generation_async")


@pytest.mark.parametrize("governed", [False, True], ids=["legacy", "governed"])
def test_sync_repair_exception_uses_only_safe_metadata(caplog, governed: bool):
    from core.chain.sql_chain import SQLGenerationChain

    llm = Mock()
    llm.invoke.side_effect = AuthenticationError(f"Bearer {SECRET}")
    chain = SQLGenerationChain(llm)

    with capture_loguru(caplog):
        if governed:
            with pytest.raises(AuthenticationError):
                chain.fix_sql_governed(
                    "SELECT missing FROM orders",
                    [{"code": "UNKNOWN_COLUMN"}],
                    "语义上下文",
                )
            stage = "sql_repair_governed"
        else:
            result = chain.fix_sql(
                "SELECT missing FROM orders", "字段不存在", {"tables": []}
            )
            assert result == {
                "sql": None,
                "status": "error",
                "error": "SQL 修复服务调用失败",
            }
            stage = "sql_repair"

    _assert_safe_failure_log(caplog.text, stage)


@pytest.mark.asyncio
@pytest.mark.parametrize("governed", [False, True], ids=["legacy", "governed"])
async def test_async_repair_exception_uses_only_safe_metadata(
    caplog, governed: bool
):
    from core.chain.sql_chain import SQLGenerationChain

    llm = Mock()
    llm.ainvoke = AsyncMock(side_effect=AuthenticationError(f"Bearer {SECRET}"))
    chain = SQLGenerationChain(llm)

    with capture_loguru(caplog):
        if governed:
            with pytest.raises(AuthenticationError):
                await chain.fix_sql_governed_async(
                    "SELECT missing FROM orders",
                    [{"code": "UNKNOWN_COLUMN"}],
                    "语义上下文",
                )
            stage = "sql_repair_governed_async"
        else:
            result = await chain.fix_sql_async(
                "SELECT missing FROM orders", "字段不存在", {"tables": []}
            )
            assert result["error"] == "SQL 修复服务调用失败"
            stage = "sql_repair_async"

    _assert_safe_failure_log(caplog.text, stage)


def test_safe_file_formatter_drops_exception_and_bearer_content(tmp_path: Path):
    from core.utils.log_sanitizer import safe_log_format

    path = tmp_path / "app.log"
    sink_id = logger.add(path, format=safe_log_format)
    try:
        try:
            raise AuthenticationError(f"Authorization: Bearer {SECRET}")
        except AuthenticationError as exc:
            logger.opt(exception=exc).error(
                "request failed Authorization: Bearer {}", SECRET
            )
    finally:
        logger.remove(sink_id)

    rendered = path.read_text(encoding="utf-8")
    assert SECRET not in rendered
    assert "Authorization" not in rendered
    assert "Bearer" not in rendered
    assert "Traceback" not in rendered


def test_safe_streamlit_sink_never_receives_bearer_content():
    from core.utils.log_sanitizer import make_safe_streamlit_sink

    rendered: list[str] = []
    sink = make_safe_streamlit_sink(rendered.append)
    sink(f"request failed Authorization: Bearer {SECRET}")

    assert len(rendered) == 1
    assert SECRET not in rendered[0]
    assert "Authorization" not in rendered[0]
    assert "Bearer" not in rendered[0]
