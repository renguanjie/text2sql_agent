import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from core.evaluation.golden import (
    CaseResult,
    EvaluationReport,
    GoldenCase,
    GoldenConfigurationError,
    GoldenEvaluator,
    GoldenGeneratedResult,
    GoldenResults,
    GoldenSuite,
    ModelParameter,
    RuleRate,
    get_git_commit,
    load_golden_results,
    load_golden_suite,
    parse_model_parameters,
)
from core.semantic import (
    SemanticEntity,
    SemanticField,
    SemanticMetric,
    SemanticPolicies,
    SemanticRelationship,
    SemanticSnapshot,
)


@pytest.fixture
def snapshot() -> SemanticSnapshot:
    return SemanticSnapshot(
        schema_version=1,
        model_version="1.0.0",
        datasource="demo",
        dialect="mysql",
        entities=(
            SemanticEntity(
                name="example_orders",
                name_cn="示例订单",
                fields=(
                    SemanticField(
                        name="id", name_cn="订单编号", data_type="bigint", role="identifier"
                    ),
                    SemanticField(
                        name="customer_id",
                        name_cn="客户编号",
                        data_type="bigint",
                        role="identifier",
                    ),
                    SemanticField(
                        name="created_at",
                        name_cn="创建时间",
                        data_type="datetime",
                        role="time",
                    ),
                    SemanticField(
                        name="amount", name_cn="金额", data_type="decimal", role="measure"
                    ),
                    SemanticField(
                        name="status", name_cn="状态", data_type="varchar", role="dimension"
                    ),
                ),
            ),
            SemanticEntity(
                name="example_customers",
                name_cn="示例客户",
                fields=(
                    SemanticField(
                        name="id", name_cn="客户编号", data_type="bigint", role="identifier"
                    ),
                    SemanticField(
                        name="segment",
                        name_cn="客户分组",
                        data_type="varchar",
                        role="dimension",
                    ),
                    SemanticField(
                        name="contact_token",
                        name_cn="联系标记",
                        data_type="varchar",
                        role="dimension",
                        sensitivity="restricted",
                        directly_queryable=False,
                    ),
                ),
            ),
        ),
        relationships=(
            SemanticRelationship(
                name="example_order_customer",
                from_entity="example_orders",
                from_field="customer_id",
                to_entity="example_customers",
                to_field="id",
            ),
        ),
        metrics=(
            SemanticMetric(
                name="example_order_amount",
                name_cn="示例订单金额",
                expression="SUM(example_orders.amount)",
                allowed_dimensions=("example_orders.created_at",),
                default_time_field="example_orders.created_at",
            ),
        ),
        policies=SemanticPolicies(denied_fields=("example_customers.contact_token",)),
        source_sha256="1" * 64,
        content_sha256="2" * 64,
        compiler_version="1",
        compiled_at="2026-09-09T00:00:00Z",
    )


@pytest.fixture
def evaluator(snapshot: SemanticSnapshot) -> GoldenEvaluator:
    return GoldenEvaluator(snapshot, model="local-demo", git_commit="a" * 40)


def _case(**updates) -> GoldenCase:
    values = {
        "id": "demo-001",
        "question": "查询示例订单",
        "datasource": "demo",
        "dialect": "mysql",
        "should_execute": True,
    }
    values.update(updates)
    return GoldenCase(**values)


def _result(sql: str | None, *, executable: bool = True, codes: tuple[str, ...] = ()) -> dict:
    return {
        "sql": sql,
        "status": "success" if executable else "rejected",
        "executable": executable,
        "validation": {
            "valid": executable,
            "issues": [{"code": code} for code in codes],
        },
    }


def test_contracts_are_deeply_immutable():
    """Break caught: allowing report metadata or case expectations to mutate after scoring."""
    case = _case(expected_tables=("example_orders",))
    result = CaseResult(
        id=case.id,
        passed=True,
        failed_assertions=(),
        error_codes=(),
        question_fingerprint="q",
        sql_fingerprint="s",
        elapsed_ms=0,
    )
    report = EvaluationReport(
        model="local-demo",
        model_parameters=(ModelParameter(name="temperature", value=0),),
        datasource="demo",
        dialect="mysql",
        semantic_version="1.0.0",
        git_commit="unknown",
        total=1,
        passed=1,
        failed=0,
        pass_rate=1.0,
        per_rule=(RuleRate(assertion="SHOULD_EXECUTE", total=1, passed=1, rate=1.0),),
        cases=(result,),
        elapsed_ms=0,
    )

    with pytest.raises(ValidationError):
        case.id = "changed"
    with pytest.raises(ValidationError):
        report.model = "changed"
    with pytest.raises(TypeError):
        report.model_parameters[0] = ModelParameter(name="seed", value=1)


def test_evaluator_accepts_alias_cte_and_structurally_equivalent_metric(evaluator):
    """Break caught: comparing SQL text or losing physical lineage through a derived source."""
    case = _case(
        question="按月统计示例订单金额",
        expected_tables=("example_orders",),
        expected_fields=("example_orders.amount", "example_orders.created_at"),
        expected_metrics=("example_order_amount",),
        expected_time_grain="month",
    )
    result = _result(
        "WITH base AS ("
        "SELECT created_at AS occurred_at, amount AS value FROM example_orders"
        ") SELECT DATE_FORMAT(occurred_at, '%Y-%m') AS month_key, SUM(value) "
        "FROM base GROUP BY DATE_FORMAT(occurred_at, '%Y-%m')"
    )

    assert evaluator.evaluate_case(case, result).passed


@pytest.mark.parametrize(
    "grain,expression",
    [
        ("day", "DATE(o.created_at)"),
        ("week", "DATE_TRUNC('WEEK', o.created_at)"),
        ("month", "DATE_FORMAT(o.created_at, '%Y-%m')"),
        ("quarter", "DATE_TRUNC('QUARTER', o.created_at)"),
        ("year", "YEAR(o.created_at)"),
    ],
)
def test_exact_time_grain_accepts_common_mysql_equivalents(evaluator, grain, expression):
    """Break caught: recognizing only DATE_FORMAT month or relying on raw function spelling."""
    case = _case(
        id=f"grain-{grain}",
        question=f"按{grain}统计示例订单金额",
        expected_metrics=("example_order_amount",),
        expected_time_grain=grain,
    )
    sql = (
        f"SELECT {expression}, SUM(o.amount) FROM example_orders o "
        f"GROUP BY {expression}"
    )

    assert evaluator.evaluate_case(case, _result(sql)).passed


@pytest.mark.parametrize(
    "requested,wrong_expression",
    [
        ("day", "WEEK(created_at)"),
        ("week", "DATE_FORMAT(created_at, '%Y-%m')"),
        ("month", "YEAR(created_at)"),
        ("quarter", "MONTH(created_at)"),
        ("year", "QUARTER(created_at)"),
    ],
)
def test_wrong_time_grain_fails_closed(evaluator, requested, wrong_expression):
    """Break caught: treating the presence of any time field as the requested granularity."""
    case = _case(
        id=f"wrong-{requested}",
        question="时间粒度回归",
        expected_time_grain=requested,
    )
    sql = (
        f"SELECT {wrong_expression}, COUNT(*) FROM example_orders "
        f"GROUP BY {wrong_expression}"
    )

    scored = evaluator.evaluate_case(case, _result(sql))

    assert not scored.passed
    assert scored.failed_assertions == ("EXPECTED_TIME_GRAIN",)


def test_plain_time_field_does_not_prove_a_requested_grain(evaluator):
    """Break caught: repeating Task 5's field-presence check instead of exact grain proof."""
    case = _case(expected_time_grain="month")
    sql = "SELECT created_at, COUNT(*) FROM example_orders GROUP BY created_at"

    scored = evaluator.evaluate_case(case, _result(sql))

    assert scored.failed_assertions == ("EXPECTED_TIME_GRAIN",)


@pytest.mark.parametrize(
    "expression",
    [
        "DATE_FORMAT(created_at, '%Y-%M')",
        "DATE_FORMAT(created_at, '%Y-%c')",
    ],
)
def test_month_grain_accepts_sqlglot_normalized_mysql_tokens(evaluator, expression):
    """Break caught: sqlglot translating MySQL month tokens before grain inspection."""
    case = _case(expected_time_grain="month")
    sql = f"SELECT {expression}, COUNT(*) FROM example_orders GROUP BY {expression}"

    assert evaluator.evaluate_case(case, _result(sql)).passed


@pytest.mark.parametrize(
    "grain,expression",
    [
        ("day", "DATE_FORMAT(created_at, '%Y-%m-%e')"),
        ("week", "DATE_FORMAT(created_at, '%x-%u')"),
    ],
)
def test_grain_tokens_normalized_to_dash_day_and_percent_w(
    evaluator, grain, expression
):
    """Break caught: version-specific normalized %-d/%G/%W tokens being missed."""
    case = _case(expected_time_grain=grain)
    sql = f"SELECT {expression}, COUNT(*) FROM example_orders GROUP BY {expression}"

    assert evaluator.evaluate_case(case, _result(sql)).passed


@pytest.mark.parametrize(
    "expression",
    [
        "DATE_FORMAT(created_at, '%Y-%m-%d')",
        "DATE_FORMAT(created_at, '%Y-%m-%D')",
        "DATE_FORMAT(created_at, '%Y-%m-%d %H')",
        "DATE_FORMAT(created_at, '%Y-%W')",
    ],
)
def test_month_grain_rejects_finer_or_non_month_date_format_tokens(
    evaluator, expression
):
    """Break caught: treating any year-and-month-looking format as exact month grain."""
    case = _case(expected_time_grain="month")
    sql = f"SELECT {expression}, COUNT(*) FROM example_orders GROUP BY {expression}"

    scored = evaluator.evaluate_case(case, _result(sql))

    assert scored.failed_assertions == ("EXPECTED_TIME_GRAIN",)


@pytest.mark.parametrize(
    "grain,components",
    [
        ("month", ("YEAR(created_at)", "MONTH(created_at)")),
        ("quarter", ("YEAR(created_at)", "QUARTER(created_at)")),
        ("week", ("YEARWEEK(created_at)",)),
        ("week", ("DATE_FORMAT(created_at, '%x-%v')",)),
    ],
)
def test_calendar_component_grains_require_complete_same_field_keys(
    evaluator, grain, components
):
    """Break caught: accepting a partial period key that merges different years."""
    case = _case(expected_time_grain=grain)
    expressions = ", ".join(components)
    sql = f"SELECT {expressions}, COUNT(*) FROM example_orders GROUP BY {expressions}"

    assert evaluator.evaluate_case(case, _result(sql)).passed


@pytest.mark.parametrize("grain,function", [("month", "MONTH"), ("quarter", "QUARTER"), ("week", "WEEK")])
def test_partial_calendar_component_is_not_an_exact_grain(evaluator, grain, function):
    """Break caught: month, quarter, or week number alone collapsing distinct years."""
    case = _case(expected_time_grain=grain)
    expression = f"{function}(created_at)"
    sql = f"SELECT {expression}, COUNT(*) FROM example_orders GROUP BY {expression}"

    assert evaluator.evaluate_case(case, _result(sql)).failed_assertions == (
        "EXPECTED_TIME_GRAIN",
    )


def test_calendar_components_from_different_time_fields_do_not_combine(snapshot):
    """Break caught: combining YEAR and MONTH facts without checking governed lineage."""
    second_time = SemanticField(
        name="updated_at",
        name_cn="更新时间",
        data_type="datetime",
        role="time",
    )
    orders = snapshot.entities[0].model_copy(
        update={"fields": (*snapshot.entities[0].fields, second_time)}
    )
    evaluator = GoldenEvaluator(
        snapshot.model_copy(update={"entities": (orders, snapshot.entities[1])})
    )
    case = _case(expected_time_grain="month")
    sql = (
        "SELECT YEAR(created_at), MONTH(updated_at), COUNT(*) FROM example_orders "
        "GROUP BY YEAR(created_at), MONTH(updated_at)"
    )

    assert evaluator.evaluate_case(case, _result(sql)).failed_assertions == (
        "EXPECTED_TIME_GRAIN",
    )


@pytest.mark.parametrize(
    "requested,components",
    [
        ("year", ("YEAR(created_at)", "MONTH(created_at)")),
        (
            "month",
            ("YEAR(created_at)", "MONTH(created_at)", "DAY(created_at)"),
        ),
    ],
)
def test_finer_component_key_does_not_also_claim_a_coarser_grain(
    evaluator, requested, components
):
    """Break caught: retaining every prefix grain instead of the exact final key."""
    case = _case(expected_time_grain=requested)
    expressions = ", ".join(components)
    sql = f"SELECT {expressions}, COUNT(*) FROM example_orders GROUP BY {expressions}"

    assert evaluator.evaluate_case(case, _result(sql)).failed_assertions == (
        "EXPECTED_TIME_GRAIN",
    )


def test_aggregated_year_component_cannot_complete_a_month_group_key(evaluator):
    """Break caught: counting temporal functions inside aggregates as grouping components."""
    case = _case(expected_time_grain="month")
    sql = (
        "SELECT MIN(YEAR(created_at)), MONTH(created_at), COUNT(*) "
        "FROM example_orders GROUP BY MONTH(created_at)"
    )

    assert evaluator.evaluate_case(case, _result(sql)).failed_assertions == (
        "EXPECTED_TIME_GRAIN",
    )


def test_unused_derived_grain_expression_cannot_satisfy_outer_result(evaluator):
    """Break caught: scanning every derived projection rather than used final lineage."""
    case = _case(expected_time_grain="month")
    sql = (
        "WITH base AS (SELECT DATE_FORMAT(created_at, '%Y-%m') AS month_key, "
        "status FROM example_orders) "
        "SELECT status, COUNT(*) FROM base GROUP BY status"
    )

    assert evaluator.evaluate_case(case, _result(sql)).failed_assertions == (
        "EXPECTED_TIME_GRAIN",
    )


def test_used_derived_grain_alias_satisfies_outer_result(evaluator):
    """Break caught: tightening derived scans without tracing referenced projection aliases."""
    case = _case(expected_time_grain="month")
    sql = (
        "WITH base AS (SELECT DATE_FORMAT(created_at, '%Y-%m') AS month_key "
        "FROM example_orders) "
        "SELECT month_key, COUNT(*) FROM base GROUP BY month_key"
    )

    assert evaluator.evaluate_case(case, _result(sql)).passed


def test_structural_assertions_cover_relationship_filter_and_forbidden_objects(evaluator):
    """Break caught: scoring joins or predicates from SQL substrings instead of resolved AST facts."""
    case = _case(
        expected_tables=("example_orders", "example_customers"),
        expected_fields=("example_orders.status", "example_customers.segment"),
        forbidden_fields=("example_customers.contact_token",),
        expected_relationships=(
            "example_orders.customer_id=example_customers.id",
        ),
        expected_filters=("example_orders.status = 'complete'",),
    )
    sql = (
        "SELECT c.segment FROM example_orders AS o "
        "JOIN example_customers AS c ON c.id = o.customer_id "
        "WHERE o.status = 'complete'"
    )

    assert evaluator.evaluate_case(case, _result(sql)).passed


def test_relationship_name_resolves_to_governed_endpoint_pair(evaluator):
    """Break caught: treating a relationship name as a textual SQL requirement."""
    case = _case(expected_relationships=("example_order_customer",))
    sql = (
        "SELECT c.segment FROM example_customers c "
        "JOIN example_orders o ON o.customer_id = c.id"
    )

    assert evaluator.evaluate_case(case, _result(sql)).passed


def test_wrong_join_pair_and_filter_operator_fail(evaluator):
    """Break caught: checking only joined tables and filter field presence."""
    case = _case(
        expected_relationships=("example_order_customer",),
        expected_filters=("example_orders.status = 'complete'",),
    )
    sql = (
        "SELECT c.segment FROM example_orders o "
        "JOIN example_customers c ON o.id = c.id "
        "WHERE o.status <> 'complete'"
    )

    scored = evaluator.evaluate_case(case, _result(sql))

    assert scored.failed_assertions == (
        "EXPECTED_FILTERS",
        "EXPECTED_RELATIONSHIPS",
    )


def test_metric_assertions_use_governed_expression_not_projection_alias(evaluator):
    """Break caught: accepting a metric because its expected name appears as an alias."""
    case = _case(expected_metrics=("example_order_amount",))
    sql = "SELECT COUNT(*) AS example_order_amount FROM example_orders"

    scored = evaluator.evaluate_case(case, _result(sql))

    assert scored.failed_assertions == ("EXPECTED_METRICS",)


def test_forbidden_metric_detects_semantically_equivalent_expression(evaluator):
    """Break caught: checking forbidden metrics only by metric name text."""
    case = _case(forbidden_metrics=("example_order_amount",))
    sql = "SELECT SUM(o.amount) AS harmless_name FROM example_orders o"

    scored = evaluator.evaluate_case(case, _result(sql))

    assert scored.failed_assertions == ("FORBIDDEN_METRICS",)


def test_expected_metric_fails_closed_when_sql_is_not_structurally_parseable(evaluator):
    """Break caught: inferring metric presence from the absence of a mismatch on syntax failure."""
    case = _case(expected_metrics=("example_order_amount",))

    with pytest.raises(GoldenConfigurationError) as caught:
        evaluator.evaluate_case(case, _result("SELECT FROM"))

    assert caught.value.code == "INVALID_RESULT"


def test_forbidden_metric_also_fails_closed_when_sql_cannot_be_analyzed(evaluator):
    """Break caught: treating an unparseable candidate as proof that a forbidden metric is absent."""
    case = _case(forbidden_metrics=("example_order_amount",))

    with pytest.raises(GoldenConfigurationError) as caught:
        evaluator.evaluate_case(case, _result("SELECT FROM"))

    assert caught.value.code == "INVALID_RESULT"


def test_unused_cte_cannot_satisfy_expected_table_or_field(evaluator):
    """Break caught: counting physical objects from a dead derived branch as returned lineage."""
    case = _case(
        expected_tables=("example_customers",),
        expected_fields=("example_customers.segment",),
    )
    sql = (
        "WITH unused AS (SELECT segment FROM example_customers) "
        "SELECT id FROM example_orders"
    )

    scored = evaluator.evaluate_case(case, _result(sql))

    assert scored.failed_assertions == ("EXPECTED_FIELDS", "EXPECTED_TABLES")


@pytest.mark.parametrize(
    "sql",
    [
        (
            "SELECT o.id FROM example_orders o WHERE EXISTS ("
            "SELECT 1 FROM example_customers c "
            "WHERE c.id = o.customer_id AND c.segment = 'internal')"
        ),
        (
            "SELECT (SELECT MAX(c.segment) FROM example_customers c "
            "WHERE c.id = o.customer_id) FROM example_orders o"
        ),
        (
            "SELECT o.id FROM example_orders o WHERE o.customer_id IN ("
            "SELECT c.id FROM example_customers c WHERE c.segment = 'internal')"
        ),
    ],
)
def test_forbidden_objects_in_active_subqueries_cannot_bypass_lineage(evaluator, sql):
    """Break caught: inspecting only FROM lineage while ignoring reachable subqueries."""
    case = _case(
        forbidden_tables=("example_customers",),
        forbidden_fields=("example_customers.segment",),
    )

    scored = evaluator.evaluate_case(case, _result(sql))

    assert scored.failed_assertions == ("FORBIDDEN_FIELDS", "FORBIDDEN_TABLES")


def test_dead_cte_stays_excluded_while_active_subqueries_are_traversed(evaluator):
    """Break caught: fixing subqueries by counting every traversed, including dead, CTE."""
    case = _case(
        forbidden_tables=("example_customers",),
        forbidden_fields=("example_customers.segment",),
    )
    sql = (
        "WITH dead AS (SELECT segment FROM example_customers) "
        "SELECT id FROM example_orders"
    )

    assert evaluator.evaluate_case(case, _result(sql)).passed


def test_forbidden_table_and_field_intersections_fail(evaluator):
    """Break caught: implementing expected subsets while omitting forbidden intersections."""
    case = _case(
        forbidden_tables=("example_customers",),
        forbidden_fields=("example_customers.segment",),
    )
    sql = "SELECT segment FROM example_customers"

    scored = evaluator.evaluate_case(case, _result(sql))

    assert scored.failed_assertions == ("FORBIDDEN_FIELDS", "FORBIDDEN_TABLES")


def test_filter_structure_resolves_a_derived_projection_alias(evaluator):
    """Break caught: comparing expected filters without tracing a CTE projection to its field."""
    case = _case(expected_filters=("example_orders.status = 'complete'",))
    sql = (
        "WITH base AS (SELECT status AS state FROM example_orders) "
        "SELECT state FROM base WHERE state = 'complete'"
    )

    assert evaluator.evaluate_case(case, _result(sql)).passed


def test_filter_equality_accepts_reversed_operands_after_alias_resolution(evaluator):
    """Break caught: treating commutative equality operand order as a semantic difference."""
    case = _case(expected_filters=("example_orders.status = 'complete'",))
    sql = "SELECT status FROM example_orders o WHERE 'complete' = o.status"

    assert evaluator.evaluate_case(case, _result(sql)).passed


def test_should_execute_and_expected_error_codes_are_scored(evaluator):
    """Break caught: ignoring the structured authorization outcome on rejected SQL."""
    case = _case(
        should_execute=False,
        expected_error_codes=("POLICY_DENIED",),
    )
    result = _result(
        "SELECT contact_token FROM example_customers",
        executable=False,
        codes=("POLICY_DENIED",),
    )

    assert evaluator.evaluate_case(case, result).passed


def test_forbidden_field_is_a_structural_failure_even_when_execution_is_rejected(evaluator):
    """Break caught: allowing a generated forbidden object because downstream execution rejected it."""
    case = _case(
        should_execute=False,
        forbidden_fields=("example_customers.contact_token",),
        expected_error_codes=("POLICY_DENIED",),
    )
    result = _result(
        "SELECT contact_token FROM example_customers",
        executable=False,
        codes=("POLICY_DENIED",),
    )

    scored = evaluator.evaluate_case(case, result)

    assert scored.failed_assertions == ("FORBIDDEN_FIELDS",)


def test_case_snapshot_mismatch_has_stable_configuration_code(evaluator):
    """Break caught: silently scoring a case against a different datasource snapshot."""
    case = _case(datasource="different")

    with pytest.raises(GoldenConfigurationError) as caught:
        evaluator.evaluate_case(case, _result("SELECT id FROM example_orders"))

    assert caught.value.code == "SNAPSHOT_MISMATCH"
    assert str(caught.value) == "Golden Set 配置无效（SNAPSHOT_MISMATCH）"


def test_report_omits_all_content_by_default(evaluator, tmp_path: Path):
    """Break caught: serializing question, SQL, literals, or raw issue details without opt-in."""
    marker = "private-marker-7d390"
    case = _case(
        id="privacy-001",
        question=f"查询联系标记 {marker}",
        should_execute=False,
        expected_error_codes=("POLICY_DENIED",),
    )
    result = _result(
        f"SELECT contact_token FROM example_customers WHERE contact_token='{marker}'",
        executable=False,
        codes=("POLICY_DENIED",),
    )
    result["validation"]["issues"][0]["message"] = f"raw exception {marker}"
    result["user_query"] = f"runtime question {marker}"
    result["error"] = f"runtime exception {marker}"
    report = evaluator.evaluate_results([(case, result)])

    path = evaluator.write_report(report, tmp_path / "report.json")
    content = path.read_text("utf-8")

    assert marker not in content
    assert "SELECT" not in content
    assert '"question"' not in content
    assert '"sql"' not in content
    assert "question_fingerprint" in content
    assert json.loads(content)["content_included"] is False


def test_include_content_is_explicit_and_visible(snapshot: SemanticSnapshot, tmp_path: Path):
    """Break caught: ignoring the debug opt-in or failing to label content-bearing reports."""
    evaluator = GoldenEvaluator(snapshot, model="local-demo", include_content=True)
    case = _case(question="本地调试问题")
    result = _result("SELECT id FROM example_orders")

    report = evaluator.evaluate_results([(case, result)])
    payload = json.loads(evaluator.write_report(report, tmp_path / "debug.json").read_text("utf-8"))

    assert payload["content_included"] is True
    assert payload["cases"][0]["question"] == "本地调试问题"
    assert payload["cases"][0]["sql"] == "SELECT id FROM example_orders"


def test_model_parameter_parser_allows_only_non_sensitive_names():
    """Break caught: copying arbitrary endpoints or credentials into evaluation reports."""
    assert parse_model_parameters(["temperature=0", "seed=17", "quantization=Q4_K_M"]) == (
        ModelParameter(name="quantization", value="Q4_K_M"),
        ModelParameter(name="seed", value=17),
        ModelParameter(name="temperature", value=0),
    )

    for raw in ("api_key=secret", "endpoint=https://invalid.example", "token=secret"):
        with pytest.raises(GoldenConfigurationError) as caught:
            parse_model_parameters([raw])
        assert caught.value.code == "MODEL_PARAMETER_DENIED"
        assert "secret" not in str(caught.value)

    with pytest.raises(ValidationError):
        ModelParameter(name="api_key", value="secret")

    with pytest.raises(GoldenConfigurationError) as nonfinite:
        parse_model_parameters(["temperature=NaN"])
    assert nonfinite.value.code == "INVALID_MODEL_PARAMETER"
    with pytest.raises(ValidationError):
        ModelParameter(name="temperature", value=float("nan"))


@pytest.mark.parametrize(
    "raw",
    [
        "temperature=true",
        "temperature=2.01",
        "top_p=0",
        "top_p=1.01",
        "top_k=1.5",
        "top_k=0",
        "max_tokens=false",
        "max_tokens=0",
        "context_length=10000001",
        "seed=-1",
        "repeat_penalty=0",
        "model_revision=https://invalid.example/revision",
        "model_revision=release?api_key=private",
        "model_revision=token=private",
        "model_revision=host.example",
        "quantization=custom format",
        "quantization=Q4_K_M?password=private",
    ],
)
def test_model_parameter_values_have_strict_safe_specs(raw):
    """Break caught: an allowlisted name becoming a carrier for secrets or invalid values."""
    with pytest.raises(GoldenConfigurationError) as caught:
        parse_model_parameters([raw])

    assert caught.value.code == "INVALID_MODEL_PARAMETER"
    assert "private" not in str(caught.value)


def test_model_parameter_direct_construction_uses_the_same_safe_specs():
    """Break caught: bypassing CLI parsing by constructing report parameters directly."""
    for name, value in (
        ("temperature", True),
        ("top_p", 2),
        ("seed", -1),
        ("model_revision", "secret=private"),
        ("quantization", "unknown"),
    ):
        with pytest.raises(ValidationError):
            ModelParameter(name=name, value=value)


def test_report_serializes_parameter_summary_as_a_sorted_mapping(snapshot: SemanticSnapshot):
    """Break caught: emitting an awkward or nondeterministically ordered parameter summary."""
    parameters = parse_model_parameters(["temperature=0", "seed=17"])
    evaluator = GoldenEvaluator(snapshot, model_parameters=parameters)

    report = evaluator.evaluate_results(
        [(_case(), _result("SELECT id FROM example_orders"))]
    )

    assert report.model_dump(mode="json")["model_parameters"] == {
        "seed": 17,
        "temperature": 0,
    }


def test_generated_result_rejects_contradictory_or_blank_success_payloads():
    """Break caught: accepting a claimed success that validation says is non-executable."""
    for sql, executable, valid in (("SELECT 1", False, False), ("   ", True, True)):
        with pytest.raises(ValidationError):
            GoldenGeneratedResult(
                id="strict-001",
                sql=sql,
                status="success",
                executable=executable,
                validation={"valid": valid, "issues": []},
            )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "sql": "SELECT id FROM example_orders",
            "status": "success",
            "executable": True,
            "validation": {"issues": []},
        },
        {
            "sql": "SELECT id FROM example_orders",
            "status": "success",
            "executable": True,
            "validation": {
                "valid": True,
                "issues": [{"code": "POLICY_DENIED"}],
            },
        },
        {
            "sql": None,
            "status": "error",
            "executable": False,
            "validation": {"valid": False, "issues": []},
        },
    ],
)
def test_generated_result_enforces_complete_outcome_invariants(payload):
    """Break caught: accepting incomplete success or unclassified failure payloads."""
    with pytest.raises(ValidationError):
        GoldenGeneratedResult(id="strict-outcome", **payload)


@pytest.mark.parametrize(
    "sql",
    ["SELECT FROM", "SELECT 1; SELECT 2", "DELETE FROM example_orders"],
)
def test_success_without_structural_expectations_still_requires_one_readonly_query(sql):
    """Break caught: skipping AST validation when a success case has no object assertions."""
    with pytest.raises(ValidationError):
        GoldenGeneratedResult(
            id="strict-query",
            sql=sql,
            status="success",
            executable=True,
            validation={"valid": True, "issues": []},
        )


@pytest.mark.parametrize(
    "code",
    ["GENERATION_FAILED", "HISTORY_CONTEXT_MISMATCH", "EXECUTION_CONTEXT_REQUIRED"],
)
def test_real_generator_failure_codes_are_accepted(code):
    """Break caught: evaluator error-code contracts drifting from public generator payloads."""
    result = GoldenGeneratedResult(
        id="generator-error",
        sql=None,
        status="error",
        executable=False,
        validation={"valid": False, "issues": [{"code": code}]},
    )

    assert result.validation.issues[0].code == code


def test_governed_generator_public_payload_can_be_loaded_without_unsafe_extras(
    tmp_path: Path,
):
    """Break caught: defining an outcome contract that rejects the generator's public shape."""
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasource": "demo",
                "dialect": "mysql",
                "semantic_version": "1.0.0",
                "results": [
                    {
                        "id": "generator-success",
                        "sql": "SELECT id FROM example_orders",
                        "status": "success",
                        "semantic_version": "1.0.0",
                        "validation": {"valid": True, "issues": []},
                        "repair_attempts": 0,
                        "executable": True,
                        "user_query": "查询示例订单",
                    },
                    {
                        "id": "generator-failure",
                        "sql": None,
                        "status": "error",
                        "semantic_version": "1.0.0",
                        "validation": {
                            "valid": False,
                            "issues": [{"code": "GENERATION_FAILED"}],
                        },
                        "repair_attempts": 0,
                        "executable": False,
                        "user_query": "查询示例订单",
                        "error": "稳定公开错误描述",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = load_golden_results(path)

    assert tuple(result.status for result in loaded.results) == ("success", "error")


def test_generator_and_evaluator_share_the_public_error_code_allowlist():
    """Break caught: maintaining an evaluator-only copy of generator error codes."""
    from core.sql.error_codes import PUBLIC_SQL_RESULT_ERROR_CODES

    assert {
        "GENERATION_FAILED",
        "HISTORY_CONTEXT_MISMATCH",
        "EXECUTION_CONTEXT_REQUIRED",
    }.issubset(PUBLIC_SQL_RESULT_ERROR_CODES)


def test_report_value_objects_reject_coercion_nonfinite_and_inconsistency():
    """Break caught: report JSON carrying coerced, NaN, or contradictory statistics."""
    with pytest.raises(ValidationError):
        RuleRate(assertion="SHOULD_EXECUTE", total=True, passed=1, rate=1.0)
    with pytest.raises(ValidationError):
        RuleRate(assertion="SHOULD_EXECUTE", total=1, passed=1, rate=float("nan"))
    with pytest.raises(ValidationError):
        RuleRate(assertion="SHOULD_EXECUTE", total=2, passed=1, rate=1.0)
    with pytest.raises(ValidationError):
        RuleRate(assertion="PRIVATE_MARKER", total=1, passed=1, rate=1.0)
    with pytest.raises(ValidationError):
        CaseResult(
            id="invalid-case",
            passed=True,
            failed_assertions=("SHOULD_EXECUTE",),
            error_codes=(),
            question_fingerprint="q",
            sql_fingerprint=None,
            elapsed_ms=0,
        )
    with pytest.raises(ValidationError):
        CaseResult(
            id="invalid-code",
            passed=True,
            failed_assertions=(),
            error_codes=("PASSWORD_PRIVATE_MARKER",),
            question_fingerprint="q",
            sql_fingerprint=None,
            elapsed_ms=0,
        )


def test_report_enforces_case_counts_rates_rules_and_content_flags():
    """Break caught: trusting caller-supplied report aggregates or privacy flags."""
    case = CaseResult(
        id="report-case",
        passed=True,
        failed_assertions=(),
        error_codes=(),
        question_fingerprint="q",
        sql_fingerprint="s",
        elapsed_ms=0,
    )
    base = {
        "model": "local-demo",
        "datasource": "demo",
        "dialect": "mysql",
        "semantic_version": "1.0.0",
        "git_commit": "unknown",
        "total": 1,
        "passed": 1,
        "failed": 0,
        "pass_rate": 1.0,
        "per_rule": (
            RuleRate(assertion="SHOULD_EXECUTE", total=1, passed=1, rate=1.0),
        ),
        "cases": (case,),
        "elapsed_ms": 0,
    }
    for update in (
        {"total": 2},
        {"passed": 0},
        {"failed": 1},
        {"pass_rate": 0.5},
        {"content_included": "false"},
        {"per_rule": ()},
        {"per_rule": (RuleRate(assertion="SHOULD_EXECUTE", total=2, passed=1, rate=0.5),)},
        {"per_rule": (RuleRate(assertion="SHOULD_EXECUTE", total=1, passed=0, rate=0.0),)},
        {
            "cases": (
                case.model_copy(update={"question": "private"}),
            ),
            "content_included": False,
        },
    ):
        with pytest.raises(ValidationError):
            EvaluationReport(**(base | update))


def test_writer_rejects_nonfinite_values_even_if_model_copy_bypasses_validation(
    evaluator, tmp_path: Path
):
    """Break caught: Python's JSON encoder emitting non-standard NaN tokens."""
    report = evaluator.evaluate_results(
        [(_case(), _result("SELECT id FROM example_orders"))]
    ).model_copy(update={"pass_rate": float("nan")})

    with pytest.raises(ValueError):
        evaluator.write_report(report, tmp_path / "nan.json")

    assert not (tmp_path / "nan.json").exists()


def test_report_json_is_canonical_and_written_via_atomic_replace(
    evaluator, tmp_path: Path, monkeypatch
):
    """Break caught: nondeterministic key ordering or replacing the destination non-atomically."""
    case = _case()
    report = evaluator.evaluate_results([(case, _result("SELECT id FROM example_orders"))])
    calls: list[tuple[Path, Path]] = []
    from core.evaluation import golden

    real_replace = golden.os.replace

    def recording_replace(source, destination):
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(golden.os, "replace", recording_replace)
    path = evaluator.write_report(report, tmp_path / "nested" / "report.json")
    first = path.read_bytes()
    evaluator.write_report(report, path)

    assert first == path.read_bytes()
    assert calls and all(source.parent == destination.parent for source, destination in calls)
    assert first.endswith(b"\n")
    assert list(json.loads(first)) == sorted(json.loads(first))


def test_default_writer_refuses_a_content_bearing_report(evaluator, snapshot, tmp_path: Path):
    """Break caught: bypassing the explicit content opt-in by passing a report from elsewhere."""
    debug = GoldenEvaluator(snapshot, include_content=True)
    report = debug.evaluate_results(
        [(_case(question="private-debug-content"), _result("SELECT id FROM example_orders"))]
    )

    with pytest.raises(GoldenConfigurationError) as caught:
        evaluator.write_report(report, tmp_path / "report.json")

    assert caught.value.code == "CONTENT_OPT_IN_REQUIRED"
    assert not (tmp_path / "report.json").exists()

    disguised = report.model_copy(update={"content_included": False})
    with pytest.raises(GoldenConfigurationError) as inconsistent:
        evaluator.write_report(disguised, tmp_path / "disguised.json")
    assert inconsistent.value.code == "REPORT_CONTENT_MISMATCH"
    assert not (tmp_path / "disguised.json").exists()


def test_elapsed_fields_do_not_affect_structural_report_assertions(evaluator):
    """Break caught: making reproducibility tests depend on wall-clock timing."""
    case = _case()
    first = evaluator.evaluate_results([(case, _result("SELECT id FROM example_orders"))])
    second = first.model_copy(
        update={
            "elapsed_ms": first.elapsed_ms + 9,
            "cases": (
                first.cases[0].model_copy(update={"elapsed_ms": first.cases[0].elapsed_ms + 3}),
            ),
        }
    )

    assert first.deterministic_dump() == second.deterministic_dump()


def test_suite_loader_is_strict_and_rejects_duplicate_ids(tmp_path: Path):
    """Break caught: accepting typoed keys, coerced booleans, or ambiguous duplicate result binding."""
    base_case = {
        "id": "duplicate",
        "question": "查询示例订单",
        "datasource": "demo",
        "dialect": "mysql",
        "should_execute": True,
    }
    path = tmp_path / "cases.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "datasource": "demo",
                "dialect": "mysql",
                "semantic_version": "1.0.0",
                "cases": [base_case, base_case],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(GoldenConfigurationError) as duplicate:
        load_golden_suite(path)
    assert duplicate.value.code == "DUPLICATE_CASE_ID"

    payload = yaml.safe_load(path.read_text("utf-8"))
    payload["cases"] = [{**base_case, "should_execute": "true", "typo": "ignored?"}]
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(GoldenConfigurationError) as invalid:
        load_golden_suite(path)
    assert invalid.value.code == "INVALID_CASE_FILE"

    payload["cases"] = [{**base_case, "id": ["not", "hashable"]}]
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(GoldenConfigurationError) as malformed_id:
        load_golden_suite(path)
    assert malformed_id.value.code == "INVALID_CASE_FILE"


def test_results_loader_rejects_duplicate_ids_and_unknown_fields(tmp_path: Path):
    """Break caught: non-deterministically choosing one of duplicate/malformed generated results."""
    item = {
        "id": "demo-001",
        "sql": "SELECT id FROM example_orders",
        "status": "success",
        "executable": True,
        "validation": {"valid": True, "issues": []},
    }
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasource": "demo",
                "dialect": "mysql",
                "semantic_version": "1.0.0",
                "results": [item, item],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GoldenConfigurationError) as duplicate:
        load_golden_results(path)
    assert duplicate.value.code == "DUPLICATE_RESULT_ID"

    payload = json.loads(path.read_text("utf-8"))
    payload["results"] = [{**item, "raw_exception": "private"}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GoldenConfigurationError) as invalid:
        load_golden_results(path)
    assert invalid.value.code == "INVALID_RESULTS_FILE"
    assert "private" not in str(invalid.value)

    payload["results"] = [
        {
            **item,
            "status": "rejected",
            "executable": False,
            "validation": {
                "valid": False,
                "issues": [{"code": "PASSWORD_PRIVATE_MARKER"}],
            },
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GoldenConfigurationError) as unknown_code:
        load_golden_results(path)
    assert unknown_code.value.code == "INVALID_RESULTS_FILE"
    assert "PRIVATE_MARKER" not in str(unknown_code.value)


def test_suite_and_results_validate_snapshot_identity(snapshot: SemanticSnapshot):
    """Break caught: evaluating artifacts from another semantic version or dialect."""
    suite = GoldenSuite(
        schema_version=1,
        datasource="demo",
        dialect="mysql",
        semantic_version="1.0.0",
        cases=(_case(),),
    )
    results = GoldenResults(
        schema_version=1,
        datasource="demo",
        dialect="mysql",
        semantic_version="1.0.1",
        results=(),
    )
    evaluator = GoldenEvaluator(snapshot)

    with pytest.raises(GoldenConfigurationError) as caught:
        evaluator.validate_inputs(suite, results)
    assert caught.value.code == "SNAPSHOT_MISMATCH"


@pytest.mark.parametrize(
    "case,code",
    [
        (_case(expected_tables=("missing_table",)), "UNKNOWN_TABLE_EXPECTATION"),
        (_case(expected_fields=("example_orders.missing_field",)), "UNKNOWN_FIELD_EXPECTATION"),
        (_case(expected_metrics=("missing_metric",)), "UNKNOWN_METRIC_EXPECTATION"),
        (
            _case(expected_relationships=("example_orders.id=example_customers.id",)),
            "UNKNOWN_RELATIONSHIP_EXPECTATION",
        ),
        (_case(expected_filters=("example_orders.status =",)), "INVALID_FILTER_EXPECTATION"),
        (
            _case(expected_filters=("example_orders.missing_field = 1",)),
            "INVALID_FILTER_EXPECTATION",
        ),
    ],
)
def test_case_expectations_are_validated_against_snapshot_before_scoring(
    evaluator, case, code
):
    """Break caught: letting malformed or cross-snapshot expectations hide behind absent SQL."""
    with pytest.raises(GoldenConfigurationError) as caught:
        evaluator.evaluate_case(case, _result(None, executable=False))

    assert caught.value.code == code


def test_git_commit_lookup_has_stable_fallback(monkeypatch, tmp_path: Path):
    """Break caught: surfacing subprocess errors or unstable empty Git metadata."""
    from core.evaluation import golden

    def unavailable(*args, **kwargs):
        raise OSError("private-path")

    monkeypatch.setattr(golden.subprocess, "run", unavailable)

    assert get_git_commit(tmp_path) == "unknown"


def test_cli_exit_codes_distinguish_failed_evaluation_from_configuration_error(
    snapshot: SemanticSnapshot, tmp_path: Path
):
    """Break caught: treating regression failures as malformed input or vice versa."""
    from core.semantic.compiler import SemanticCompiler
    from scripts.evaluate_golden import main

    source = tmp_path / "demo.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model_version": "1.0.0",
                "datasource": "demo",
                "dialect": "mysql",
                "entities": [
                    {
                        "name": entity.name,
                        "name_cn": entity.name_cn,
                        "fields": [field.model_dump(mode="json") for field in entity.fields],
                    }
                    for entity in snapshot.entities
                ],
                "relationships": [item.model_dump(mode="json") for item in snapshot.relationships],
                "metrics": [item.model_dump(mode="json") for item in snapshot.metrics],
                "policies": snapshot.policies.model_dump(mode="json"),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    compiled = tmp_path / "compiled"
    SemanticCompiler(clock=lambda: "2026-09-09T00:00:00Z").compile_file(source, compiled)

    suite_path = tmp_path / "cases.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "datasource": "demo",
                "dialect": "mysql",
                "semantic_version": "1.0.0",
                "cases": [_case(expected_tables=("example_orders",)).model_dump(mode="json")],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasource": "demo",
                "dialect": "mysql",
                "semantic_version": "1.0.0",
                "results": [
                    {
                        "id": "demo-001",
                        **_result("SELECT id FROM example_customers"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    failed = main(
        [
            "--cases",
            str(suite_path),
            "--results",
            str(result_path),
            "--output",
            str(output),
            "--model",
            "local-demo",
            "--semantic-version",
            "1.0.0",
            "--compiled-root",
            str(compiled),
        ]
    )
    malformed = main(
        [
            "--cases",
            str(suite_path),
            "--results",
            str(result_path),
            "--output",
            str(output),
            "--model",
            "local-demo",
            "--semantic-version",
            "9.9.9",
            "--compiled-root",
            str(compiled),
        ]
    )

    assert failed == 1
    assert malformed == 2
    assert json.loads(output.read_text("utf-8"))["failed"] == 1


def test_repository_golden_suite_has_required_anonymous_coverage():
    """Break caught: shipping too few cases or omitting a required structural rule family."""
    path = Path(__file__).resolve().parents[2] / "eval/golden/default.yaml"

    suite = load_golden_suite(path)
    serialized = json.dumps(suite.model_dump(mode="json"), ensure_ascii=False)

    assert len(suite.cases) >= 6
    assert any(case.expected_metrics for case in suite.cases)
    assert any(case.expected_time_grain == "month" for case in suite.cases)
    assert any(case.expected_relationships for case in suite.cases)
    assert any(case.forbidden_fields for case in suite.cases)
    assert any("POLICY_DENIED" in case.expected_error_codes for case in suite.cases)
    assert any("UNKNOWN_COLUMN" in case.expected_error_codes for case in suite.cases)
    assert not re.search(r"\b1[3-9]\d{9}\b", serialized)
    assert not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", serialized)
    assert "password" not in serialized.casefold()


def test_repository_semantic_content_and_golden_suite_use_new_version(tmp_path: Path):
    """Break caught: publishing changed semantic content under an already compiled version."""
    from core.semantic.compiler import SemanticCompiler
    from core.semantic.registry import SemanticRegistry

    root = Path(__file__).resolve().parents[2]
    semantic_payload = yaml.safe_load(
        (root / "semantic_models/default.yaml").read_text("utf-8")
    )
    suite = load_golden_suite(root / "eval/golden/default.yaml")

    assert semantic_payload["model_version"] == "1.1.0"
    assert suite.semantic_version == "1.1.0"

    old_source = tmp_path / "old-default.yaml"
    old_source.write_text(
        yaml.safe_dump(
            semantic_payload | {"model_version": "1.0.0"},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    compiled = tmp_path / "compiled"
    compiler = SemanticCompiler(clock=lambda: "2026-09-09T00:00:00Z")
    compiler.compile_file(old_source, compiled)
    compiler.compile_file(root / "semantic_models/default.yaml", compiled)

    manifest = json.loads((compiled / "manifest.json").read_text("utf-8"))
    active = SemanticRegistry(compiled, "default").load_active()
    assert (compiled / "default/1.0.0.json").exists()
    assert manifest["active"]["default"] == "1.1.0"
    assert active.model_version == "1.1.0"
