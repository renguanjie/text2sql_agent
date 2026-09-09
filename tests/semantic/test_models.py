from copy import deepcopy

from pydantic import ValidationError
import pytest

from core.semantic.models import SemanticModel, ValidationIssue


def minimal_model() -> dict:
    return {
        "schema_version": 1,
        "model_version": "1.0.0",
        "datasource": "default",
        "dialect": "mysql",
        "entities": [{
            "name": "orders",
            "name_cn": "订单",
            "fields": [{
                "name": "amount",
                "name_cn": "金额",
                "data_type": "decimal",
                "role": "measure",
            }],
        }],
    }


def test_semantic_model_is_immutable_and_rejects_bad_version():
    model = SemanticModel.model_validate(minimal_model())
    with pytest.raises(ValidationError):
        model.model_version = "2.0.0"
    payload = minimal_model()
    payload["model_version"] = "release-one"
    with pytest.raises(ValidationError):
        SemanticModel.model_validate(payload)


def test_validation_issue_has_stable_public_shape():
    issue = ValidationIssue(code="UNKNOWN_COLUMN", message="字段不存在", objects=["orders.foo"])
    assert issue.model_dump() == {
        "code": "UNKNOWN_COLUMN",
        "message": "字段不存在",
        "objects": ["orders.foo"],
        "stage": "semantic",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("entities", 0, "fields", 0, "directly_queryable"), "false"),
    ],
)
def test_semantic_model_rejects_scalar_type_coercion(path, value):
    """Break caught: strings and booleans being coerced into security-relevant types."""
    payload = deepcopy(minimal_model())
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        SemanticModel.model_validate(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("datasource",), ""),
        (("datasource",), "../escaped"),
        (("datasource",), "sales\nshadow"),
        (("entities", 0, "name"), ""),
        (("entities", 0, "name"), "orders\u0000shadow"),
        (("entities", 0, "fields", 0, "name"), "amount\nMATCH"),
    ],
)
def test_semantic_model_rejects_empty_or_unsafe_identifiers(path, value):
    """Break caught: unsafe names reaching filesystem paths, SQL, or graph identifiers."""
    payload = deepcopy(minimal_model())
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        SemanticModel.model_validate(payload)
