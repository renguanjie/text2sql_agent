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
