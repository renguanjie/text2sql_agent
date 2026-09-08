import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        protected_namespaces=(),
    )


class SemanticField(FrozenModel):
    name: str
    name_cn: str
    data_type: str
    role: Literal["identifier", "dimension", "measure", "time"]
    synonyms: tuple[str, ...] = ()
    enum_values: tuple[str, ...] = ()
    sensitivity: Literal["public", "internal", "sensitive", "restricted"] = "internal"
    directly_queryable: bool = True


class SemanticEntity(FrozenModel):
    name: str
    name_cn: str
    synonyms: tuple[str, ...] = ()
    fields: tuple[SemanticField, ...]


class SemanticRelationship(FrozenModel):
    name: str
    from_entity: str
    from_field: str
    to_entity: str
    to_field: str
    join_type: Literal["INNER JOIN", "LEFT JOIN"] = "LEFT JOIN"
    extra_condition: str | None = None


class SemanticMetric(FrozenModel):
    name: str
    name_cn: str
    synonyms: tuple[str, ...] = ()
    expression: str
    allowed_dimensions: tuple[str, ...] = ()
    required_filters: tuple[str, ...] = ()
    default_time_field: str | None = None


class SemanticTerm(FrozenModel):
    term: str
    target: str
    when: str | None = None


class SemanticPolicies(FrozenModel):
    denied_entities: tuple[str, ...] = ()
    denied_fields: tuple[str, ...] = ()


class SemanticModel(FrozenModel):
    schema_version: Literal[1]
    model_version: str
    datasource: str
    dialect: Literal["mysql"]
    entities: tuple[SemanticEntity, ...]
    relationships: tuple[SemanticRelationship, ...] = ()
    metrics: tuple[SemanticMetric, ...] = ()
    terms: tuple[SemanticTerm, ...] = ()
    policies: SemanticPolicies = SemanticPolicies()

    @field_validator("model_version")
    @classmethod
    def validate_semver(cls, value: str) -> str:
        if not re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", value):
            raise ValueError("model_version 必须是 MAJOR.MINOR.PATCH")
        return value


class SemanticSnapshot(SemanticModel):
    source_sha256: str
    content_sha256: str
    compiler_version: str
    compiled_at: str


class SemanticContext(FrozenModel):
    datasource: str
    semantic_version: str
    entities: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    relationships: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    policies: tuple[str, ...] = ()
    rendered: str = ""


class ValidationIssue(FrozenModel):
    code: str
    message: str
    objects: tuple[str, ...] = ()
    stage: Literal["syntax", "safety", "semantic", "preflight"] = "semantic"

    @field_serializer("objects")
    def serialize_objects(self, value: tuple[str, ...]) -> list[str]:
        return list(value)


class ValidationReport(FrozenModel):
    valid: bool
    issues: tuple[ValidationIssue, ...] = ()


class PreflightResult(FrozenModel):
    valid: bool
    estimated_rows: int = 0
    issues: tuple[ValidationIssue, ...] = ()


class CorrectnessResult(FrozenModel):
    status: Literal["success", "rejected", "error"]
    sql: str | None
    semantic_version: str | None
    validation: ValidationReport
    repair_attempts: Literal[0, 1]
    executable: bool
