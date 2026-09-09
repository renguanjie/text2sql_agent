import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    StrictBool,
    StrictStr,
    StringConstraints,
    field_serializer,
    field_validator,
)


def _yaml_tuple(value: Any) -> Any:
    """Accept YAML arrays while retaining strict, deeply immutable tuple fields."""
    return tuple(value) if isinstance(value, list) else value


Identifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    ),
]
DatasourceIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    ),
]
NonEmptyText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]
StringTuple = Annotated[tuple[StrictStr, ...], BeforeValidator(_yaml_tuple)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        protected_namespaces=(),
    )


class SemanticField(FrozenModel):
    name: Identifier
    name_cn: NonEmptyText
    data_type: NonEmptyText
    role: Literal["identifier", "dimension", "measure", "time"]
    synonyms: StringTuple = ()
    enum_values: StringTuple = ()
    sensitivity: Literal["public", "internal", "sensitive", "restricted"] = "internal"
    directly_queryable: StrictBool = True


class SemanticEntity(FrozenModel):
    name: Identifier
    name_cn: NonEmptyText
    synonyms: StringTuple = ()
    fields: Annotated[tuple[SemanticField, ...], BeforeValidator(_yaml_tuple)]


class SemanticRelationship(FrozenModel):
    name: Identifier
    from_entity: Identifier
    from_field: Identifier
    to_entity: Identifier
    to_field: Identifier
    join_type: Literal["INNER JOIN", "LEFT JOIN"] = "LEFT JOIN"
    extra_condition: NonEmptyText | None = None


class SemanticMetric(FrozenModel):
    name: Identifier
    name_cn: NonEmptyText
    synonyms: StringTuple = ()
    expression: NonEmptyText
    allowed_dimensions: StringTuple = ()
    required_filters: StringTuple = ()
    default_time_field: NonEmptyText | None = None


class SemanticTerm(FrozenModel):
    term: NonEmptyText
    target: NonEmptyText
    when: NonEmptyText | None = None


class SemanticPolicies(FrozenModel):
    denied_entities: Annotated[tuple[Identifier, ...], BeforeValidator(_yaml_tuple)] = ()
    denied_fields: StringTuple = ()


class SemanticModel(FrozenModel):
    schema_version: Literal[1]
    model_version: StrictStr
    datasource: DatasourceIdentifier
    dialect: Literal["mysql"]
    entities: Annotated[tuple[SemanticEntity, ...], BeforeValidator(_yaml_tuple)]
    relationships: Annotated[
        tuple[SemanticRelationship, ...], BeforeValidator(_yaml_tuple)
    ] = ()
    metrics: Annotated[tuple[SemanticMetric, ...], BeforeValidator(_yaml_tuple)] = ()
    terms: Annotated[tuple[SemanticTerm, ...], BeforeValidator(_yaml_tuple)] = ()
    policies: SemanticPolicies = SemanticPolicies()

    @field_validator("model_version")
    @classmethod
    def validate_semver(cls, value: str) -> str:
        if not re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", value):
            raise ValueError("model_version 必须是 MAJOR.MINOR.PATCH")
        return value


class SemanticSnapshot(SemanticModel):
    source_sha256: StrictStr
    content_sha256: StrictStr
    compiler_version: StrictStr
    compiled_at: StrictStr


class SemanticContext(FrozenModel):
    datasource: DatasourceIdentifier
    semantic_version: StrictStr
    entities: StringTuple = ()
    fields: StringTuple = ()
    relationships: StringTuple = ()
    metrics: StringTuple = ()
    terms: StringTuple = ()
    policies: StringTuple = ()
    rendered: StrictStr = ""


class ValidationIssue(FrozenModel):
    code: StrictStr
    message: StrictStr
    objects: StringTuple = ()
    stage: Literal["syntax", "safety", "semantic", "preflight"] = "semantic"

    @field_serializer("objects")
    def serialize_objects(self, value: tuple[str, ...]) -> list[str]:
        return list(value)


class ValidationReport(FrozenModel):
    valid: StrictBool
    issues: Annotated[tuple[ValidationIssue, ...], BeforeValidator(_yaml_tuple)] = ()


class PreflightResult(FrozenModel):
    valid: StrictBool
    estimated_rows: int = 0
    issues: Annotated[tuple[ValidationIssue, ...], BeforeValidator(_yaml_tuple)] = ()


class CorrectnessResult(FrozenModel):
    status: Literal["success", "rejected", "error"]
    sql: str | None
    semantic_version: str | None
    validation: ValidationReport
    repair_attempts: Literal[0, 1]
    executable: StrictBool
