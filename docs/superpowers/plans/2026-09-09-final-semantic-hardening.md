# Final Semantic Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the final semantic compiler, term resolution, prompt budgeting, artifact identity, and Neo4j publication correctness gaps without changing unrelated runtime, history, or Golden behavior.

**Architecture:** Compile every user-authored reference into a validated canonical form before any write, then make Context and Validator consume one fail-closed term resolver. Preserve complete semantic dependency bundles at the prompt boundary and scope Neo4j identities/replacement to full datasource provenance inside the existing atomic importer.

**Tech Stack:** Python 3.11, Pydantic 2.5, PyYAML 6, sqlglot 21–30, Neo4j Python driver, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-09-text2sql-correctness-kernel-design.md` plus the final-review requirements dispatched for `codex/final-semantic`.

## Global Constraints

- Work only in `/Users/rgj/PycharmProjects/text2sql_agent/.worktrees/final-semantic` on `codex/final-semantic`.
- Use `/Users/rgj/PycharmProjects/text2sql_agent/.worktrees/text2sql-correctness-kernel/.venv/bin/python -m pytest` for tests.
- Every production change must have a focused failing regression test observed first.
- Semantic compilation and runtime validation fail closed whenever a reference or conditional term cannot be proven unique and valid.
- Existing runtime/history/Golden behavior stays unchanged except where a shared interface requires a compatibility update.
- Neo4j publication continues to use one data-write transaction.

---

### Task 1: Strict canonical compiler and immutable registry artifacts

**Files:**
- Modify: `core/semantic/models.py`
- Modify: `core/semantic/compiler.py`
- Modify: `core/semantic/registry.py`
- Modify: `tests/semantic/test_models.py`
- Modify: `tests/semantic/test_compiler.py`
- Modify: `tests/semantic/test_registry.py`
- Modify: `tests/evaluation/test_golden.py` only at the shared compiler call site if an explicit expected datasource is required.

**Interfaces:**
- Consumes: `SemanticCompiler.compile_file(source, output_root)`.
- Produces: backward-compatible `SemanticCompiler.compile_file(source, output_root, *, expected_datasource=None)` and canonical compiled references.

- [x] **Step 1: Write failing parameterized model/compiler tests**

Add literal fixtures that assert rejection of non-boolean `directly_queryable`, empty/control-character identifiers, malformed or multi-expression metrics, unknown fields in relationship/term predicates, invalid field roles, unknown/ambiguous term targets, datasource traversal/stem mismatches, and symlinked output paths. Add assertions that an unqualified unique policy field compiles to `entity.field` while case-insensitive ambiguous fields are rejected.

- [x] **Step 2: Run focused tests and record expected RED failures**

Run:

```bash
/Users/rgj/PycharmProjects/text2sql_agent/.worktrees/text2sql-correctness-kernel/.venv/bin/python -m pytest tests/semantic/test_models.py tests/semantic/test_compiler.py tests/semantic/test_registry.py -q
```

Expected: new cases fail because Pydantic coerces values, regex parsing accepts malformed SQL, references are not canonicalized, unsafe paths are writable, and versions overwrite in place.

- [x] **Step 3: Implement strict immutable source types and sqlglot compilation**

Use strict scalar validation while explicitly converting YAML lists to immutable tuples. Parse metric expressions and boolean predicates with `sqlglot.parse`, require exactly one AST, validate every `Column`, require allowed aggregate functions, and validate `allowed_dimensions`/`default_time_field` roles. Rebuild a normalized `SemanticModel` whose relationships, metric reference lists, policies, and term targets use canonical declared names.

- [x] **Step 4: Make filesystem validation precede every write**

Validate datasource syntax, source stem or explicit expected datasource, resolved containment, and every existing path component before creating directories. Reject symlinks. Pre-read manifest and an existing version artifact; return the existing artifact for equal content and reject unequal content without changing the snapshot, schema, or manifest.

- [x] **Step 5: Verify focused GREEN**

Run the same focused command and require zero failures.

---

### Task 2: Shared fail-closed term resolution and policy enforcement

**Files:**
- Create: `core/semantic/terms.py`
- Modify: `core/semantic/context.py`
- Modify: `core/semantic/validator.py`
- Modify: `core/semantic/__init__.py`
- Modify: `tests/semantic/test_context.py`
- Modify: `tests/semantic/test_validator.py`

**Interfaces:**
- Produces: `SemanticTermResolver.resolve(question, snapshot) -> ResolvedSemanticTerms` used by both Context and Validator.
- Consumes: canonical targets `metric:<name>`, `entity:<name>`, `field:<entity>.<field>`, and optional compiled boolean `when` predicates.

- [x] **Step 1: Write failing real-behavior tests**

Add a `成交额 -> gross_revenue` fixture. Assert Context includes the governed metric bundle and Validator rejects `COUNT(orders.id)` with `METRIC_MISMATCH`. Assert a matched term's `when` predicate must be present structurally in SQL, ambiguous duplicate term definitions fail closed, and an unqualified denied field is enforced after compilation.

- [x] **Step 2: Run focused tests and record expected RED failures**

Run:

```bash
/Users/rgj/PycharmProjects/text2sql_agent/.worktrees/text2sql-correctness-kernel/.venv/bin/python -m pytest tests/semantic/test_context.py tests/semantic/test_validator.py -q
```

Expected: term-derived metrics are absent and the mismatched aggregate/required condition is accepted.

- [x] **Step 3: Implement one resolver and consume it twice**

Normalize question and term text identically, group matches by normalized term, and reject multiple distinct definitions. Context turns resolved targets/conditions into mandatory items and complete metric bundles. Validator obtains matched metrics only from direct aliases plus resolved metric targets and compares every matched `when` predicate against alias-aware conjunctive AST filter signatures.

- [x] **Step 4: Verify focused GREEN**

Run the same command and require zero failures.

---

### Task 3: Preserve complete semantic bundles in generation and repair prompts

**Files:**
- Modify: `core/chain/prompts.py`
- Modify: `core/chain/sql_chain.py`
- Modify: `core/sql/correctness.py` only to pass the structured context through the shared boundary.
- Modify: `tests/sql/test_governed_prompts.py`
- Modify: `tests/sql/test_correctness.py`

**Interfaces:**
- Generation and governed repair accept `str | SemanticPromptBundle`; strings retain bounded legacy behavior, while the structured governed bundle is escaped but never internally truncated.

- [x] **Step 1: Write failing sync/async prompt-boundary tests**

Construct a semantic context whose required relationship/policy appears after the old 10% or repair 30% cutoff. Assert both synchronous and asynchronous generation/repair messages contain the final dependency and do not contain a truncation marker inside `<semantic_data>`. Add a too-small-slot case that raises a stable prompt budget error before invoking the LLM.

- [x] **Step 2: Run focused tests and record expected RED failures**

Run:

```bash
/Users/rgj/PycharmProjects/text2sql_agent/.worktrees/text2sql-correctness-kernel/.venv/bin/python -m pytest tests/sql/test_governed_prompts.py tests/sql/test_correctness.py -q
```

Expected: the tail relationship/policy is truncated by the second prompt budget.

- [x] **Step 3: Implement semantic-first budget allocation**

Escape the complete structured semantic rendering, reserve it before allocating user/schema/knowledge/few-shot or SQL/issues slots, reduce only those untrusted optional blocks, and raise if fixed prompt overhead plus the semantic bundle cannot fit. Keep sync and async chains on the same message builders.

- [x] **Step 4: Verify focused GREEN**

Run the same command and require zero failures.

---

### Task 4: Collision-free and replaceable Neo4j semantic projections

**Files:**
- Modify: `core/semantic/publisher.py`
- Modify: `core/knowledge/neo4j_client.py`
- Modify: `tests/semantic/test_publisher.py`
- Modify: `tests/test_regressions.py`

**Interfaces:**
- Semantic publisher emits full-length deterministic datasource identity and `datasource` provenance on nodes/edges.
- Existing `import_knowledge_graph(payload, overwrite=True)` remains the transaction boundary.

- [x] **Step 1: Write failing payload/import tests**

Use the known SHA-256-prefix collision pair and assert distinct full datasource identities. Assert overwrite cleanup is by semantic datasource provenance, so removed/renamed concepts disappear without touching another datasource. Assert connect installs Database, Table, Column, and BusinessConcept uniqueness constraints suitable for their actual keys.

- [x] **Step 2: Run focused tests and record expected RED failures**

Run:

```bash
/Users/rgj/PycharmProjects/text2sql_agent/.worktrees/text2sql-correctness-kernel/.venv/bin/python -m pytest tests/semantic/test_publisher.py tests/test_regressions.py -q
```

Expected: the pair collides, cleanup only sees new names, and no constraints are installed.

- [x] **Step 3: Implement provenance-scoped atomic replacement and constraints**

Use the complete SHA-256 hex digest for semantic `database_id`, attach datasource provenance, collect a datasource-scoped `DETACH DELETE` for old Git semantic concepts before upserts, and keep it in the existing write batch. Install idempotent DB constraints after connectivity succeeds and before the client is announced ready.

- [x] **Step 4: Verify focused GREEN**

Run the same command and require zero failures.

---

### Task 5: Compatibility matrix and final evidence

**Files:**
- Create: `/tmp/final-fix-semantic-report.md`
- Modify: only files required by failures attributable to Tasks 1–4.

- [x] **Step 1: Run semantic/prompt/Neo4j suites**

```bash
/Users/rgj/PycharmProjects/text2sql_agent/.worktrees/text2sql-correctness-kernel/.venv/bin/python -m pytest tests/semantic tests/sql/test_governed_prompts.py tests/sql/test_correctness.py tests/test_regressions.py -q
```

- [x] **Step 2: Run the full suite, Ruff, and compile checks**

```bash
/Users/rgj/PycharmProjects/text2sql_agent/.worktrees/text2sql-correctness-kernel/.venv/bin/python -m pytest -q
/Users/rgj/PycharmProjects/text2sql_agent/.worktrees/text2sql-correctness-kernel/.venv/bin/python -m ruff check .
/Users/rgj/PycharmProjects/text2sql_agent/.worktrees/text2sql-correctness-kernel/.venv/bin/python -m compileall -q core tests scripts
```

- [x] **Step 3: Verify sqlglot 21 and 30 compatibility**

Run the compiler/validator cases once with the supplied sqlglot 30 environment and once with a temporary isolated sqlglot 21 import path; do not change the project environment.

- [x] **Step 4: Inspect diff and write evidence report**

Record baseline, each focused RED failure reason, each GREEN command/count, compatibility evidence, exact shared-interface changes, commit IDs, and residual risks in `/tmp/final-fix-semantic-report.md`.

- [x] **Step 5: Commit coherent changes**

Commit compiler/term/prompt work and Neo4j work as independently reviewable commits after fresh verification.
