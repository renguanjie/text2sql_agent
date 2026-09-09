# 语义治理运维手册

正确性内核以 Git 中的 YAML 为唯一人工语义真相源。它为 SQL 生成提供紧凑中文上下文，并在生成与执行前执行确定性校验。Neo4j 仅是已发布快照的派生检索索引：图不可用会影响召回，但不能替代、修改或覆盖 Git 语义，更不能授权执行。

以下命令均假定在项目根目录、使用 README 推荐的已激活 `.venv` 执行：

```bash
source .venv/bin/activate

# Windows PowerShell: .\.venv\Scripts\Activate.ps1
```

因此手册统一使用 `python`、`pytest`、`ruff`，而不混用宿主解释器或硬编码虚拟环境路径。首次准备环境时，请按 README 使用 `python3.11 -m venv .venv`（或 `uv venv --python 3.11 --seed .venv`）、激活环境、`python -m pip install -r requirements.txt`，并在执行发布门禁前运行 `python -m pip install ruff==0.15.20`。该版本与仓库已验收的规则集一致；不固定版本会随 Ruff 新规则改变门禁结果。`uv` 创建的未 seed 环境没有 `pip`；应重建为上述 `--seed` 形式或用 `uv pip` 安装，而不要退回宿主 Python。

## 1. 默认值与边界

以下配置来自 `config.py` 与 `env.example`；默认均兼容旧行为。

| 变量 | 默认值 | 约束 |
| --- | --- | --- |
| `SEMANTIC_GOVERNANCE_ENABLED` | `false` | 仅设为 `true` 才创建语义注册表与正确性管线。 |
| `SEMANTIC_MODEL_DIR` | `semantic_models` | 相对路径相对于项目根目录；源文件为 `<目录>/<datasource>.yaml`。 |
| `SEMANTIC_DATASOURCE` | `default` | 必须对应一个 YAML 数据源。 |
| `SEMANTIC_CONTEXT_MAX_CHARS` | `2400` | 必须大于 `0`。 |
| `SEMANTIC_CONTEXT_MAX_ITEMS` | `24` | 必须大于 `0`。 |
| `SQL_REPAIR_MAX_ATTEMPTS` | `1` | 只允许 `0` 或 `1`；`1` 是最大值。 |
| `SQL_EXPLAIN_ENABLED` | `false` | 设为 `true` 时必须同时启用 `SEMANTIC_GOVERNANCE_ENABLED=true`。 |
| `SQL_EXPLAIN_TIMEOUT_MS` | `3000` | 必须大于 `0`，单位毫秒。 |
| `SQL_EXPLAIN_MAX_ESTIMATED_ROWS` | `0` | 不得为负；`0` 表示不因估算扫描行数拒绝。 |
| `SQL_MAX_LENGTH` | `2000` | 必须是 `1`–`100000` 的整数。 |
| `SQL_QUERY_MAX_ROWS` | `500` | 必须是 `1`–`100000` 的整数。 |
| `SQL_QUERY_TIMEOUT_MS` | `10000` | 必须是 `1`–`3600000` 的整数，单位毫秒。 |
| `PROMPT_MAX_LENGTH` | `4000` | 必须是 `1`–`100000` 的整数。 |
| `MAX_PREPARED_REQUESTS` | `256` | 治理 capability 容量，必须是 `1`–`4096` 的整数。 |
| `PREPARED_REQUEST_TTL_SECONDS` | `120` | 治理 capability TTL，必须大于 `0` 且不超过 `3600` 秒。 |
| `GOVERNED_HISTORY_BINDING_TTL_SECONDS` | `900` | 治理历史 binding TTL，必须大于 `0` 且不超过 `3600` 秒。 |
| `HISTORY_MAX_RECORDS` | `1000` | 必须是 `1`–`10000` 的整数；也限制进程内治理历史 binding 容量。 |

所有布尔环境变量只接受不区分大小写的 `true` 或 `false`（允许首尾空白）；
`1`、`0`、`yes`、`on` 及拼写错误都会让启动或热重载失败。错误只指出变量名，
不会回显环境变量值。语义治理或 `EXPLAIN` 任一开启时，`SQL_DIALECT` 必须为
`mysql`；二者都关闭时仍可使用其他受旧链路支持的方言，例如 `postgres`。

首版只接受 `schema_version: 1`、`dialect: mysql`，且 `model_version` 必须是严格的 `MAJOR.MINOR.PATCH`。模型字段角色限定为 `identifier`、`dimension`、`measure`、`time`。治理源、编译快照、Golden 用例都不得包含凭据或真实业务值。

关系的 `join_type` 是执行授权契约，只允许规范形式 `INNER JOIN`、`LEFT JOIN`、
`RIGHT JOIN`。裸 `JOIN` 规范为 `INNER JOIN`；`LEFT OUTER JOIN` 规范为
`LEFT JOIN`。`LEFT`/`RIGHT` 还固定 `from_entity` 与 `to_entity` 的左右位置，不能用
相反方向或其他 JOIN 种类替代。`CROSS JOIN`、`FULL JOIN`、`NATURAL JOIN` 不能声明为
语义关系，也不能通过添加 `ON` 条件冒充已治理关系。

关系 `extra_condition` 只允许字段/字面量的 `=`、字段 `IS NULL`、字段
`IS NOT NULL`，以及这些原子的 `AND` 合取；函数、子查询、`OR`、非确定性表达式和
其他运算符均在编译时拒绝。编译后的完整条件会进入语义上下文，并且每个原子必须以
AST 等价形式出现在当前活动 JOIN 的 `ON` 中；注释、文本命中或 `WHERE` 条件不能替代。

## 2. 作者 → 校验/编译 → 激活

1. 在 Git 分支中编辑 `semantic_models/default.yaml`（或你的 `<datasource>.yaml`）。YAML 是唯一人工源文件；请评审实体、字段、关系、指标、术语和策略。
2. 编译并执行交叉引用校验：

   ```bash
   python scripts/compile_semantic_models.py semantic_models/default.yaml
   ```

   成功输出的格式为 `datasource@model_version content_sha256`。编译器写入 `semantic_models/.compiled/<datasource>/<model_version>.json`，并原子更新 `semantic_models/.compiled/manifest.json` 中的 `active` 映射。激活版本由该清单决定，不按文件时间推断。
3. 检查已激活版本与摘要：

   ```bash
   python -m json.tool semantic_models/.compiled/manifest.json
   ```

   `content_sha256` 标识规范化语义内容，`source_sha256` 标识 YAML 原始字节。已存在的
   `<datasource>@<model_version>` 只有在源字节完全相同时才是幂等重编译；空白、换行或
   注释变化即使不改变 `content_sha256`，也必须先升级 `model_version`。否则编译器以
   `已由不同源字节编译；请升级 model_version` 拒绝，并保持 artifact、manifest 和
   schema 不变。`compiled_at` 不参与内容摘要，完全相同的源字节会返回已有 artifact，
   不刷新 provenance。若编译失败，不会生成部分快照或替换已加载的有效快照。

`semantic_models/default.yaml`、`semantic_models/schema/semantic-model.schema.json` 和 `eval/golden/*.yaml` 是受版本控制的源资产；`.compiled/` 是运行产物，必须保持忽略。

## 3. 显式发布到 Neo4j

完成编译与清单检查后，才可将**已激活的编译快照**发布为图检索索引：

```bash
python scripts/publish_semantic_model.py default \
  --compiled-root semantic_models/.compiled
```

命令仅接受编译目录和数据源，不能以原始 YAML 绕过编译。发布是显式运维动作，不会在应用启动时自动执行；它委托现有图导入的原子覆盖路径。同步 driver 缺失或已关闭时，原子导入明确返回 `Neo4j 未连接，无法原子导入知识图谱`；semantic publisher 也会拒绝零节点写入，不能宣称派生索引已更新。连接正常时，通用 v2 importer 的合法空 payload 仍可返回零计数成功；semantic publisher 的 payload 至少包含 Database 节点，因此不属于该例外。不要在 Neo4j 中手工编辑图来“修正”语义：下一次发布会由 Git 快照重建索引，SQL 校验始终读取注册表中的 Git 编译快照。

## 4. 启动与热重载

默认关闭示例：

```bash
SEMANTIC_GOVERNANCE_ENABLED=false SQL_EXPLAIN_ENABLED=false python -m streamlit run app.py
```

启用语义治理、不启用数据库预检：

```bash
SEMANTIC_GOVERNANCE_ENABLED=true \
SEMANTIC_MODEL_DIR=semantic_models \
SEMANTIC_DATASOURCE=default \
SQL_REPAIR_MAX_ATTEMPTS=1 \
SQL_EXPLAIN_ENABLED=false \
python -m streamlit run app.py
```

启用 MySQL `EXPLAIN` 预检时，使用独立、短生命周期的连接，并同时施加数据库侧提示与
进程侧墙钟截止时间；它必须依赖已启用的治理：

```bash
SEMANTIC_GOVERNANCE_ENABLED=true \
SQL_EXPLAIN_ENABLED=true \
SQL_EXPLAIN_TIMEOUT_MS=3000 \
SQL_EXPLAIN_MAX_ESTIMATED_ROWS=100000 \
python -m streamlit run app.py
```

现有进程可由拥有 `ApplicationContext` 的运维代码调用 `reload_governance(...)`。它只接受语义开关、模型目录/数据源、语义上下文预算、一次修复、EXPLAIN、治理 SQL 方言与限制、以及治理 capability/history binding 限制；会先候选构建（编译、加载、校验、新检索器/生成器），成功后一次替换对象图；失败时旧的不可变对象图继续服务。`sql_dialect` 在该热重载中会一并重建 SQL chain。

LLM 的 provider/model/key/base URL 必须使用 `reload_llm(...)`；Embedding 的 provider/model/key/base/dimension 必须使用 `reload_embedding(...)`。检索预算、连接、普通运行时开关等其他配置不能通过 `reload_governance(...)` 部分发布，必须完整重启应用。

生成阶段的 prepared capability 默认最多保留 256 个、120 秒，并在消费或放弃时
原子删除；达到上限会稳定地拒绝新请求。受治理历史 binding 默认 900 秒过期，
容量不超过 `HISTORY_MAX_RECORDS`（并有 4096 的进程内硬上限），只保存带进程随机密钥
的会话、问题和规范化 SQL 摘要，不保留这些字段的明文。binding 在一次匹配的执行尝试
开始时即消费；业务数据库失败或历史状态 CAS 失败也不能用同一 history ID 重试。
数据库历史本身按 `HISTORY_MAX_RECORDS`（1–10000，默认 1000）持续裁剪。

受治理历史写入另有一对仅存在于当前进程内的 HMAC issuer/verifier。公共
`save_sql_history(...)` 始终写入 `governance_validated=false` 且治理身份为 `NULL`；
公共 legacy 执行更新也只允许命中未治理行。治理生成器使用一次性、短时 proof 写入
完整审计身份，但该行仍保持未治理；只有真实业务查询成功后，proof-backed
`pending → success` CAS 才会同时置为 `governance_validated=true`。无需部署或迁移持久
密钥，已有治理身份列保持不变。

数据库中的治理字段只用于审计与 CAS 谓词，不是运行时证明。启用治理时，启动、
治理重载和 embedding 重载均不会从持久 SQL 历史恢复 Few-Shot；索引从空集开始，
只接收当前进程中“重新授权成功 + 真实执行成功 + CAS 成功”的即时晋升。其安全成本是
冷启动或上述重载会丢失此前 Few-Shot 学习，需要新的成功执行重新积累；关闭治理时仍
保留 legacy 历史加载兼容性。

```python
context.reload_governance(
    semantic_governance_enabled=True,
    semantic_model_dir="semantic_models",
    semantic_datasource="default",
    semantic_context_max_chars=2400,
    semantic_context_max_items=24,
    sql_repair_max_attempts=1,
    sql_explain_enabled=False,
    sql_explain_timeout_ms=3000,
    sql_explain_max_estimated_rows=0,
    max_prepared_requests=256,
    prepared_request_ttl_seconds=120,
    governed_history_binding_ttl_seconds=900,
)
```

没有持有该上下文的部署应重启应用；启动同样会先编译、加载并验证当前数据源。首次启动的治理配置无效会失败，而不是静默降级。

## 5. 生成、一次纠错与执行再授权

一次受治理请求固定执行以下流程：以单一不可变快照构建受限中文语义上下文 → 生成一条 SQL → 语法/只读安全/查询范围/语义校验 →（若开启）MySQL `EXPLAIN` → 必要时将脱敏的结构化错误和最小语义片段交给模型纠错一次 → 从第一项校验重新验证修复 SQL。`repair_attempts` 只会是 `0` 或 `1`；第二次失败不会再次调用模型，且 `executable` 为 `false`。

只有全部阶段通过时结果才是 `status: "success"`、`executable: true`。公共结果同时携带
`semantic_version` 与 64 位小写 `content_sha256`。执行前会用当前激活快照重新授权，
并同时核对生成时绑定的原问题、版本和内容摘要；任一身份变化、上下文缺失或 SQL
重新校验失败都会拒绝执行。携带 history ID 的受治理执行还必须持有同一生成器进程
颁发且尚未消费的本地 binding；重启后即使持久行身份完整，也会以
`HISTORY_CONTEXT_MISMATCH` 失败关闭，调用方必须重新生成。启用预检时，MySQL `EXPLAIN` 的数据库错误、墙钟超时、
空计划或超过阈值的估算扫描行数都会阻断执行；超时连接会失效并关闭，不会提交或回池复用。

## 6. 回滚

回滚必须在 Git 源上完成，再重新编译和激活；不要回滚 Neo4j 图来代替 Git 回滚。

```bash
git checkout <已审查提交> -- semantic_models/default.yaml
python scripts/compile_semantic_models.py semantic_models/default.yaml
python -m json.tool semantic_models/.compiled/manifest.json
python scripts/publish_semantic_model.py default \
  --compiled-root semantic_models/.compiled
```

若团队策略要求保留线性历史，使用 `git revert <变更提交>` 替代 `git checkout`。完成后重启应用或调用 `reload_governance(...)`。编译或重载失败时，已在内存中运行的旧有效快照不会被替换。

## 7. Golden Set 离线评估

Golden 用例在 `eval/golden/`；生成结果和报告是私有运行产物，分别写入已忽略的 `eval/results/` 与 `eval/reports/`。评测不会调用 LLM 或数据库。仓库跟踪的 `eval/examples/default-results.example.json` 是与默认七个匿名用例、`default@1.1.0` 严格匹配的 CLI smoke 输入；它只证明评测链路可运行，**不是**模型质量基准，也不应替代真实模型输出。

先编译当前 Git 源，再创建忽略目录并复制匿名示例：

```bash
python scripts/compile_semantic_models.py semantic_models/default.yaml
mkdir -p eval/results eval/reports
cp eval/examples/default-results.example.json eval/results/default-results.json
```

随后执行评测：

```bash
python scripts/evaluate_golden.py \
  --cases eval/golden/default.yaml \
  --results eval/results/default-results.json \
  --output eval/reports/default-report.json \
  --model local-model \
  --semantic-version 1.1.0
```

该匿名 smoke 输入应以退出码 `0` 完成。退出码为：`0` 全部用例通过，`1` 已完成评测但至少一例不通过，`2` 配置/输入/快照/I/O 无效。默认报告只保存问题与 SQL 指纹、稳定错误码和摘要元数据；模型版本修订仅以 `model_revision_sha256` 摘要表示。只有明确传入 `--include-content` 才会把完整问题和 SQL 写入本地调试报告，且该文件应按敏感数据保护。

### 从真实生成结果导出窄评测输入

真实评测应调用启用治理的 `SQLGenerator.generate_sql(...)` 或 `await SQLGenerator.generate_sql_async(...)`，并把其**公共结果映射**转换为严格的离线输入。当前代码没有结果导出 CLI；请在调用方使用受限序列化器，而不是把完整运行时对象、日志或历史记录写入文件。

外层 JSON 只允许 `schema_version`（固定 `1`）、`datasource`、`dialect`（`mysql`）、`semantic_version` 和 `results`。每个 `results` 项只允许 `id`（Golden case ID）、`sql`、`status`、`executable`、`semantic_version`、可选 `content_sha256`、`repair_attempts` 和 `validation`；`validation` 只允许 `valid` 与 `issues`，每个 issue 只保留稳定 `code`。不要复制 `user_query`、`error`、issue `message`、原始数据库异常、日志、连接信息、凭据或任何未知字段。

例如调用方可对每一个已知 case ID 取治理生成结果，并构造如下最小项（`result` 是上述公共 payload）：

```python
safe_result = {
    "id": case_id,
    "sql": result.get("sql"),
    "status": result["status"],
    "executable": result["executable"],
    "semantic_version": result.get("semantic_version"),
    "content_sha256": result.get("content_sha256"),
    "repair_attempts": result.get("repair_attempts"),
    "validation": {
        "valid": result["validation"].get("valid"),
        "issues": [
            {"code": issue["code"]}
            for issue in result["validation"].get("issues", [])
        ],
    },
}
```

将这些项放入上述外层 envelope 后，再传给现有 `scripts/evaluate_golden.py`；加载器会拒绝不匹配的 case ID、数据源、方言、语义版本、结果状态或任意未知字段。

## 8. 稳定错误码

对外生成结果和 Golden 结果共用下列稳定错误码（定义见 `core/sql/error_codes.py`）。前端、自动化和评测应依赖代码而不是英文/中文消息文本。

| 错误码 | 含义 |
| --- | --- |
| `AMBIGUOUS_COLUMN` | 未限定字段在输入表中无法唯一解析。 |
| `CORRECTNESS_UNAVAILABLE` | 正确性管线暂不可用，已失败关闭。 |
| `DIMENSION_NOT_ALLOWED` | 指标使用了未允许的维度。 |
| `EXECUTION_CONTEXT_REQUIRED` | 受治理执行缺少所需上下文。 |
| `EXPLAIN_FAILED` | MySQL 预检失败、超时、空计划或计划不被接受。 |
| `GENERATION_FAILED` | SQL 生成失败。 |
| `HISTORY_CONTEXT_MISMATCH` | 历史记录与受治理上下文不匹配。 |
| `INVALID_JOIN` | 连接不符合已治理的关系定义。 |
| `METRIC_MISMATCH` | 命中治理指标但聚合表达式不等价。 |
| `POLICY_DENIED` | 引用了禁用对象或不可直接查询的敏感字段。 |
| `QUERY_SCOPE_DENIED` | 表、schema 或阻止列表不允许该查询范围。 |
| `REPAIR_FAILED` | 唯一一次纠错未返回有效候选或执行失败。 |
| `SEMANTIC_CONTENT_MISMATCH` | 生成快照与执行授权快照的语义内容摘要不同。 |
| `SEMANTIC_VALIDATION_FAILED` | 语义校验失败且没有更具体的问题项。 |
| `SEMANTIC_VERSION_MISMATCH` | 生成快照与执行授权快照版本不同。 |
| `SQL_LENGTH_EXCEEDED` | SQL 在解析、修复或预检前超过统一安全长度上限。 |
| `SQL_VALIDATION_FAILED` | 基础 SQL 校验失败且没有更具体分类。 |
| `SYNTAX_ERROR` | SQL 解析或语法校验失败。 |
| `TIME_DIMENSION_REQUIRED` | 时间问题没有使用受治理的时间字段。 |
| `UNKNOWN_COLUMN` | 字段不存在或无法在别名作用域中解析。 |
| `UNKNOWN_TABLE` | 表不属于激活语义模型。 |
| `UNSAFE_SQL` | SQL 未通过只读安全校验。 |
| `UNSUPPORTED_SQL` | SQL 结构不在可确定校验的支持范围内。 |

## 9. 日志、隐私与故障排查

正常日志和默认 Golden 报告不写入完整问题、SQL 字面量、数据库原始异常、连接信息或凭据。请求审计记录语义版本、阶段耗时、错误码、是否纠错与可执行性；数据库异常仅进入受控、脱敏日志。不要把报告或日志加入 Few-Shot 数据源。

| 现象 | 检查与处置 |
| --- | --- |
| 编译失败或应用提示“语义治理配置无效” | 重新执行编译命令；检查 YAML 引用、重复名称、别名歧义、受限字段和版本/方言约束。修复 Git 源后再编译，不要手改 `.compiled/`。 |
| `SQL_EXPLAIN_ENABLED=true` 无法启动 | 同时设置 `SEMANTIC_GOVERNANCE_ENABLED=true`，并确认超时大于零、行数阈值非负。 |
| `EXPLAIN_FAILED` | 检查只读 MySQL 连接、超时、查询计划与扫描行数阈值；不要因预检失败跳过治理直接执行。 |
| SQL 被拒绝 | 读取 `validation.issues[].code`，按上表修正 YAML 或请求/SQL；一次纠错已经用尽时需发起新请求。 |
| 图检索结果与 YAML 不一致 | 重新编译、检查清单激活版本，再显式发布；Neo4j 不可作为覆盖 Git 的修复入口。 |
| Golden CLI 返回 `2` | 检查 cases/results 格式、case/result ID、语义版本、已激活快照和输出路径。 |

发布前至少运行 `pytest -q`、`ruff check .`、`python -m compileall -q core tests scripts`、编译器两次和 `git diff --check`。所有编译快照、评测结果、报告、日志、凭据和 IDE 文件都不得提交。
