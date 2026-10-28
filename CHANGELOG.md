# CHANGELOG

本项目的所有重要变更记录于此文件。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased] - 2026-10-25

### Fixed

- **模型配置面板：连续保存静默丢失用户输入**（`app.py` `_reseed_model_widgets`）
  上一次保存/恢复成功后设置的 reseed 标记，会在下一次保存按钮触发的 rerun 里
  先清掉 `cfg_llm_model` 再用旧配置重新播种，导致用户新输入的模型名被静默丢弃、
  界面却显示保存成功（连续第二次保存永远写回旧值）。
  修复：保存/恢复按钮触发的本轮 rerun 不清除控件状态，标记留到下一轮消费。

- **检索元数据缺少 `database_id`，按库过滤后零命中**
  （`core/app_context.py` `_populate_retriever` / `_load_knowledge_from_neo4j`，
  `core/sql/generator.py` `create_sql_generator`）
  索引元数据未携带 `database_id`，检索过滤后无任何命中。已在所有索引入口补齐。

- **MySQL 管理表不会自动创建，新库指向即全线瘫痪**
  （`core/history/mysql_client.py` 新增 `_ensure_management_tables`）
  代码中原本没有任何建表语句，6 张管理表依赖手工执行 `schema.sql`。
  现改为连接验证成功后幂等自动创建（`CREATE TABLE IF NOT EXISTS`），
  随后照常执行治理列迁移。首次指向新数据库（如 `business_db`）开箱可用。

### Changed

- **默认 LLM 模型**：`gemma3:e4b` → `gemma4:E4B`
  （`config.py` 环境变量兜底值、`core/provider_catalog.py` ollama 提供商推荐模型）。
  旧默认值在本地 ollama 不存在，导致 `OllamaEndpointNotFoundError`。

- **测试断言同步**：`tests/test_comprehensive.py` 的 `MYSQL_DATABASE`
  断言由 `text2sql_db` 更新为 `business_db`。

### Environment（配置与数据，非代码）

- `.env`：`LLM_MODEL=gemma4:E4B`、`MYSQL_HOST=127.0.0.1`、`MYSQL_DATABASE=business_db`。
  - `MYSQL_HOST` 改 `127.0.0.1`：本机 MySQL 运行在 Docker 容器中，`localhost`
    连接经 NAT 回环被识别为 `root'@'192.168.97.1`，触发 Access denied。
  - `MYSQL_DATABASE` 改 `business_db`：应用单一 MySQL 连接同时承担业务执行与
    历史存储，原指向的 `text2sql_db` 只有管理表，业务 SQL 执行必然失败。

- **Neo4j 知识图谱库名统一**：`legacy_db` → `business_db`（Database 节点 name 属性，
  全图扫描确认无残留；101 节点 / 100 边无损）。

- **历史记录迁移**：`text2sql_db.sql_generation_history` 的 39 条记录迁入
  `business_db`（含 4 月旧记录），未丢失。

### Verified

- 全量回归：`934 passed, 16 skipped`。
- MySQL 真实集成测试（`TEXT2SQL_RUN_DB_TESTS=1`）：6 个全绿，
  跑后 `table_schema`/`column_schema`/`sql_generation_history` 零测试残留。
- 端到端：检索（命中 risk_blacklist）→ SQL 生成 → 执行 → 历史回写，
  在干净图谱 + 新配置下全链路通过。

### Fixed（本轮补充）

- `fix(mysql)`：`table_schema` 建表 DDL 缺失 `properties` 列，
  新增幂等迁移 `_migrate_table_schema()` 为旧库补列；
  `upsert_table_schema` 失败时不再静默返回 True（`raise_on_error=True`）。
- `fix(mysql)`：其余 4 处 `execute_update` 失败静默写法统一改 `raise_on_error=True`
  （`upsert_column_schema`、`update_sql_execution`、`save_feedback`×2、
  `generator.fix_sql` 历史更新），写失败立即显式报错。
- `feat(ui)`：侧栏品牌 logo 换新（`assets/logo.png` 缩至 480 宽，
  `assets/logo_icon.png` 由新图徽标区重裁 128×128 页签图标，
  `assets/logo_full.png` 保留全分辨率原图）；
  透明底（RGBA）处理，侧栏白底正常融合；
  logo 点击放大全屏、再点击缩小（components.html 同源 iframe
  内通过 frameElement 撑满视口实现，全屏态用全分辨率图）。
- `fix(ui)`：侧栏知识库统计 Cypher 把关系与节点行做笛卡尔积
  （`OPTIONAL MATCH ()-[r]-()` 未绑定节点），显示值 = 节点数 × 关系数；
  改为 `OPTIONAL MATCH (n)-[r]-()` + `count(DISTINCT r)`，
  真实 101 节点 / 100 关系不再显示为 20200。
- `feat(ui)`：图标体系从 emoji 升级为 Fluent Color 内联 SVG
  （新增 `app/ui_icons.py`，19 个图标，MIT 协议，零外部请求，
  https://github.com/microsoft/fluentui-system-icons ，
  微软 Fluent UI 彩色圆角风格，经 Iconify 抓取 24px 版本）；
  彩色固定配色，不再继承文本色；
  节标题 / 侧栏状态行换 SVG 图标（UIkit iconnav 的等距排列风格），
  按钮、页签、反馈消息去除装饰 emoji；kg_builder hero 徽章同步换图标。
- `test(mysql)`：集成测试连接串改读 `config`，
  自带夹具播种/清理，不再硬编码 `localhost`/`text2sql_db`。
