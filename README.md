# Text2SQL 智能体

基于本地知识库的 SQL 智能生成系统

> **最新版本**: v1.3
> **核心升级**: 关系权重追踪 + JOIN 提取引擎 + 反馈驱动图谱增强 + 权重感知 Prompt

## 🎯 功能特性

### 核心功能
- 📊 **自然语言转 SQL**: 输入自然语言查询，自动生成 SQL 语句
- 🧠 **知识库检索**: BM25 + FAISS 两路混合检索
- 🔍 **SQL 校验**: 使用 SQLFluff 进行语法校验
- 📜 **历史记录**: 完整的 SQL 生成和执行历史管理
- 🗄️ **数据库选择**: 支持选择不同数据库，生成对应方言的 SQL
- 🔗 **Neo4j 知识图谱**: 从图数据库同步元数据
- 💾 **MySQL 存储**: 历史数据和元数据持久化
- 🏗️ **知识图谱构建**: 支持 DDL 文件/活体数据库构建图谱

### 新增功能 (v1.3)

| 功能 | 说明 | 状态 |
|------|------|------|
| ⚖️ **关系权重追踪** | CONNECTS 关系新增 weight/occurrence_count/source 属性 | ✅ |
| 🔍 **JOIN 提取引擎** | 基于 sqlglot AST 从历史 SQL 中提取表关联关系 | ✅ |
| 🔄 **权重重平衡** | 历史记录驱动，平滑增量权重 + 新关系比例收缩 | ✅ |
| 💪 **提取&增强** | 历史页一键提取 JOIN 关系并增强知识图谱 | ✅ |
| 🧠 **权重感知 Prompt** | LLM 优先选择高权重关联关系，关系按权重降序展示 | ✅ |
| 🏗️ **应用上下文** | ApplicationContext 统一管理组件生命周期，消除全局单例 | ✅ |

### 新增功能 (v1.2)

| 功能 | 说明 | 状态 |
|------|------|------|
| 🏗️ **知识图谱构建** | 从 DDL/JSON 文件或活体数据库构建知识图谱 | ✅ |
| 🗄️ **多数据库支持** | 支持 MySQL、Oracle、Hive、PostgreSQL、SparkSQL | ✅ |
| 🎯 **数据库选择** | 前端可选择目标数据库，生成对应方言 SQL | ✅ |
| 🤖 **AI 语义增强** | LLM 自动推断表关系和生成业务标签 | ✅ |
| 📤 **图谱导出/导入** | 支持将知识图谱导出为 JSON 并重新导入 | ✅ |
| 🔄 **自动重试机制** | LLM 调用失败自动重试 | ✅ |
| 🧹 **一键清除** | 支持清除知识图谱数据 | ✅ |

### v1.1 功能

| 功能 | 说明 | 状态 |
|------|------|------|
| 🗄️ **MySQL 连接池** | SQLAlchemy QueuePool，支持高并发 | ✅ |
| 🔗 **Neo4j 连接池** | Driver 级别连接池管理 | ✅ |
| ⚡ **异步 LLM 调用** | ainvoke 替代 invoke，提升吞吐量 | ✅ |
| 🧠 **CoT 思维链** | XML 标签格式，先思考后输出 | ✅ |
| 📚 **Few-Shot 学习** | 从历史成功记录检索相似示例 | ✅ |
| 🔌 **真实 Embedding** | 通义千问 text-embedding-v1 (1536 维) | ✅ |

## 📁 项目结构

```
text2sql_agent/
├── app.py                      # Streamlit 主入口
├── config.py                   # 配置文件
├── requirements.txt            # 依赖
├── schema.sql                  # 数据库建表语句
├── core/                       # 核心模块
│   ├── llm_factory.py          # LLM 工厂
│   ├── embedding_factory.py    # Embedding 工厂
│   ├── knowledge/
│   │   └── neo4j_client.py     # Neo4j 客户端 (连接池 + 异步 + 知识图谱操作)
│   ├── history/
│   │   └── mysql_client.py     # MySQL 客户端 (连接池)
│   ├── retrieval/
│   │   └── bm25_tfidf.py       # 两路混合检索 (BM25 + FAISS) + Few-Shot 索引
│   ├── chain/
│   │   ├── prompts.py          # CoT 思维链模板
│   │   └── sql_chain.py        # SQL 生成链 (带重试机制)
│   ├── sql/
│   │   ├── generator.py        # SQL 生成器
│   │   ├── dialect_detector.py # SQL 方言检测器
│   │   ├── validator.py        # SQL 校验器
│   │   ├── field_validator.py  # sqlglot AST 解析
│   │   └── join_extractor.py   # ✨ JOIN 提取引擎 (v1.3 新增)
│   ├── app_context.py          # ✨ 应用上下文 (v1.3 新增)
│   └── graph_builder/          # ✨ 知识图谱构建模块 (v1.2 新增)
│       ├── __init__.py
│       ├── models.py           # IR 数据模型 (含权重字段)
│       ├── extractors/         # 抽取器 (DDL/JSON/活体数据库)
│       ├── enrichment/         # AI 语义增强
│       ├── compiler/           # SQL 预编译
│       └── loader/             # Neo4j 装载器
│           ├── neo4j_writer.py # 批量装载 (含权重初始化)
│           └── weight_initializer.py  # ✨ 权重初始化/重平衡 (v1.3 新增)
├── app/                        # ✨ Streamlit 页面模块 (v1.2 新增)
│   └── pages/
│       ├── kg_builder.py       # 知识图谱构建页面
│       └── graph_builder.py    # 图谱状态页面
├── tests/                      # 测试
│   └── test_graph_builder.py
└── logs/                       # 日志目录
```

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.9+
python --version

# 创建虚拟环境
cd /Users/rgj/PycharmProjects/text2sql_agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**新增依赖说明**:
- `faiss-cpu>=1.7.4` - FAISS 向量检索
- `dashscope>=1.14.0` - 通义千问 Embedding
- `sqlglot>=21.0.0` - SQL AST 解析

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，配置数据库和 LLM
```

### 4. 初始化数据库

```bash
# MySQL
mysql -u root -p < schema.sql
```

### 5. 启动应用

```bash
streamlit run app.py
```

访问 http://localhost:8501

## 📋 配置说明

### 环境变量完整列表

#### 数据库配置
| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| MYSQL_HOST | MySQL 主机 | localhost |
| MYSQL_PORT | MySQL 端口 | 3306 |
| MYSQL_USER | MySQL 用户 | root |
| MYSQL_PASSWORD | MySQL 密码 | root |
| MYSQL_DATABASE | MySQL 数据库 | text2sql_db |
| **MYSQL_POOL_SIZE** | ✨ MySQL 连接池大小 | 10 |
| **MYSQL_MAX_OVERFLOW** | ✨ 最大溢出连接数 | 20 |
| **MYSQL_POOL_TIMEOUT** | ✨ 获取连接超时（秒） | 30 |

#### Neo4j 配置
| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| NEO4J_ENABLED | 是否启用 Neo4j | false |
| NEO4J_URI | Neo4j 连接 URI | bolt://localhost:7687 |
| NEO4J_USER | Neo4j 用户 | neo4j |
| NEO4J_PASSWORD | Neo4j 密码 | neo4j@123 |
| **NEO4J_MAX_POOL_SIZE** | ✨ 最大连接池大小 | 50 |
| **NEO4J_CONNECTION_TIMEOUT** | ✨ 连接超时（秒） | 30 |

#### LLM 配置
| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| LLM_PROVIDER | LLM 提供商 | openai |
| LLM_MODEL | LLM 模型 | gpt-4 |
| LLM_API_KEY | LLM API Key | - |
| LLM_BASE_URL | LLM API 基础 URL | - |

#### Embedding 配置（新增）
| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| EMBEDDING_PROVIDER | Embedding 提供商 | dashscope |
| EMBEDDING_MODEL | Embedding 模型 | text-embedding-v1 |
| EMBEDDING_API_KEY | API Key | - |

#### Few-Shot 配置（新增）
| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| FEW_SHOT_ENABLED | 启用 Few-Shot | true |
| FEW_SHOT_TOP_K | 检索示例数量 | 3 |
| FEW_SHOT_MIN_SIMILARITY | 最小相似度 | 0.5 |

#### 检索配置
| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| BM25_K1 | BM25 k1 参数 | 1.5 |
| BM25_B | BM25 b 参数 | 0.75 |
| BM25_WEIGHT | BM25 权重 | 0.6 |
| DENSE_WEIGHT | FAISS 权重 | 0.4 |

### LLM 配置示例

```bash
# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
LLM_API_KEY=sk-xxx

# Azure OpenAI
LLM_PROVIDER=azure
LLM_MODEL=gpt-4
LLM_API_KEY=xxx
LLM_BASE_URL=https://xxx.openai.azure.com

# 本地 Ollama
LLM_PROVIDER=ollama
LLM_MODEL=llama2
LLM_BASE_URL=http://localhost:11434

# 通义千问
LLM_PROVIDER=openai
LLM_MODEL=qwen-max
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 📖 使用示例

### 1. 提取&增强（v1.3 新增）

在"执行历史"页面：
1. 找到已执行的 SQL 记录
2. 点击"提取&增强"按钮
3. 系统自动提取 SQL 中的表关联关系（JOIN）
4. 确认无误后点击"确认增强到知识图谱"
5. 系统自动更新 Neo4j 中 CONNECTS 关系的权重

### 2. 知识图谱构建（v1.2 新增）

在"知识图谱构建"页面：
1. 上传 DDL 文件或 JSON 文件
2. 填写数据库名称和类型（支持 MySQL、Oracle、Hive、PostgreSQL、SparkSQL）
3. 选择是否启用 AI 语义增强
4. 点击"开始构建"
5. 构建完成后，点击侧边栏的"🔄 重新初始化系统"刷新索引

**DDL 文件示例**:
```sql
CREATE TABLE users (
    id INT PRIMARY KEY COMMENT '用户 ID',
    name VARCHAR(100) COMMENT '用户姓名',
    email VARCHAR(255) COMMENT '邮箱'
) COMMENT='用户表';

CREATE TABLE orders (
    id INT PRIMARY KEY COMMENT '订单 ID',
    user_id INT COMMENT '用户 ID',
    amount DECIMAL(10,2) COMMENT '订单金额',
    created_at DATETIME COMMENT '创建时间'
) COMMENT='订单表';
```

### 3. SQL 生成（带数据库选择）

在主页选择目标数据库，然后输入自然语言查询：
- 选择数据库：`ride_hailing_db (sparksql)`
- 查询："查询近 7 天接单量排名前 10 的司机"

**CoT 输出示例**:
```
🧠 思考过程:
用户意图：统计近 7 天接单量排名前 10 的司机
关键实体：司机、接单量、排名、近 7 天
需要用到的表：driver, order
关联关系：通过 driver_id 关联
SQL 结构:
1. 从 order 表中筛选近 7 天的记录
2. 按 driver_id 分组，计算 COUNT(*)
3. 使用 ORDER BY 降序排列
4. 使用 LIMIT 10 取前 10 名
```

### 4. 数据库选择（v1.2 新增）

在"SQL 生成"页面左侧，可以选择目标数据库：
- 下拉框显示：`数据库名 (数据库类型)`
- 选择不同数据库后，系统会生成对应方言的 SQL
- 支持 MySQL、Oracle、Hive、PostgreSQL、SparkSQL

### 5. 图谱导出/导入（v1.2 新增）

在侧边栏"知识库管理"中：
- **📤 导出**：将知识图谱导出为 JSON 文件
- **📥 导入**：上传 JSON 文件，恢复知识图谱

### 6. 清除知识库（v1.2 新增）

在侧边栏"知识库管理"中：
1. 勾选"✓ 确认清除"
2. 点击"🗑️ 清除知识库"
3. 系统自动重新初始化

### 7. 查看历史

切换到"执行历史"标签页，查看历史生成记录。

## 🏗️ 架构设计

### 知识图谱权重架构（v1.3 新增）

```
CONNECTS 关系属性:
├── relationship_type: foreign_key
├── join_type: LEFT JOIN / INNER JOIN / ...
├── join_sql: 预编译的 JOIN SQL
├── from_column / to_column: 关联字段
├── weight: 0.0 ~ 1.0 (同一表对之间权重和 = 1)
├── occurrence_count: 出现次数
├── source: manual | history_extracted
└── updated_at: 最后更新时间

权重初始化:
  表 A → 表 B 有 N 条关系
  每条 weight = 1/N

权重更新 (历史记录驱动):
  已存在: new_weight = old_weight × (1 - η) + η
  新关系: weight = 1 / (已有数量 + 1)
         旧关系按比例收缩
```

### 完整处理流程（v1.2）

```
用户输入
   ↓
数据库选择 ────────────────────┐
   ↓                            │
查询改写 (可选)                  │
   ↓                            │
知识检索 ──────────────────────┼┐
   ├─ BM25 检索                 ││
   └─ FAISS 向量检索 (Embedding)││
   ↓                            ││
Few-Shot 检索 ←── 历史成功记录 ─┘│
   ↓                            │
提示词构建 ←──────── 数据库方言 ─┘
   ├─ Schema 信息（按数据库过滤，关系按权重降序）
   ├─ 相关知识
   ├─ Few-Shot 示例
   └─ CoT 思维链指令（优先选择高权重关联）
   ↓
异步 LLM 调用 (带重试)
   ↓
XML 标签提取
   ├─ <thinking> 思考过程
   └─ <sql> SQL 语句
   ↓
SQL 校验 (SQLFluff)
   ↓
执行/保存
   ├─ MySQL 连接池
   └─ 历史记录 → Few-Shot 索引
```

### 知识图谱构建流水线（v1.2 新增）

```
┌─────────────────────────────────────────────────────┐
│                 GraphBuilder Pipeline               │
├─────────────────────────────────────────────────────┤
│  Extractor 层 → Enrichment 层 → Compiler 层 → Loader 层 │
│  (多源抽取)    (AI 增强)      (SQL 预编译)  (Neo4j 装载) │
└─────────────────────────────────────────────────────┘

数据源:
├── DDL 文件 (.sql)
├── JSON 文件 (.json)
└── 活体数据库 (MySQL/Oracle/Spark)

输出:
└── Neo4j 知识图谱
    ├── Database 节点
    ├── Table 节点
    ├── Column 节点
    ├── BusinessConcept 节点
    └── CONNECTS 关系边
```

### 连接池架构

```
应用层
   ↓
SQLAlchemy QueuePool (MySQL)
   ├─ pool_size = 10
   ├─ max_overflow = 20
   ├─ pool_timeout = 30s
   ├─ pool_pre_ping = true (自动检测失效连接)
   └─ pool_recycle = 3600s (1 小时回收)

Neo4j Driver Pool
   ├─ max_connection_pool_size = 50
   └─ connection_timeout = 30s
```

## ⚡ 性能提升

| 场景 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 并发请求 (3 个) | 串行 ~12 秒 | 并行 ~10 秒 | ~17% |
| 复杂查询准确率 | 基础 | CoT+Few-Shot | ~30% |
| 连接等待 | 每次新建 | 连接池 | ~80% |
| 语义检索 | 词频匹配 | Embedding | 显著 |
| 多表 JOIN 准确率 | 随机选择关联 | 权重优先 | 显著提升 |

## 🤖 AI Agent 专属技能

项目提供了命令行 Schema 检索工具，供外部 AI Agent（如 Claude Code）调用。

### Schema Agent CLI

**文件位置**: `scripts/02_schema_search_cli.py`

**基本用法**:

```bash
# 检索 Schema 并由 LLM 总结输出
python scripts/02_schema_search_cli.py "查询用户订单信息"

# 仅返回原始 Schema 信息（不加 LLM 总结）
python scripts/02_schema_search_cli.py "查询用户订单信息" --raw

# 指定返回结果数量
python scripts/02_schema_search_cli.py "查询用户订单信息" --top-k 10

# 指定数据库 ID
python scripts/02_schema_search_cli.py "查询用户订单信息" --database-id 2

# 以 JSON 格式输出（便于程序解析）
python scripts/02_schema_search_cli.py "查询用户订单信息" --json
```

**参数说明**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `query` | 自然语言查询语句（位置参数） | - |
| `--top-k` | 返回结果数量 | 5 |
| `--database-id` | 数据库 ID（用于过滤） | None |
| `--raw` | 仅返回原始 Schema 信息 | False |
| `--json` | 以 JSON 格式输出 | False |

**输出示例**:

```
根据您的需求"查询用户订单信息"，建议使用以下数据库表：

**建议使用的表**:
- `bas_user_info` - 用户基本信息表
- `bas_user_auth` - 用户认证信息表
- `acc_account_main` - 账户主表

**需要关注的核心字段**:
- `bas_user_info.user_id` - 用户 ID（主键）
- `bas_user_info.user_name` - 用户姓名
- `acc_account_main.account_id` - 账户 ID
- `acc_account_main.balance` - 账户余额

**表关联关系**:
- `bas_user_info` 和 `acc_account_main` 通过 `user_id` 字段关联
```

### Schema Export Tool（可选）

**文件位置**: `scripts/01_export_graph_to_md.py`

导出 Neo4j 知识图谱为 Markdown 文档，用于知识快照或离线查看：

```bash
python scripts/01_export_graph_to_md.py
```

输出文件：`data/schemas/schema_export_YYYYMMDD_HHMMSS.md`

## 🛠️ 技术栈

- **前端**: Streamlit
- **后端**: Python 3.9+
- **数据库**: MySQL (连接池), Neo4j (连接池)
- **LLM**: LangChain (异步调用)
- **检索**: BM25, FAISS
- **Embedding**: 通义千问 text-embedding-v1
- **校验**: SQLFluff, sqlglot

## ⚠️ 注意事项

1. **安全性**: 只允许执行 SELECT 查询
2. **性能**: 大表查询自动添加 LIMIT 警告
3. **准确性**: 依赖知识库的完整性
4. **LLM 成本**: 注意 API 调用频率
5. **连接池**: 高并发场景建议调大 pool_size

## 📝 更新日志

### v1.3 (2026-04-12)

**关系权重追踪**:
- ✅ CONNECTS 关系新增 `weight`/`occurrence_count`/`source` 属性
- ✅ 图谱构建时自动平分同一表对之间的多条关系权重
- ✅ Neo4j 关系存储权重信息并支持查询排序

**JOIN 提取引擎**:
- ✅ 基于 sqlglot AST 解析从历史 SQL 中提取 JOIN 关系
- ✅ 提取 left_table、right_table、join_condition、join_type
- ✅ 支持复杂 ON 条件拆分（主关联 + 额外条件）
- ✅ 支持 LEFT/RIGHT/INNER/FULL/CROSS JOIN

**权重重平衡算法**:
- ✅ 平滑增量：`new_weight = old_weight × (1 - η) + η`
- ✅ 新关系创建时按比例收缩旧关系权重
- ✅ 自动维护同一表对权重之和 = 1

**反馈驱动图谱增强**:
- ✅ 历史页新增"提取&增强"按钮
- ✅ 提取 JOIN → 展示确认 → 更新 Neo4j 权重
- ✅ 闭环优化知识图谱的表关联关系

**权重感知 Prompt**:
- ✅ 关系信息按权重降序排列
- ✅ LLM 优先选择高权重关联关系生成 SQL
- ✅ 提升多表 JOIN 场景的准确率

**架构优化**:
- ✅ ApplicationContext 统一管理组件生命周期
- ✅ 消除全局单例，改为依赖注入模式
- ✅ `_get_schema_from_neo4j` 修复关系类型查询
- ✅ 消除 sync/async 方法代码重复

### v1.2 (2026-04-01)

**知识图谱构建**:
- ✅ 新增 GraphBuilder Pipeline 模块
- ✅ 支持从 DDL 文件抽取元数据
- ✅ 支持从 JSON 文件抽取元数据
- ✅ 支持从活体数据库抽取元数据
- ✅ AI 语义增强：LLM 自动推断表关系
- ✅ AI 语义增强：LLM 生成业务概念标签
- ✅ SQL 预编译：生成 JOIN SQL 模板

**多数据库支持**:
- ✅ 支持 MySQL、Oracle、Hive、PostgreSQL、SparkSQL
- ✅ 数据库 ID 自增管理
- ✅ UNWIND + MERGE 批量装载模式

**前端功能**:
- ✅ 知识图谱构建页面
- ✅ 数据库选择下拉框
- ✅ 图谱导出为 JSON
- ✅ 从 JSON 导入图谱
- ✅ 一键清除知识库
- ✅ 重新初始化系统按钮

**优化改进**:
- ✅ LLM 调用增加重试机制
- ✅ 方言检测器支持多数据库
- ✅ 错误提示优化

### v1.1 (2026-03-25)

**数据库优化**:
- ✅ MySQL 使用 SQLAlchemy QueuePool 连接池
- ✅ Neo4j 配置 Driver 级别连接池
- ✅ 支持连接池参数配置（pool_size, max_overflow 等）

**异步改造**:
- ✅ SQLGenerationChain 新增 generate_async 方法
- ✅ QueryRewriteChain 新增 rewrite_async 方法
- ✅ SQLGenerator 新增 generate_sql_async 方法

**CoT 思维链**:
- ✅ 系统提示词支持 XML 标签格式
- ✅ 先生成思考过程，再生成 SQL
- ✅ 前端展示思考过程（可折叠）

**Few-Shot 学习**:
- ✅ 接入通义千问 Embedding (1536 维)
- ✅ 使用 FAISS 构建 Few-Shot 索引
- ✅ 自动从历史成功记录检索相似示例

**AST 解析**:
- ✅ 使用 sqlglot 替代正则提取表名/字段名
- ✅ 支持 CTE、子查询、Window 函数等复杂场景

### v1.0 (初始版本)

- ✅ 基础 SQL 生成功能
- ✅ BM25 + FAISS 混合检索
- ✅ Streamlit UI
- ✅ 历史记录管理

## 📄 许可证

MIT License
