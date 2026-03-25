-- ==========================================
-- Text2SQL 智能体数据库建表语句
-- ==========================================

-- 创建数据库
CREATE DATABASE IF NOT EXISTS text2sql_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE text2sql_db;

-- ==========================================
-- 1. 知识库元数据表
-- 存储从 Neo4j 同步的图节点元数据
-- ==========================================
CREATE TABLE IF NOT EXISTS knowledge_metadata (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    node_id VARCHAR(255) NOT NULL COMMENT 'Neo4j 节点 ID',
    node_type VARCHAR(100) NOT NULL COMMENT '节点类型 (表/字段/业务概念)',
    node_label VARCHAR(255) NOT NULL COMMENT '节点标签/名称',
    description TEXT COMMENT '节点描述',
    properties JSON COMMENT '节点属性 (JSON 格式)',
    related_tables VARCHAR(500) COMMENT '关联的数据表',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_node_type (node_type),
    INDEX idx_node_label (node_label),
    INDEX idx_related_tables (related_tables)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='知识库元数据表';

-- ==========================================
-- 2. 数据表信息表
-- 存储 MySQL 数据库的表结构信息
-- ==========================================
CREATE TABLE IF NOT EXISTS table_schema (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    table_name VARCHAR(255) NOT NULL COMMENT '表名',
    table_comment VARCHAR(500) COMMENT '表注释',
    database_name VARCHAR(100) DEFAULT 'text2sql_db' COMMENT '数据库名',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_table_name (table_name, database_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='数据表信息表';

-- ==========================================
-- 3. 字段信息表
-- 存储 MySQL 数据库的字段结构信息
-- ==========================================
CREATE TABLE IF NOT EXISTS column_schema (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    table_id BIGINT NOT NULL COMMENT '关联的表 ID',
    column_name VARCHAR(255) NOT NULL COMMENT '字段名',
    column_type VARCHAR(100) NOT NULL COMMENT '字段类型',
    is_nullable TINYINT DEFAULT 1 COMMENT '是否可为空',
    is_primary_key TINYINT DEFAULT 0 COMMENT '是否主键',
    column_comment VARCHAR(500) COMMENT '字段注释',
    default_value VARCHAR(255) COMMENT '默认值',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (table_id) REFERENCES table_schema(id) ON DELETE CASCADE,
    INDEX idx_table_name (table_id),
    INDEX idx_column_name (column_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='字段信息表';

-- ==========================================
-- 4. SQL 生成历史表
-- 存储 SQL 生成和使用历史
-- ==========================================
CREATE TABLE IF NOT EXISTS sql_generation_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) COMMENT '会话 ID',
    user_query TEXT NOT NULL COMMENT '用户自然语言查询',
    generated_sql TEXT COMMENT '生成的 SQL 语句',
    sql_dialect VARCHAR(50) DEFAULT 'mysql' COMMENT 'SQL 方言',
    validation_status VARCHAR(20) DEFAULT 'pending' COMMENT '验证状态：pending/pass/fail',
    validation_error TEXT COMMENT '验证错误信息',
    execution_status VARCHAR(20) DEFAULT 'pending' COMMENT '执行状态：pending/success/fail',
    execution_error TEXT COMMENT '执行错误信息',
    execution_result JSON COMMENT '执行结果 (脱敏)',
    matched_knowledge JSON COMMENT '匹配的知识节点 (JSON)',
    retrieval_query TEXT COMMENT '检索使用的查询向量',
    feedback_score TINYINT COMMENT '用户反馈评分 (1-5)',
    feedback_comment TEXT COMMENT '用户反馈备注',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_created_at (created_at),
    INDEX idx_validation_status (validation_status),
    INDEX idx_execution_status (execution_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='SQL 生成历史表';

-- ==========================================
-- 5. 检索词表
-- 存储用于 BM25+TF-IDF 检索的分词
-- ==========================================
CREATE TABLE IF NOT EXISTS retrieval_terms (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    term VARCHAR(255) NOT NULL COMMENT '检索词',
    term_type VARCHAR(50) COMMENT '词类型：table/column/business/alias',
    frequency INT DEFAULT 1 COMMENT '词频',
    related_node_ids JSON COMMENT '关联的节点 ID 列表',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_term (term),
    INDEX idx_term_type (term_type),
    INDEX idx_frequency (frequency DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='检索词表';

-- ==========================================
-- 6. 用户反馈表
-- 存储用户对生成 SQL 的反馈
-- ==========================================
CREATE TABLE IF NOT EXISTS user_feedback (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    history_id BIGINT NOT NULL COMMENT '关联的历史记录 ID',
    feedback_type VARCHAR(50) NOT NULL COMMENT '反馈类型：correct/incorrect/improve',
    feedback_score TINYINT COMMENT '评分 (1-5)',
    feedback_comment TEXT COMMENT '反馈备注',
    corrected_sql TEXT COMMENT '用户修正后的 SQL',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (history_id) REFERENCES sql_generation_history(id) ON DELETE CASCADE,
    INDEX idx_history (history_id),
    INDEX idx_feedback_type (feedback_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='用户反馈表';

-- ==========================================
-- 初始化数据 - 示例表结构
-- ==========================================

-- 示例：用户表
INSERT INTO table_schema (table_name, table_comment, database_name)
VALUES ('users', '用户信息表', 'text2sql_db');

SET @table_id = LAST_INSERT_ID();

INSERT INTO column_schema (table_id, column_name, column_type, is_nullable, is_primary_key, column_comment) VALUES
(@table_id, 'id', 'INT', 0, 1, '用户 ID'),
(@table_id, 'username', 'VARCHAR(50)', 0, 0, '用户名'),
(@table_id, 'email', 'VARCHAR(100)', 1, 0, '邮箱地址'),
(@table_id, 'created_at', 'TIMESTAMP', 1, 0, '创建时间'),
(@table_id, 'status', 'TINYINT', 0, 0, '状态：0-禁用，1-启用');

-- 示例：订单表
INSERT INTO table_schema (table_name, table_comment, database_name)
VALUES ('orders', '订单信息表', 'text2sql_db');

SET @table_id = LAST_INSERT_ID();

INSERT INTO column_schema (table_id, column_name, column_type, is_nullable, is_primary_key, column_comment) VALUES
(@table_id, 'id', 'INT', 0, 1, '订单 ID'),
(@table_id, 'user_id', 'INT', 0, 0, '用户 ID'),
(@table_id, 'order_no', 'VARCHAR(50)', 0, 0, '订单编号'),
(@table_id, 'amount', 'DECIMAL(10,2)', 0, 0, '订单金额'),
(@table_id, 'status', 'VARCHAR(20)', 0, 0, '订单状态'),
(@table_id, 'created_at', 'TIMESTAMP', 1, 0, '创建时间');

-- 示例：产品表
INSERT INTO table_schema (table_name, table_comment, database_name)
VALUES ('products', '产品信息表', 'text2sql_db');

SET @table_id = LAST_INSERT_ID();

INSERT INTO column_schema (table_id, column_name, column_type, is_nullable, is_primary_key, column_comment) VALUES
(@table_id, 'id', 'INT', 0, 1, '产品 ID'),
(@table_id, 'name', 'VARCHAR(200)', 0, 0, '产品名称'),
(@table_id, 'price', 'DECIMAL(10,2)', 0, 0, '产品价格'),
(@table_id, 'category', 'VARCHAR(50)', 1, 0, '产品类别'),
(@table_id, 'stock', 'INT', 0, 0, '库存数量');
