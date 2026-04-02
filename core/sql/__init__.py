"""
core/sql 模块
提供 SQL 生成、方言转换、SQL 验证等功能
"""
from .dialect_detector import SQLDialectDetector

__all__ = ["SQLDialectDetector"]
