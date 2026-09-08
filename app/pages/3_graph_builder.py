"""Streamlit 多页入口：复用统一的知识图谱构建页面。"""
import streamlit as st

st.set_page_config(page_title="知识图谱构建", page_icon="🏗️", layout="wide")

from app.pages.kg_builder import render_graph_builder  # noqa: E402


render_graph_builder()
