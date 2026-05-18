#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="${PYTHON_BIN}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "未找到 python3 或 python，请先安装 Python 3.9+。" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/.env" && -f "${PROJECT_ROOT}/env.example" ]]; then
    cp "${PROJECT_ROOT}/env.example" "${PROJECT_ROOT}/.env"
    echo "已从 env.example 创建 .env，请按服务器环境补充数据库和 LLM 配置。"
fi

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]] && continue

        key="${line%%=*}"
        value="${line#*=}"
        key="${key%"${key##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"

        if [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ && -z "${!key:-}" ]]; then
            export "${key}=${value}"
        fi
    done < "${PROJECT_ROOT}/.env"
fi

PYTHON_DEPS_DIR="${PYTHON_DEPS_DIR:-python_deps}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-.pip_cache}"
STREAMLIT_ADDRESS="${STREAMLIT_ADDRESS:-0.0.0.0}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
APP_ENTRY="${APP_ENTRY:-app.py}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

if [[ "${PYTHON_DEPS_DIR}" != /* ]]; then
    PYTHON_DEPS_DIR="${PROJECT_ROOT}/${PYTHON_DEPS_DIR}"
fi

if [[ "${PIP_CACHE_DIR}" != /* ]]; then
    PIP_CACHE_DIR="${PROJECT_ROOT}/${PIP_CACHE_DIR}"
fi

mkdir -p "${PROJECT_ROOT}/logs" "${PYTHON_DEPS_DIR}" "${PIP_CACHE_DIR}"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
    REQUIREMENTS_HASH="$("${PYTHON}" - <<'PY'
from pathlib import Path
import hashlib

requirements = Path("requirements.txt")
print(hashlib.sha256(requirements.read_bytes()).hexdigest())
PY
)"
    STAMP_FILE="${PYTHON_DEPS_DIR}/.requirements.sha256"
    INSTALLED_HASH=""
    if [[ -f "${STAMP_FILE}" ]]; then
        INSTALLED_HASH="$(cat "${STAMP_FILE}")"
    fi

    if [[ "${REQUIREMENTS_HASH}" != "${INSTALLED_HASH}" ]]; then
        echo "正在安装/更新 Python 依赖到 ${PYTHON_DEPS_DIR} ..."
        "${PYTHON}" -m pip install \
            --upgrade \
            --target "${PYTHON_DEPS_DIR}" \
            --cache-dir "${PIP_CACHE_DIR}" \
            -r requirements.txt
        echo "${REQUIREMENTS_HASH}" > "${STAMP_FILE}"
    else
        echo "Python 依赖未变化，跳过安装。"
    fi
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHON_DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "启动 Text2SQL: http://${STREAMLIT_ADDRESS}:${STREAMLIT_PORT}"
exec "${PYTHON}" -m streamlit run "${APP_ENTRY}" \
    --server.address "${STREAMLIT_ADDRESS}" \
    --server.port "${STREAMLIT_PORT}" \
    --server.headless true
