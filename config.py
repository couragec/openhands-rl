"""简单配置 - 主要通过环境变量获取"""

import os


def get_llm_api_key() -> str:
    return os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


def get_llm_model() -> str:
    return os.environ.get("LLM_MODEL") or os.environ.get("CHAT_MODEL", "gpt-4o")


def get_llm_base_url() -> str:
    return os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")


def get_grading_server_url() -> str:
    return os.environ.get("GRADING_SERVER_URL", "http://localhost:5000")
