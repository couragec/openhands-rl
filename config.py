"""配置管理 - 通过环境变量获取 LLM 和 FSM 相关配置。"""

import os
import time
from pathlib import Path


def get_llm_api_key() -> str:
    return os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


def get_llm_model() -> str:
    return os.environ.get("LLM_MODEL") or os.environ.get("CHAT_MODEL", "gpt-4o")


def get_llm_base_url() -> str:
    return os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")


def get_grading_server_url() -> str:
    return os.environ.get("GRADING_SERVER_URL", "http://localhost:5000")


def get_fsm_enabled() -> bool:
    return os.environ.get("FSM_ENABLED", "").lower() in ("1", "true", "yes")


def get_fsm_target_repo() -> str:
    return os.environ.get("FSM_TARGET_REPO", ".")


def get_fsm_repair_iters() -> int:
    try:
        return int(os.environ.get("FSM_REPAIR_ITERS", "3"))
    except ValueError:
        return 3


def get_fsm_deploy_engine() -> str:
    return os.environ.get("FSM_DEPLOY_ENGINE", "vllm")


def get_opencode_url() -> str:
    return os.environ.get("OPENCODE_URL", "")


def get_opencode_model() -> str:
    return os.environ.get("OPENCODE_MODEL", "")


def build_fsm_config() -> dict:
    """从环境变量构建 FSM 配置字典。"""
    return {
        "enabled": get_fsm_enabled(),
        "target_repo": get_fsm_target_repo(),
        "deploy_engine": get_fsm_deploy_engine(),
        "repair_iters": get_fsm_repair_iters(),
        "mode": os.environ.get("FSM_MODE", "smoke"),
        "opencode_url": get_opencode_url(),
        "opencode_model": get_opencode_model(),
    }


def make_run_dir(benchmark: str, base: str = "runs") -> str:
    """生成带时间戳的运行目录路径。"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = Path(base) / f"{benchmark}_{ts}"
    return str(p.resolve())
