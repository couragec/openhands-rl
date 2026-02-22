"""Pipeline 各阶段执行逻辑。"""

import os
import subprocess
import time
from pathlib import Path

import requests
from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

from .prompts import build_code_prompt, build_fix_prompt
from .types import IterationResult


def create_llm() -> LLM:
    """创建 LLM 实例（从环境变量读取配置）"""
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "gpt-4.1")
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")

    if not api_key:
        raise ValueError("LLM_API_KEY or OPENAI_API_KEY required")

    kwargs = {
        "model": model,
        "api_key": SecretStr(api_key),
        "prompt_cache_retention": None,
        "model_canonical_name": "gpt-4",
    }
    if base_url:
        kwargs["base_url"] = base_url

    return LLM(**kwargs)


def create_code_agent(llm: LLM) -> Agent:
    """创建代码生成 Agent"""
    return Agent(
        llm=llm,
        tools=[
            Tool(name=FileEditorTool.name),
            Tool(name=TerminalTool.name, params={"no_change_timeout_seconds": 120}),
        ],
    )


def phase_code_generation(
    llm: LLM,
    iteration: int,
    workspace: str,
    base_model: str,
    task_description: str,
    history: list[IterationResult],
    max_agent_steps: int = 25,
    data_stats: dict | None = None,
    gpu_info: dict | None = None,
    rollout_stats: dict | None = None,
) -> str:
    """Phase 1: 代码生成

    Agent 用 FileEditor + Terminal 探索数据并编写训练代码。
    返回代码文件路径。
    """
    print(f"\n{'='*60}")
    print(f"  Phase 1: Code Generation (iteration {iteration})")
    print(f"{'='*60}")

    agent = create_code_agent(llm)
    conv = Conversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=max_agent_steps,
    )

    prompt = build_code_prompt(
        iteration, workspace, base_model, task_description, history,
        data_stats=data_stats, gpu_info=gpu_info, rollout_stats=rollout_stats,
    )
    conv.send_message(prompt)
    conv.run()

    code_path = Path(workspace) / "code" / "train.py"
    if code_path.exists():
        print(f"  Code generated: {code_path} ({code_path.stat().st_size} bytes)")
    else:
        print(f"  WARNING: {code_path} not found after agent finished")

    return str(code_path)


def phase_training(
    workspace: str,
    code_path: str,
    timeout: int = 3600,
) -> tuple[int, str, float]:
    """Phase 2: 训练执行（pipeline 控制，非 agent）

    用 accelerate launch 执行 train.py，自动多卡 DDP。
    """
    print(f"\n{'='*60}")
    print(f"  Phase 2: Training Execution")
    print(f"{'='*60}")

    if not Path(code_path).exists():
        msg = f"Code file not found: {code_path}"
        print(f"  ERROR: {msg}")
        return -1, msg, 0.0

    abs_code_path = str(Path(code_path).resolve())
    start = time.time()

    cmd = ["accelerate", "launch", abs_code_path]
    print(f"  CMD: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path(workspace).resolve()),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        exit_code = proc.returncode
        stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        exit_code = -1
        stdout = f"Timeout after {timeout}s\n{e.stdout or ''}\n{e.stderr or ''}"
        print(f"  TIMEOUT after {timeout}s")

    elapsed = time.time() - start
    print(f"  Exit code: {exit_code}")
    print(f"  Time: {elapsed:.1f}s")

    if exit_code != 0:
        tail = "\n".join(stdout.strip().splitlines()[-15:])
        print(f"  Error tail:\n{tail}")

    return exit_code, stdout, elapsed


def phase_evaluation(
    workspace: str,
    grading_url: str,
) -> dict | None:
    """Phase 3: 评测提交（pipeline 控制，非 agent）

    找到 $OUTPUT_DIR 下最新模型，POST 到 Grading Server。
    """
    print(f"\n{'='*60}")
    print(f"  Phase 3: Evaluation")
    print(f"{'='*60}")

    output_dir = Path(os.environ.get("OUTPUT_DIR", str(Path(workspace) / "output")))
    if not output_dir.exists() or not any(output_dir.iterdir()):
        print("  No model output, skipping evaluation")
        return None

    subdirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if subdirs:
        model_path = str(max(subdirs, key=lambda d: d.stat().st_mtime))
    else:
        model_path = str(output_dir)

    print(f"  Submitting: {model_path}")

    try:
        resp = requests.post(
            f"{grading_url}/submit",
            json={"model_path": model_path},
            timeout=600,
        )
        result = resp.json()
        print(f"  Score: {result.get('score')}")
        print(f"  Improvement: {result.get('improvement')}")
        print(f"  Best: {result.get('best', {}).get('score')}")
        return result
    except Exception as e:
        print(f"  Evaluation failed: {e}")
        return None


def phase_fix_training(
    llm: LLM,
    code_path: str,
    stdout: str,
    data_path: str,
    workspace: str,
) -> None:
    """训练失败后的修复尝试。"""
    fix_prompt = build_fix_prompt(code_path, stdout, data_path, workspace)
    fix_agent = create_code_agent(create_llm())
    fix_conv = Conversation(
        agent=fix_agent, workspace=workspace,
        max_iteration_per_run=15,
    )
    fix_conv.send_message(fix_prompt)
    fix_conv.run()
