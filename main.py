#!/usr/bin/env python3
"""
OpenHands RL Post-training Pipeline (Fixed-Stage)

参考 openhands-magic SFT pipeline 的固定阶段式设计。
每轮迭代：代码生成 → 训练执行 → 评测提交 → 反馈注入。

与旧版"一个大 prompt 自由发挥"的区别：
- pipeline 控制流程，agent 只负责写代码
- 训练和评测由 pipeline 执行，不依赖 agent
- 每轮 prompt 包含上轮结果，精准引导迭代优化
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    """一轮迭代的完整结果"""
    iteration: int
    exit_code: int = -1
    stdout: str = ""
    training_time: float = 0.0
    score: float | None = None
    improvement: float | None = None
    best_score: float | None = None
    model_path: str = ""
    code_path: str = ""


# ---------------------------------------------------------------------------
# LLM & Agent
# ---------------------------------------------------------------------------

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
        "model_canonical_name": "gpt-4",  # 强制使用 completion API
    }
    if base_url:
        kwargs["base_url"] = base_url

    return LLM(**kwargs)


def create_code_agent(llm: LLM) -> Agent:
    """创建代码生成 Agent

    工具：FileEditor（写代码）+ Terminal（探索数据，不执行训练）
    """
    return Agent(
        llm=llm,
        tools=[
            Tool(name=FileEditorTool.name),
            Tool(name=TerminalTool.name, params={"no_change_timeout_seconds": 120}),
        ],
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_code_prompt(
    iteration: int,
    workspace: str,
    base_model: str,
    task_description: str,
    history: list[IterationResult],
) -> str:
    """构建代码生成阶段的 prompt

    关键设计：
    - 第 1 轮：让 agent 探索数据格式 + 写初版代码
    - 后续轮：注入上轮结果（score/error），引导定向优化
    """
    model_path = os.environ.get("MODEL_PATH", "")
    data_path = os.environ.get("DATA_PATH", "")
    output_dir = os.environ.get("OUTPUT_DIR", "")

    # ---- 历史结果表格 ----
    history_section = ""
    if history:
        rows = []
        for h in history:
            score_s = f"{h.score:.2f}" if h.score is not None else "-"
            imp_s = f"{h.improvement:+.2f}" if h.improvement is not None else "-"
            status = "OK" if h.exit_code == 0 else f"FAIL({h.exit_code})"
            rows.append(f"| {h.iteration} | {status} | {h.training_time:.0f}s | {score_s} | {imp_s} |")

        history_section += "\n## 历史实验结果\n"
        history_section += "| 轮次 | 状态 | 耗时 | Score | vs Baseline |\n"
        history_section += "|------|------|------|-------|-------------|\n"
        history_section += "\n".join(rows) + "\n"

        # 上轮详细信息
        last = history[-1]
        if last.exit_code != 0 and last.stdout:
            # 训练失败 → 给出错误日志
            tail = "\n".join(last.stdout.strip().splitlines()[-40:])
            history_section += f"\n### 上轮错误日志（最后 40 行）\n```\n{tail}\n```\n"
            history_section += "\n**请根据错误信息修复代码。**\n"
        elif last.score is not None:
            history_section += f"\n### 上轮评测\n- Score: {last.score}\n- Improvement: {last.improvement}\n"
            history_section += "\n**请调整策略（reward 函数、超参数等）提升 score。**\n"
        elif last.exit_code == 0:
            history_section += "\n### 上轮训练成功但无评测结果\n"
            history_section += "可能原因：模型未保存到 $OUTPUT_DIR，或输出目录为空。\n"
            history_section += "**请确保代码中有 `trainer.save_model(OUTPUT_DIR)` 或等效保存操作。**\n"

    # ---- 上轮代码 ----
    prev_code_section = ""
    if history:
        code_file = Path(history[-1].code_path)
        if code_file.exists():
            code_text = code_file.read_text()
            if len(code_text) > 4000:
                code_text = code_text[:4000] + "\n# ... (truncated)"
            prev_code_section = f"\n## 上轮代码\n```python\n{code_text}\n```\n"

    # ---- 主 prompt ----
    if iteration == 1:
        task_instruction = f"""## 你的任务（第 1 轮：链路打通）
1. 用 terminal 探索数据格式：`head -5 {data_path}/train.jsonl` 或 `ls {data_path}/`
2. 阅读 {workspace}/description.md 了解任务要求
3. 在 {workspace}/code/ 下编写 train.py
4. 代码要求：
   - 路径通过 os.environ 获取（MODEL_PATH, DATA_PATH, OUTPUT_DIR）
   - 使用 trl 的 GRPOTrainer（推荐），设计合适的 reward 函数
   - 训练完成后保存模型到 $OUTPUT_DIR
   - 合理设置参数：小 batch、少步数，先验证链路能跑通
5. 完成后调用 finish 工具结束

**重要**：你只负责写代码，不要自己执行训练脚本。pipeline 会自动运行。"""
    else:
        task_instruction = f"""## 你的任务（第 {iteration} 轮：迭代优化）
1. 分析上轮结果（见历史实验和错误日志）
2. 修改 {workspace}/code/train.py
3. 改进方向：
   - 如果上轮失败：修复错误
   - 如果 score 为空：确保模型保存到 $OUTPUT_DIR
   - 如果有 score：优化 reward 函数、调整超参数、尝试不同策略
4. 完成后调用 finish 工具结束

**重要**：你只负责写代码，不要自己执行训练脚本。pipeline 会自动运行。"""

    return f"""你是 RL 后训练专家。

## 安全限制
- 只能在 {workspace} 内操作
- 禁止 pip install 或任何包管理命令
- 预装库：transformers, trl, torch, vllm, datasets, accelerate, peft

## 目录结构
- 代码区: {workspace}/code/
- 训练数据: {data_path}（只读）
- 基础模型: {model_path}（只读，{base_model}）
- 模型输出: {output_dir}

## 环境变量（代码中用 os.environ 读取）
- MODEL_PATH={model_path}
- DATA_PATH={data_path}
- OUTPUT_DIR={output_dir}

## 任务描述
{task_description}
{history_section}{prev_code_section}{task_instruction}"""


# ---------------------------------------------------------------------------
# Pipeline Phases
# ---------------------------------------------------------------------------

def phase_code_generation(
    llm: LLM,
    iteration: int,
    workspace: str,
    base_model: str,
    task_description: str,
    history: list[IterationResult],
    max_agent_steps: int = 25,
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

    在 $WORKSPACE/code/ 下执行 train.py，捕获输出。
    """
    print(f"\n{'='*60}")
    print(f"  Phase 2: Training Execution")
    print(f"{'='*60}")

    if not Path(code_path).exists():
        msg = f"Code file not found: {code_path}"
        print(f"  ERROR: {msg}")
        return -1, msg, 0.0

    code_dir = Path(workspace) / "code"
    start = time.time()

    try:
        proc = subprocess.run(
            ["python", str(code_path)],
            cwd=str(code_dir),
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
        # 打印最后几行错误
        tail = "\n".join(stdout.strip().splitlines()[-10:])
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

    # 找最新模型目录
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


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    task: str,
    base_model: str,
    workspace: str,
    max_iterations: int = 5,
    training_timeout: int = 3600,
    max_agent_steps: int = 25,
):
    """运行固定阶段 Pipeline

    每轮迭代：
      Phase 1: Agent 写代码（探索数据 + 编写 train.py）
      Phase 2: Pipeline 执行训练（subprocess）
      Phase 3: Pipeline 提交评测（HTTP POST）
      Phase 4: 记录结果，注入到下轮 prompt
    """
    pipeline_start = time.time()
    grading_url = os.environ.get("GRADING_SERVER_URL", "http://localhost:5000")

    print(f"{'#'*60}")
    print(f"  OpenHands RL Pipeline (Fixed-Stage)")
    print(f"{'#'*60}")
    print(f"  Task: {task}")
    print(f"  Model: {base_model}")
    print(f"  Workspace: {workspace}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Training timeout: {training_timeout}s")
    print(f"  Agent steps/iter: {max_agent_steps}")
    print(f"  Grading Server: {grading_url}")
    print(f"  LLM: {os.environ.get('LLM_MODEL', 'gpt-4.1')}")
    print()

    # 读取任务描述（pipeline 读取，注入到 prompt，省去 agent 自己 cat）
    task_description = ""
    for fname in ["description.md", "instructions.md"]:
        fpath = Path(workspace) / fname
        if fpath.exists():
            task_description += fpath.read_text() + "\n\n"

    if not task_description.strip():
        print("WARNING: No description.md or instructions.md found in workspace")

    # 创建 LLM（所有迭代复用）
    llm = create_llm()
    print(f"  LLM initialized: {llm.model}\n")

    history: list[IterationResult] = []
    best_score: float | None = None
    best_iteration = -1

    for iteration in range(1, max_iterations + 1):
        iter_start = time.time()
        elapsed_total = iter_start - pipeline_start

        print(f"\n{'#'*60}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"  Elapsed: {elapsed_total:.0f}s")
        print(f"{'#'*60}")

        result = IterationResult(iteration=iteration)

        # Phase 1: Code Generation
        try:
            code_path = phase_code_generation(
                llm, iteration, workspace, base_model,
                task_description, history, max_agent_steps,
            )
            result.code_path = code_path
        except Exception as e:
            print(f"  Code generation failed: {e}")
            result.code_path = str(Path(workspace) / "code" / "train.py")
            result.exit_code = -1
            result.stdout = f"Code generation error: {e}"
            history.append(result)
            continue

        # Phase 2: Training Execution
        exit_code, stdout, train_time = phase_training(
            workspace, code_path, timeout=training_timeout,
        )
        result.exit_code = exit_code
        result.stdout = stdout
        result.training_time = train_time

        # Phase 3: Evaluation (only if training succeeded)
        if exit_code == 0:
            eval_result = phase_evaluation(workspace, grading_url)
            if eval_result:
                result.score = eval_result.get("score")
                result.improvement = eval_result.get("improvement")
                result.best_score = eval_result.get("best", {}).get("score")
                result.model_path = eval_result.get("model_path", "")

                if best_score is None or (result.score is not None and result.score > best_score):
                    best_score = result.score
                    best_iteration = iteration

        # Phase 4: Record
        history.append(result)

        iter_elapsed = time.time() - iter_start
        print(f"\n  --- Iteration {iteration} Summary ---")
        print(f"  Exit code: {result.exit_code}")
        print(f"  Score: {result.score}")
        print(f"  Best so far: {best_score} (iter {best_iteration})")
        print(f"  Iteration time: {iter_elapsed:.0f}s")

    # Pipeline 完成
    total_time = time.time() - pipeline_start
    print(f"\n{'#'*60}")
    print(f"  Pipeline Complete")
    print(f"  Best score: {best_score}")
    print(f"  Best iteration: {best_iteration}")
    print(f"  Total iterations: {len(history)}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"{'#'*60}")

    # 保存结果摘要
    summary = {
        "task": task,
        "base_model": base_model,
        "best_score": best_score,
        "best_iteration": best_iteration,
        "total_time": total_time,
        "iterations": [
            {
                "iteration": r.iteration,
                "exit_code": r.exit_code,
                "training_time": r.training_time,
                "score": r.score,
                "improvement": r.improvement,
            }
            for r in history
        ],
    }
    summary_path = Path(workspace) / "pipeline_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  Results saved: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenHands RL Pipeline (Fixed-Stage)")
    parser.add_argument("--benchmark", type=str, default="gsm8k")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Pipeline 迭代次数（每轮=写代码+训练+评测）")
    parser.add_argument("--training-timeout", type=int, default=3600,
                        help="每轮训练的超时时间（秒）")
    parser.add_argument("--max-agent-steps", type=int, default=25,
                        help="每轮代码生成阶段 agent 的最大步数")
    args = parser.parse_args()

    run_pipeline(
        task=args.benchmark,
        base_model=args.base_model,
        workspace=args.workspace,
        max_iterations=args.max_iterations,
        training_timeout=args.training_timeout,
        max_agent_steps=args.max_agent_steps,
    )


if __name__ == "__main__":
    main()
