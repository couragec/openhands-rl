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
# Helpers
# ---------------------------------------------------------------------------

def get_data_stats(data_path: str) -> dict:
    """获取数据目录信息。优先尝试解析 train.jsonl，否则列出目录文件。"""
    path = Path(data_path)
    if not path.exists():
        return {"count": 0, "files": [], "type": "missing"}

    jsonl = path / "train.jsonl" if path.is_dir() else path
    if jsonl.exists() and jsonl.suffix == ".jsonl":
        samples = []
        try:
            with open(jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        except Exception:
            pass
        if samples:
            prompt_lens, answer_lens = [], []
            for s in samples:
                prompt = s.get("prompt") or s.get("question") or s.get("instruction") or ""
                answer = s.get("answer") or s.get("response") or s.get("output") or ""
                prompt_lens.append(len(prompt))
                answer_lens.append(len(answer))
            return {
                "count": len(samples),
                "avg_prompt_len": sum(prompt_lens) // max(len(prompt_lens), 1),
                "avg_answer_len": sum(answer_lens) // max(len(answer_lens), 1),
                "type": "jsonl",
                "files": [jsonl.name],
            }

    if path.is_dir():
        files = sorted(f.name for f in path.iterdir() if not f.name.startswith("."))
        return {"count": 0, "files": files, "type": "directory"}

    return {"count": 0, "files": [path.name], "type": "file"}


def load_data_preview(data_path: str, num_samples: int = 3) -> str:
    """返回数据预览。支持 jsonl 文件和普通目录。"""
    path = Path(data_path)

    if not path.exists():
        return "（数据路径不存在）"

    jsonl = path / "train.jsonl" if path.is_dir() else path
    if jsonl.exists() and jsonl.suffix == ".jsonl":
        records = []
        try:
            with open(jsonl, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    if line.strip():
                        records.append(json.loads(line))
        except Exception as e:
            return f"（读取失败: {e}）"
        return json.dumps(records, ensure_ascii=False, indent=2)

    if path.is_dir():
        files = sorted(f.name for f in path.iterdir() if not f.name.startswith("."))
        return f"数据目录包含文件: {', '.join(files[:20])}"

    return "（无法预览）"


def get_gpu_info() -> dict:
    """获取 GPU 数量和型号。"""
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    num_gpus = len(cuda_devices.split(",")) if cuda_devices else 0

    gpu_name = "Unknown"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            names = [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
            if names:
                gpu_name = names[0]
    except Exception:
        pass

    return {"num_gpus": num_gpus, "gpu_name": gpu_name, "cuda_devices": cuda_devices}


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
    analysis: str = ""


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
    data_stats: dict | None = None,
    gpu_info: dict | None = None,
) -> str:
    """构建代码生成阶段的 prompt

    关键设计：
    - 第 1 轮：给 GRPO 参考代码 + 数据统计，让 agent 快速写出能跑的代码
    - 后续轮：注入上轮结果（score/error），引导定向优化
    """
    model_path = os.environ.get("MODEL_PATH", "")
    data_path = os.environ.get("DATA_PATH", "")
    output_dir = os.environ.get("OUTPUT_DIR", "")

    # ---- GPU 信息 ----
    gpu_section = ""
    if gpu_info:
        gpu_section = f"""
## 硬件环境
- GPU: {gpu_info['num_gpus']}x {gpu_info['gpu_name']}
- CUDA_VISIBLE_DEVICES={gpu_info['cuda_devices']}
- Pipeline 会用 `accelerate launch` 执行你的代码，自动启用多卡 DDP，你不需要手动处理分布式
- 参数建议：{gpu_info['num_gpus']} 卡时 per_device_train_batch_size=4, gradient_accumulation_steps=2
"""

    # ---- 数据统计 ----
    data_stats_section = ""
    if data_stats:
        ds_type = data_stats.get("type", "")
        if ds_type == "jsonl" and data_stats.get("count", 0) > 0:
            data_stats_section = f"""
## 数据统计
- 样本数: {data_stats['count']}
- 平均 prompt 长度: {data_stats['avg_prompt_len']} 字符
- 平均 answer 长度: {data_stats['avg_answer_len']} 字符
"""
        elif ds_type == "directory" and data_stats.get("files"):
            files_list = ", ".join(data_stats["files"][:15])
            data_stats_section = f"""
## 数据目录
- 文件: {files_list}
- 提示: 用 ls/cat 探索具体文件内容
"""

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
            tail = "\n".join(last.stdout.strip().splitlines()[-60:])
            history_section += f"\n### 上轮错误日志（最后 60 行）\n```\n{tail}\n```\n"
            history_section += "\n**请根据错误信息修复代码。**\n"
        elif last.score is not None:
            history_section += f"\n### 上轮评测\n- Score: {last.score}\n- Improvement: {last.improvement}\n"
            history_section += "\n**请调整策略（reward 函数、超参数等）提升 score。**\n"
        elif last.exit_code == 0:
            history_section += "\n### 上轮训练成功但无评测结果\n"
            history_section += "可能原因：模型未保存到 $OUTPUT_DIR，或输出目录为空。\n"
            history_section += "**请确保代码中有 `trainer.save_model(OUTPUT_DIR)` 或等效保存操作。**\n"

        # 上轮自分析报告（跨轮次记忆）
        if last.analysis and last.analysis.strip():
            analysis_text = last.analysis.strip()
            if len(analysis_text) > 3000:
                analysis_text = analysis_text[:3000] + "\n... (截断)"
            history_section += f"\n### 上轮自分析报告\n{analysis_text}\n"

    # ---- 上轮代码 ----
    prev_code_section = ""
    if history:
        code_file = Path(history[-1].code_path)
        if code_file.exists():
            code_text = code_file.read_text()
            if len(code_text) > 4000:
                code_text = code_text[:4000] + "\n# ... (truncated)"
            prev_code_section = f"\n## 上轮代码\n```python\n{code_text}\n```\n"

    # ---- GRPO 参考模板（仅第 1 轮附带） ----
    grpo_reference = """
## GRPO 参考代码（TRL 0.27+）

以下是 GRPO RL 训练的参考模板，根据任务实际情况调整 reward 函数和数据加载逻辑：

```python
import os, json, re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]

# 1. 加载数据 → Dataset 格式，必须有 "prompt" 字段
with open(os.path.join(DATA_PATH, "train.jsonl")) as f:
    raw = [json.loads(l) for l in f if l.strip()]

prompts = []
for item in raw:
    question = item.get("question") or item.get("prompt") or item.get("instruction")
    prompts.append([{{"role": "user", "content": question}}])

dataset = Dataset.from_dict({{"prompt": prompts}})

# 2. Reward 函数（根据任务替换）
def reward_fn(completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        if isinstance(comp, list):
            text = comp[-1]["content"] if comp else ""
        else:
            text = str(comp)
        rewards.append(1.0 if re.search(r"\\\\d+", text) else 0.0)
    return rewards

# 3. 训练
config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=8,
    learning_rate=5e-6,
    max_completion_length=256,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    report_to="none",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

**GRPO 注意事项**：
- `GRPOTrainer(model, reward_funcs, args, train_dataset, processing_class=tokenizer)` — 注意是 `processing_class` 不是 `tokenizer`
- prompt 必须是 chat 格式（list of dict），completions 也是 chat 格式
- `num_generations` 必须能被 `per_device_train_batch_size * num_gpus` 整除

**注意**：此模板适用于有静态数据集的任务。交互式环境任务（如 ALFWorld）需参考 eval.py 设计 rollout + reward 流程，但训练框架（GRPO/PPO）思路相同。
"""

    # ---- 主 prompt ----
    if iteration == 1:
        task_instruction = f"""## 你的任务（第 1 轮：理解任务 + 编写训练代码）

1. **探索工作区**：`ls` 查看当前目录所有可用文件
2. **阅读任务描述**：`cat description.md`
3. **阅读评测代码**（如果有 eval.py）：`cat eval.py` — 包含环境交互、模型推理、reward 计算的完整逻辑，对理解任务至关重要
4. **探索数据**：`ls data/` 查看数据格式，用 head/cat 查看内容
5. **编写 train.py**：在 `code/` 下编写训练脚本
   - 路径通过环境变量获取：MODEL_PATH, DATA_PATH, OUTPUT_DIR
   - 训练方式：SFT、GRPO、PPO 等均可，最终目标是 RL post-training
   - 训练完成后保存模型到 $OUTPUT_DIR
6. 建议：先用少量数据验证能跑通，后续轮再全量训练
7. 完成后调用 finish 工具结束

**重要**：你只负责写代码，不要自己执行训练脚本。pipeline 会自动运行。
{grpo_reference}"""
    else:
        task_instruction = f"""## 你的任务（第 {iteration} 轮：迭代优化）
1. 分析上轮结果（见历史实验和错误日志）
2. 修改 `code/train.py`
3. 改进方向：
   - 如果上轮失败：修复错误
   - 如果 score 为空：确保模型保存到 $OUTPUT_DIR
   - 如果有 score：尝试不同训练策略（reward 函数、超参数、rollout 设计等）提升分数,最终目标是 RL post-training
4. 完成后调用 finish 工具结束

**重要**：你只负责写代码，不要自己执行训练脚本。pipeline 会自动运行。"""

    return f"""你是 RL 后训练专家。

## 工作区规则
- 当前目录就是你的工作区，所有需要的文件都在这里
- **禁止 `cd` 到当前目录之外**（不要访问父目录或其他路径）
- **只使用相对路径**（如 `./code/train.py`、`./data/`）
- 如果看到 symlink 指向外部路径，忽略它——直接用相对路径访问
{gpu_section}
## 目录结构
- 代码区: `./code/`
- 训练数据: `./data/`（只读）
- 基础模型: `./models/{base_model}`（只读）
- 模型输出: `./output/`

## 环境变量（训练脚本中用 os.environ 读取，pipeline 自动设置）
- MODEL_PATH — 基础模型路径（等价于 `./models/{base_model}`）
- DATA_PATH — 训练数据路径（等价于 `./data/`）
- OUTPUT_DIR — 模型输出路径（等价于 `./output/`）
- CUDA_VISIBLE_DEVICES={gpu_info['cuda_devices'] if gpu_info else ''}
{data_stats_section}
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
    data_stats: dict | None = None,
    gpu_info: dict | None = None,
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
        data_stats=data_stats, gpu_info=gpu_info,
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

    code_dir = Path(workspace) / "code"
    start = time.time()

    # 用训练环境的 Python 执行（TRAINING_PYTHON 由 start.sh 设置，指向 cwy-rl 环境）
    training_python = os.environ.get("TRAINING_PYTHON", "python")
    cmd = [training_python, "-m", "accelerate.commands.launch", str(code_path)]
    print(f"  CMD: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
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


def build_fix_prompt(
    code_path: str,
    error_log: str,
    data_path: str,
    workspace: str,
) -> str:
    """构造训练失败后的修复 prompt（开新 Conversation 用）。"""
    code_text = ""
    if Path(code_path).exists():
        code_text = Path(code_path).read_text()
        if len(code_text) > 6000:
            code_text = code_text[:6000] + "\n# ... (truncated)"

    data_preview = load_data_preview(data_path, num_samples=3)
    error_tail = "\n".join(error_log.strip().splitlines()[-100:]) if error_log else "（无日志）"

    return f"""训练脚本执行失败，请分析错误并修复。

## 错误日志（最后 100 行）
```
{error_tail}
```

## 当前 train.py
```python
{code_text}
```

## 数据预览（前 3 条）
```json
{data_preview}
```

## 要求
1. 仔细分析错误原因
2. 修复 `code/train.py`
3. 使用 file_editor 工具保存修复后的代码
4. 不要执行脚本，pipeline 会自动运行
5. 完成后调用 finish 工具结束
"""


def build_analysis_prompt(
    iteration: int,
    workspace: str,
    result: "IterationResult",
    best_score: float | None,
    best_iteration: int,
) -> str:
    """构造 Analysis 阶段的 prompt，让 agent 总结本轮并规划下一步。"""
    score_info = f"Score: {result.score}" if result.score is not None else "未获得分数"
    if result.exit_code != 0:
        status = f"训练失败（exit code {result.exit_code}）"
    elif result.score is not None:
        status = f"训练成功，{score_info}"
    else:
        status = "训练成功但未获得评测分数"

    best_info = f"历史最佳: {best_score}（第 {best_iteration} 轮）" if best_score is not None else "暂无历史分数"

    code_text = ""
    code_file = Path(result.code_path)
    if code_file.exists():
        code_text = code_file.read_text()
        if len(code_text) > 4000:
            code_text = code_text[:4000] + "\n# ... (truncated)"

    error_section = ""
    if result.exit_code != 0 and result.stdout:
        tail = "\n".join(result.stdout.strip().splitlines()[-40:])
        error_section = f"\n## 错误日志（最后 40 行）\n```\n{tail}\n```\n"

    return f"""你是 RL 训练专家。请分析第 {iteration} 轮实验的结果，写一份简短的诊断报告。

## 本轮状态
- {status}
- 训练耗时: {result.training_time:.0f}s
- {best_info}

## 本轮代码
```python
{code_text}
```
{error_section}
## 请写一份简短分析（200-400字），包含：
1. **做了什么**：本轮训练方法、关键配置
2. **结果分析**：为什么成功/失败/分数高低
3. **下一步计划**：具体要改什么、尝试什么策略

直接输出分析文本，不需要写代码。完成后调用 finish 工具结束。
"""


def phase_analysis(
    llm: LLM,
    iteration: int,
    workspace: str,
    result: "IterationResult",
    best_score: float | None,
    best_iteration: int,
) -> str:
    """Phase 4: Analysis — 让 agent 总结本轮并规划下一步。

    返回 analysis 文本。
    """
    print(f"\n{'='*60}")
    print(f"  Phase 4: Analysis (iteration {iteration})")
    print(f"{'='*60}")

    agent = create_code_agent(llm)
    conv = Conversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=10,
    )

    prompt = build_analysis_prompt(
        iteration, workspace, result, best_score, best_iteration,
    )
    conv.send_message(prompt)
    conv.run()

    analysis_file = Path(workspace) / "analysis.md"
    if analysis_file.exists():
        analysis_text = analysis_file.read_text().strip()
        if analysis_text:
            print(f"  Analysis saved: {len(analysis_text)} chars")
            return analysis_text

    last_msg = ""
    if hasattr(conv, "messages") and conv.messages:
        for msg in reversed(conv.messages):
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if content and len(content) > 50:
                last_msg = content
                break

    if last_msg:
        analysis_file.write_text(last_msg)
        print(f"  Analysis extracted from conversation: {len(last_msg)} chars")
        return last_msg

    print("  WARNING: No analysis produced")
    return ""


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
      Phase 4: Agent 自分析（总结本轮 + 规划下一步，作为跨轮记忆）
      Phase 5: 记录结果，注入到下轮 prompt
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

    # 预计算数据统计和 GPU 信息
    data_path = os.environ.get("DATA_PATH", "")
    data_stats = get_data_stats(data_path)
    gpu_info = get_gpu_info()
    ds_info = f"{data_stats['count']} samples" if data_stats.get("count") else f"{len(data_stats.get('files', []))} files ({data_stats.get('type', '')})"
    print(f"  Data: {ds_info}")
    print(f"  GPU: {gpu_info['num_gpus']}x {gpu_info['gpu_name']}")

    # 创建 LLM（所有迭代复用）
    llm = create_llm()
    print(f"  LLM initialized: {llm.model}\n")

    history: list[IterationResult] = []
    best_score: float | None = None
    best_iteration = -1
    max_fix_retries = 3  # 训练失败后最多重试次数

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
                data_stats=data_stats, gpu_info=gpu_info,
            )
            result.code_path = code_path
        except Exception as e:
            print(f"  Code generation failed: {e}")
            result.code_path = str(Path(workspace) / "code" / "train.py")
            result.exit_code = -1
            result.stdout = f"Code generation error: {e}"
            history.append(result)
            continue

        # Phase 2: Training Execution（含重试）
        exit_code, stdout, train_time = phase_training(
            workspace, code_path, timeout=training_timeout,
        )

        for retry in range(max_fix_retries):
            if exit_code == 0:
                break
            print(f"\n  --- Fix retry {retry + 1}/{max_fix_retries} ---")
            try:
                fix_prompt = build_fix_prompt(code_path, stdout, data_path, workspace)
                fix_agent = create_code_agent(create_llm())
                fix_conv = Conversation(
                    agent=fix_agent, workspace=workspace,
                    max_iteration_per_run=15,
                )
                fix_conv.send_message(fix_prompt)
                fix_conv.run()
                print(f"  Agent fix attempt done, re-running training...")
            except Exception as e:
                print(f"  Fix agent failed: {e}")
                break

            exit_code, stdout, extra_time = phase_training(
                workspace, code_path, timeout=training_timeout,
            )
            train_time += extra_time

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

        # Phase 4: Analysis（让 agent 总结本轮，生成跨轮记忆）
        try:
            analysis_text = phase_analysis(
                llm, iteration, workspace, result, best_score, best_iteration,
            )
            result.analysis = analysis_text
        except Exception as e:
            print(f"  Analysis failed (non-fatal): {e}")

        # Phase 5: Record
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
                "analysis": r.analysis[:500] if r.analysis else "",
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
