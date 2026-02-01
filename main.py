#!/usr/bin/env python3
"""
OpenHands RL Post-training Agent

简单实现：让 OpenHands Agent 自主完成 RL 后训练任务。
参考 openhands-magic 的模式。
"""

import argparse
import os
from pathlib import Path

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool


def load_task_description(workspace: str, task: str) -> str:
    """加载任务描述（description.md）"""
    # 查找 description.md
    search_paths = [
        Path(workspace) / "description.md",
        Path(workspace).parent / "tasks" / task / "description.md",
    ]
    
    # 也检查 autorl_bench 的 tasks 目录
    autorl_bench_path = Path(__file__).parent.parent.parent / "RD-Agent" / "rdagent" / "scenarios" / "rl" / "autorl_bench" / "tasks" / task / "description.md"
    if autorl_bench_path.exists():
        search_paths.insert(0, autorl_bench_path)
    
    for path in search_paths:
        if path.exists():
            return path.read_text()
    
    return ""


def load_data_preview(data_path: str, num_lines: int = 5) -> str:
    """加载数据预览"""
    import json
    
    path = Path(data_path)
    if not path.exists():
        # 尝试在 data 目录下查找
        return f"数据路径: {data_path}"
    
    try:
        lines = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                if line.strip():
                    lines.append(json.loads(line))
        return json.dumps(lines, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"无法读取数据: {e}"


def create_agent(llm: LLM) -> Agent:
    """创建 Agent"""
    return Agent(
        llm=llm,
        tools=[
            Tool(name=FileEditorTool.name),
            Tool(name=TerminalTool.name, params={"no_change_timeout_seconds": 600}),
        ],
    )


def build_task_prompt(
    task: str,
    base_model: str,
    workspace: str,
    task_description: str,
    data_preview: str,
    grading_server_url: str,
) -> str:
    """构建任务 Prompt"""
    return f"""你是一个 RL 后训练专家。请完成以下任务：

## 任务信息
- Benchmark: {task}
- 基础模型: {base_model}
- Workspace: {workspace}
- 数据目录: {workspace}/data/{task}
- 模型目录: {workspace}/models/{base_model}
- 输出目录: {workspace}/output

## 任务描述
{task_description if task_description else "请分析数据格式并设计训练策略。"}

## 数据预览
```json
{data_preview}
```

## 评测说明
训练完成后，请将模型保存到 {workspace}/output 目录。
评测会通过 Grading Server ({grading_server_url}) 自动进行。

你可以通过 HTTP 请求提交评测：
```bash
curl -X POST {grading_server_url}/submit -H "Content-Type: application/json" -d '{{"model_path": "{workspace}/output"}}'
```

## 训练框架
使用 trl 库 (版本 >= 0.27.0)，推荐 GRPO 算法：

```python
from trl import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=GRPOConfig(...),
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
model.save_pretrained("./output")
tokenizer.save_pretrained("./output")
```

## 任务流程
1. 分析数据格式（查看 data 目录下的 train.jsonl）
2. 设计 reward function
3. 编写训练脚本
4. 执行训练
5. 保存模型到 output 目录
6. 提交评测

请开始执行任务。
"""


def run_pipeline(
    task: str,
    base_model: str,
    workspace: str,
    max_iterations: int,
):
    """运行 Pipeline"""
    print(f"=== OpenHands RL Pipeline ===")
    print(f"Task: {task}")
    print(f"Model: {base_model}")
    print(f"Workspace: {workspace}")
    
    # 获取配置
    llm_api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    llm_model = os.environ.get("LLM_MODEL", "gpt-4o")
    llm_base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
    grading_server_url = os.environ.get("GRADING_SERVER_URL", "http://localhost:5000")
    
    if not llm_api_key:
        print("ERROR: LLM_API_KEY or OPENAI_API_KEY not set")
        return
    
    print(f"LLM Model: {llm_model}")
    print(f"Grading Server: {grading_server_url}")
    
    # 加载任务描述
    task_description = load_task_description(workspace, task)
    if task_description:
        print(f"Loaded task description ({len(task_description)} chars)")
    
    # 加载数据预览
    data_path = Path(workspace) / "data" / task / "train.jsonl"
    if not data_path.exists():
        data_path = Path(workspace) / "data" / "train.jsonl"
    data_preview = load_data_preview(str(data_path))
    
    # 创建 LLM
    llm_kwargs = {
        "model": llm_model,
        "api_key": SecretStr(llm_api_key),
    }
    if llm_base_url:
        llm_kwargs["base_url"] = llm_base_url
    
    llm = LLM(**llm_kwargs)
    
    # 创建 Agent
    agent = create_agent(llm)
    
    # 创建 Conversation
    conv = Conversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=max_iterations,
    )
    
    # 构建任务 Prompt
    task_prompt = build_task_prompt(
        task=task,
        base_model=base_model,
        workspace=workspace,
        task_description=task_description,
        data_preview=data_preview,
        grading_server_url=grading_server_url,
    )
    
    print(f"\n--- Sending task to agent ---")
    conv.send_message(task_prompt)
    
    print(f"\n--- Running agent (max {max_iterations} iterations) ---")
    conv.run()
    
    print(f"\n=== Pipeline completed ===")


def main():
    parser = argparse.ArgumentParser(description="OpenHands RL Post-training Agent")
    parser.add_argument("--benchmark", type=str, default="gsm8k", help="Benchmark name")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--max-iterations", type=int, default=50)
    args = parser.parse_args()
    
    run_pipeline(
        task=args.benchmark,
        base_model=args.base_model,
        workspace=args.workspace,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
