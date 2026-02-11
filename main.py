#!/usr/bin/env python3
"""
OpenHands RL Post-training Agent

简单实现：让 OpenHands Agent 自主完成 RL 后训练任务。
Agent 会自己读取 workspace 里的 instructions.txt 和 description.md。
"""

import argparse
import os

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool


def create_agent(llm: LLM, condenser_llm: LLM | None = None) -> Agent:
    """创建 Agent
    
    Args:
        llm: 主 LLM，用于 agent 推理
        condenser_llm: 用于上下文压缩的 LLM（可选，默认复用主 LLM）
    """
    # 创建 condenser 来处理长时间运行任务的上下文压缩
    # 官方推荐配置：LLMSummarizingCondenser
    # 
    # 触发条件（满足任一即触发）：
    # 1. max_tokens: token 数量超过限制（主要限制）
    # 2. max_size: 事件数量超过限制（次要限制）
    condenser = LLMSummarizingCondenser(
        llm=condenser_llm or llm,  # 复用主 LLM 或使用独立的便宜模型
        max_size=100,       # 超过 100 个事件时触发压缩
        keep_first=4,       # 保留前 4 个事件（任务描述等）
        max_tokens=30000,   # 超过 30K tokens 时触发压缩（注意：这是 messages token，不是 API 报告的 reasoning token）
    )
    
    return Agent(
        llm=llm,
        tools=[
            Tool(name=FileEditorTool.name),
            Tool(name=TerminalTool.name, params={"no_change_timeout_seconds": 600}),
        ],
        condenser=condenser,  # 启用上下文压缩
    )


def build_task_prompt(
    task: str,
    base_model: str,
    workspace: str,
) -> str:
    """构建任务 Prompt - 让 agent 自主完成任务"""
    return f"""你是一个 RL 后训练专家。你的任务是自主完成模型训练并提交评测。

## 重要：自主执行
- 你必须自主完成所有步骤，不要等待用户确认
- 遇到问题自己解决，不要询问用户
- 完成训练并提交评测后，调用 finish 工具结束

## 安全限制（严格遵守）
- 只能在 Workspace 目录下操作：{workspace}
- 禁止读写 Workspace 以外的任何文件
- 禁止修改环境：不允许 pip install、conda install 或任何包管理命令
- 所有依赖已预装在 openhands 环境中，直接使用即可

## 运行环境
- Conda 环境：openhands（已激活，无需手动激活）
- 预装库：transformers, trl, torch, vllm, datasets, accelerate, peft 等

## 目录结构
- 代码区: {workspace}/code/ — 所有自行编写的代码放在此处
- 训练数据: {workspace}/data/（只读）
- 基础模型: {workspace}/models/（只读，{base_model}）
- 模型输出: {workspace}/output/ — 训练好的模型保存在此
- 任务说明: {workspace}/instructions.md, {workspace}/description.md

## 任务流程
1. 阅读 {workspace}/instructions.md 和 {workspace}/description.md 了解任务
2. 在 {workspace}/code/ 下编写训练脚本（使用 trl 库的 GRPO）
3. 执行训练，模型保存到 {workspace}/output/
4. 提交评测：POST $GRADING_SERVER_URL/submit，指定 model_path
5. 根据 score 反馈迭代优化，可多次提交不同版本
6. 调用 finish 结束

现在开始执行，不要等待用户输入。
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
    
    # 获取 LLM 配置
    llm_api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    llm_model = os.environ.get("LLM_MODEL", "gpt-5")  # 默认 gpt-5，和 sft 一致
    llm_base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
    
    if not llm_api_key:
        print("ERROR: LLM_API_KEY or OPENAI_API_KEY not set")
        return
    
    print(f"LLM Model: {llm_model}")
    
    # 创建 LLM
    # 重要：设置 model_canonical_name="gpt-4" 来禁用 Responses API
    # gpt-5 在 RESPONSES_API_MODELS 中，会触发 /responses endpoint 调用
    # 但我们的 API proxy 的 /responses endpoint 有兼容性问题
    # 通过设置 model_canonical_name 为非 gpt-5 模型，强制使用 completion API
    llm_kwargs = {
        "model": llm_model,
        "api_key": SecretStr(llm_api_key),
        "prompt_cache_retention": None,  # 禁用 Azure 不支持的 prompt cache retention
        "model_canonical_name": "gpt-4",  # 禁用 Responses API，使用 completion API
    }
    if llm_base_url:
        llm_kwargs["base_url"] = llm_base_url
    
    llm = LLM(**llm_kwargs)
    
    # 创建 Agent（带 LLM Summarizing Condenser 处理长上下文）
    print(f"Condenser: LLMSummarizingCondenser (max_tokens=30K, max_size=100, keep_first=4)")
    agent = create_agent(llm, condenser_llm=llm)
    
    # 创建 Conversation
    conv = Conversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=max_iterations,
    )
    
    # 构建任务 Prompt（简化版，让 agent 自己读取文档）
    task_prompt = build_task_prompt(
        task=task,
        base_model=base_model,
        workspace=workspace,
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
