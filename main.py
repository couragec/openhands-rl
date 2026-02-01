#!/usr/bin/env python3
"""
OpenHands RL Post-training Pipeline

使用 OpenHands SDK 自动化 RL 后训练流程：
1. 分析任务和数据
2. 设计 reward function
3. 使用 trl 进行 GRPO/PPO 训练
4. 评测并迭代优化
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from litellm import completion
from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

from config import Config
from tools.trl_training import TRLTrainingTool
from tools.benchmark_eval import BenchmarkEvalTool


def create_agent(config: Config, llm: LLM) -> Agent:
    """创建配置好工具的 Agent"""
    tools = [
        Tool(name=FileEditorTool.name),
        Tool(name=TerminalTool.name),
        Tool(name=TRLTrainingTool.name),
        Tool(name=BenchmarkEvalTool.name),
    ]
    return Agent(llm=llm, tools=tools)


def build_system_prompt(config: Config) -> str:
    """构建系统 prompt"""
    return f"""你是一个 RL 后训练专家，负责提升模型在 {config.benchmark} benchmark 上的性能。

## 环境信息
- 基础模型: {config.base_model}
- 数据目录: {config.data_path}
- 输出目录: {config.output_path}
- Benchmark: {config.benchmark}

## 可用工具
1. FileEditorTool: 读写文件
2. TerminalTool: 执行命令
3. TRLTrainingTool: 使用 trl 库进行 RL 训练
4. BenchmarkEvalTool: 评测模型性能

## 训练框架
使用 trl 库 (版本 >= 0.27.0)，推荐算法：
- **GRPO**: 推荐，适合数学推理，不需要偏好对
- DPO: 需要 (chosen, rejected) 偏好对
- PPO/RLOO: 其他选择

## 任务流程
1. 分析数据格式和任务特点
2. 设计合适的 reward function
3. 编写训练脚本 (使用 GRPOTrainer)
4. 训练并保存模型到 {config.output_path}
5. 使用 BenchmarkEvalTool 评测
6. 根据结果迭代优化

## 注意事项
- 训练完成后必须保存模型到 output 目录
- 每次只改动一个变量，便于归因
- 关注 GPU 内存使用，避免 OOM
"""


def build_task_prompt(config: Config, history: list = None) -> str:
    """构建任务 prompt"""
    history_str = ""
    if history:
        history_str = "\n## 历史实验\n"
        for exp in history[-5:]:  # 只保留最近 5 次
            history_str += f"- 配置: {exp.get('config', 'N/A')}\n"
            history_str += f"  分数: {exp.get('score', 'N/A')}\n"
    else:
        history_str = "\n## 历史实验\n无（首次运行）\n"
    
    return f"""## 当前任务
请为 {config.benchmark} benchmark 设计并执行 RL 后训练。

{history_str}

请开始分析数据并设计训练策略。
"""


def run_pipeline(config: Config):
    """运行 RL 训练 pipeline"""
    print(f"=== OpenHands RL Pipeline ===")
    print(f"Benchmark: {config.benchmark}")
    print(f"Model: {config.base_model}")
    print(f"Workspace: {config.workspace}")
    
    # 创建 LLM
    llm = LLM(
        model=config.llm_model,
        api_key=SecretStr(config.llm_api_key),
        base_url=config.llm_base_url,
    )
    
    # 创建 Agent
    agent = create_agent(config, llm)
    
    # 创建 Conversation
    conv = Conversation(
        agent=agent,
        workspace=config.workspace,
        system_prompt=build_system_prompt(config),
    )
    
    # 发送任务
    task_prompt = build_task_prompt(config)
    print(f"\n--- Sending task to agent ---")
    conv.send_message(task_prompt)
    
    # 运行
    print(f"\n--- Running agent ---")
    conv.run(max_iterations=config.max_iterations)
    
    print(f"\n=== Pipeline completed ===")


def main():
    parser = argparse.ArgumentParser(description="OpenHands RL Post-training Pipeline")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--benchmark", type=str, default="gsm8k", help="Benchmark name")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--max-iterations", type=int, default=50)
    args = parser.parse_args()
    
    # 加载配置
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config = Config(**config_dict)
    else:
        config = Config(
            benchmark=args.benchmark,
            base_model=args.base_model,
            workspace=args.workspace,
            max_iterations=args.max_iterations,
        )
    
    run_pipeline(config)


if __name__ == "__main__":
    main()
