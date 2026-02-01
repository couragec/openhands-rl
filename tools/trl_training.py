"""TRL 训练工具"""

import os
import subprocess
from typing import Optional

from openhands.sdk.tool import register_tool


@register_tool
class TRLTrainingTool:
    """使用 trl 库进行 RL 训练的工具"""
    
    name = "trl_training"
    description = """使用 trl 库进行 RL 后训练。
    
支持的算法：
- GRPO (推荐): 适合数学推理任务，不需要偏好对
- DPO: 需要 (chosen, rejected) 偏好对
- PPO/RLOO: 其他选择

参数：
- script_path: 训练脚本路径
- model_path: 基础模型路径
- output_dir: 输出目录
- algorithm: 训练算法 (grpo/dpo/ppo)
- extra_args: 额外参数 (如 --batch_size 8)
"""
    
    @staticmethod
    def run(
        script_path: str,
        model_path: str,
        output_dir: str,
        algorithm: str = "grpo",
        extra_args: Optional[str] = None,
    ) -> dict:
        """执行训练脚本
        
        Args:
            script_path: 训练脚本路径
            model_path: 基础模型路径
            output_dir: 输出目录
            algorithm: 训练算法
            extra_args: 额外命令行参数
            
        Returns:
            dict: 包含 exit_code, stdout, stderr
        """
        # 构建命令
        cmd = [
            "python", script_path,
            "--model_path", model_path,
            "--output_dir", output_dir,
            "--algorithm", algorithm,
        ]
        
        if extra_args:
            cmd.extend(extra_args.split())
        
        # 执行
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hours
                cwd=os.path.dirname(script_path) or ".",
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout,
                "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": "Training timeout (2 hours)",
                "success": False,
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
            }


# 提供一个简单的训练脚本模板
GRPO_TRAINING_TEMPLATE = '''#!/usr/bin/env python3
"""GRPO Training Script Template"""

import argparse
import json
import os

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def load_data(data_path: str):
    """加载训练数据"""
    records = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            # 根据数据格式调整
            prompt = f"Question: {item['question']}\\n\\nAnswer:"
            records.append({
                "prompt": prompt,
                "answer": item.get("answer", ""),
            })
    return records


def reward_func(completions, answer, **kwargs):
    """Reward function - 根据任务自定义"""
    rewards = []
    for completion, gold in zip(completions, answer):
        # 简单的精确匹配，实际应用需要更复杂的逻辑
        if gold.strip() in completion:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_path", default="./data/train.jsonl")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    # 加载数据
    train_data = load_data(args.data_path)
    dataset = Dataset.from_list(train_data)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        max_completion_length=256,
        num_generations=4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    
    # 训练
    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
'''
