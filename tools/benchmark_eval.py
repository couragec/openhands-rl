"""Benchmark 评测工具"""

import json
import os
import subprocess
from typing import Optional

import requests
from openhands.sdk.tool import register_tool


@register_tool
class BenchmarkEvalTool:
    """使用 OpenCompass 或 Grading Server 评测模型"""
    
    name = "benchmark_eval"
    description = """评测模型在 benchmark 上的性能。

支持两种模式：
1. Grading Server: 通过 HTTP 提交评测（用于 autorl_bench）
2. OpenCompass: 直接运行 OpenCompass 评测

参数：
- model_path: 要评测的模型路径
- benchmark: benchmark 名称 (gsm8k, math, etc.)
- grading_server_url: Grading Server URL（可选）
"""
    
    @staticmethod
    def run(
        model_path: str,
        benchmark: str = "gsm8k",
        grading_server_url: Optional[str] = None,
    ) -> dict:
        """评测模型
        
        Args:
            model_path: 模型路径
            benchmark: benchmark 名称
            grading_server_url: Grading Server URL
            
        Returns:
            dict: 包含 score, details, success
        """
        # 方式1: 使用 Grading Server
        if grading_server_url:
            return BenchmarkEvalTool._eval_via_grading_server(
                model_path, grading_server_url
            )
        
        # 方式2: 直接用 OpenCompass
        return BenchmarkEvalTool._eval_via_opencompass(model_path, benchmark)
    
    @staticmethod
    def _eval_via_grading_server(model_path: str, server_url: str) -> dict:
        """通过 Grading Server 评测"""
        try:
            resp = requests.post(
                f"{server_url}/submit",
                json={"model_path": model_path},
                timeout=600,
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "score": data.get("score"),
                    "details": data,
                    "success": True,
                }
            else:
                return {
                    "score": None,
                    "details": {"error": f"HTTP {resp.status_code}"},
                    "success": False,
                }
        except Exception as e:
            return {
                "score": None,
                "details": {"error": str(e)},
                "success": False,
            }
    
    @staticmethod
    def _eval_via_opencompass(model_path: str, benchmark: str) -> dict:
        """直接使用 OpenCompass 评测"""
        try:
            # 生成配置
            config_content = f'''
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='custom_model',
        path='{model_path}',
        tokenizer_path='{model_path}',
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]

datasets = gsm8k_datasets
'''
            config_path = "/tmp/opencompass_eval_config.py"
            with open(config_path, "w") as f:
                f.write(config_content)
            
            # 运行 OpenCompass
            result = subprocess.run(
                ["opencompass", config_path, "--work-dir", "/tmp/opencompass_results"],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            
            # 解析结果
            # TODO: 解析 OpenCompass 输出获取分数
            
            return {
                "score": None,  # 需要从输出解析
                "details": {
                    "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-1000:],
                },
                "success": result.returncode == 0,
            }
        except Exception as e:
            return {
                "score": None,
                "details": {"error": str(e)},
                "success": False,
            }
