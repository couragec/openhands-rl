"""Pipeline 工具函数：数据统计、GPU 信息、数据预览。"""

import json
import os
import subprocess
from pathlib import Path


def get_data_stats(data_path: str) -> dict:
    """读取 train.jsonl，返回样本数和长度统计。"""
    samples = []
    path = Path(data_path)
    jsonl = path / "train.jsonl" if path.is_dir() else path

    if not jsonl.exists():
        return {"count": 0, "avg_prompt_len": 0, "avg_answer_len": 0}

    try:
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    except Exception:
        return {"count": 0, "avg_prompt_len": 0, "avg_answer_len": 0}

    if not samples:
        return {"count": 0, "avg_prompt_len": 0, "avg_answer_len": 0}

    prompt_lens = []
    answer_lens = []
    for s in samples:
        prompt = s.get("prompt") or s.get("question") or s.get("instruction") or ""
        answer = s.get("answer") or s.get("response") or s.get("output") or ""
        prompt_lens.append(len(prompt))
        answer_lens.append(len(answer))

    return {
        "count": len(samples),
        "avg_prompt_len": sum(prompt_lens) // max(len(prompt_lens), 1),
        "avg_answer_len": sum(answer_lens) // max(len(answer_lens), 1),
    }


def load_data_preview(data_path: str, num_samples: int = 3) -> str:
    """返回 train.jsonl 前几条样本的 JSON 预览。"""
    path = Path(data_path)
    jsonl = path / "train.jsonl" if path.is_dir() else path

    if not jsonl.exists():
        return "（数据文件不存在）"

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


def get_rollout_samples_stats(samples_path: str) -> dict | None:
    """读取 rollout 产生的 samples.jsonl，返回统计信息。"""
    p = Path(samples_path)
    if not p.exists():
        return None

    total = 0
    rewards = []
    prompt_lens = []
    completion_lens = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if not isinstance(item, dict):
                    continue
                total += 1
                r = item.get("reward")
                if isinstance(r, (int, float)):
                    rewards.append(float(r))
                pr = item.get("prompt", "")
                co = item.get("completion", "")
                if isinstance(pr, str):
                    prompt_lens.append(len(pr))
                if isinstance(co, str):
                    completion_lens.append(len(co))
    except Exception:
        return None

    if total == 0:
        return None

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    avg_prompt = sum(prompt_lens) // max(len(prompt_lens), 1)
    avg_completion = sum(completion_lens) // max(len(completion_lens), 1)

    return {
        "total_samples": total,
        "avg_reward": round(avg_reward, 4),
        "avg_prompt_len": avg_prompt,
        "avg_completion_len": avg_completion,
        "reward_positive_ratio": round(
            sum(1 for r in rewards if r > 0) / max(len(rewards), 1), 4
        ),
    }
