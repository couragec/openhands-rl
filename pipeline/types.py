"""Pipeline 数据类型定义。"""

from dataclasses import dataclass


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
    samples_path: str = ""
