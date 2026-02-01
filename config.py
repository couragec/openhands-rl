"""配置管理"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """OpenHands RL Pipeline 配置
    
    优先从环境变量读取（用于 autorl_bench 集成），否则使用默认值。
    
    环境变量映射：
    - TASK -> benchmark
    - BASE_MODEL -> base_model
    - WORKSPACE -> workspace
    - MODEL_PATH -> model_path
    - DATA_PATH -> data_path
    - OUTPUT_DIR -> output_path
    - GRADING_SERVER_URL -> grading_server_url
    """
    
    # 任务配置（从环境变量或参数）
    benchmark: str = field(default_factory=lambda: os.environ.get("TASK", "gsm8k"))
    base_model: str = field(default_factory=lambda: os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct"))
    
    # 路径配置（从环境变量或参数）
    workspace: str = field(default_factory=lambda: os.environ.get("WORKSPACE", "./workspace"))
    data_path: Optional[str] = field(default_factory=lambda: os.environ.get("DATA_PATH"))
    output_path: Optional[str] = field(default_factory=lambda: os.environ.get("OUTPUT_DIR"))
    model_path: Optional[str] = field(default_factory=lambda: os.environ.get("MODEL_PATH"))
    
    # LLM 配置
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o"))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    llm_base_url: Optional[str] = field(default_factory=lambda: os.environ.get("LLM_BASE_URL"))
    
    # Pipeline 配置
    max_iterations: int = field(default_factory=lambda: int(os.environ.get("MAX_ITERATIONS", "50")))
    timeout: int = 7200  # 2 hours
    
    # Grading Server（用于 autorl_bench 集成）
    grading_server_url: Optional[str] = field(default_factory=lambda: os.environ.get("GRADING_SERVER_URL"))
    
    def __post_init__(self):
        """初始化派生路径（如果环境变量未提供）"""
        ws = Path(self.workspace)
        
        if self.data_path is None:
            self.data_path = str(ws / "data")
        if self.output_path is None:
            self.output_path = str(ws / "output")
        if self.model_path is None:
            self.model_path = str(ws / "models" / self.base_model)
