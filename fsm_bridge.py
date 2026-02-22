"""FSM-Runner 桥接模块。

封装 OpenCode-FSM-Runner 的 setup/rollout/evaluate API，
为 openhands-rl pipeline 提供自动部署、rollout 采样和评测能力。
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class FSMBridge:
    """openhands-rl 与 FSM-Runner 之间的桥接层。

    职责：
    1. 初始化 FSM 合同环境（setup）
    2. 部署训练好的模型为推理服务 + 生成 rollout 样本
    3. 运行评测并收集指标
    4. 将 samples.jsonl 路径回传给训练循环
    """

    def __init__(
        self,
        target_repo: str = ".",
        deploy_engine: str = "vllm",
        repair_iters: int = 3,
        opencode_url: str = "",
        opencode_model: str = "",
        audit: str = "on",
        cache_enabled: bool = True,
        data_path: str = "",
        output_dir: str = "",
    ):
        self.target_repo = target_repo
        self.deploy_engine = deploy_engine
        self.repair_iters = repair_iters
        self.opencode_url = opencode_url
        self.opencode_model = opencode_model
        self.audit = audit
        self.cache_enabled = cache_enabled
        self.data_path = data_path
        self.output_dir = output_dir
        self._session = None

    def _ensure_session(self):
        """延迟初始化 FSM session。"""
        if self._session is not None:
            return

        try:
            from runner_fsm.env import EnvSession, setup
        except ImportError:
            from runner.env import EnvSession, setup

        self._session = setup(
            target=self.target_repo,
            require_metrics=True,
            audit=self.audit,
            use_cache=self.cache_enabled,
            opencode_model=self.opencode_model,
            opencode_url=self.opencode_url,
            unattended="strict",
            strict_opencode=True,
        )
        self._session.command_hints = []
        self._session.hint_anchors = []

    def deploy_and_rollout(
        self,
        model_path: str,
        mode: str = "smoke",
        require_samples: bool = True,
    ) -> dict[str, Any]:
        """部署模型并执行 rollout 采样。

        Args:
            model_path: 训练好的模型目录路径
            mode: "smoke" 或 "full"
            require_samples: 是否要求生成 samples.jsonl

        Returns:
            dict 含 ok, samples_path, reason 等字段
        """
        self._ensure_session()

        model_dir = Path(model_path).resolve()
        if not model_dir.exists():
            return {"ok": False, "reason": f"model_path_not_found: {model_dir}"}

        env_overrides = self._build_env_overrides(model_path)

        try:
            rollout_result = self._session.rollout(
                llm=model_dir,
                mode=mode,
                require_samples=require_samples,
                env_overrides=env_overrides,
                repair_iters=self.repair_iters,
            )
        except Exception as e:
            return {"ok": False, "reason": f"rollout_exception: {e}"}

        if not rollout_result.ok:
            verify = rollout_result.verify
            failed = getattr(verify, "failed_stage", "unknown") if verify else "unknown"
            return {"ok": False, "reason": f"rollout_failed: stage={failed}"}

        samples_path = self._extract_samples_path(rollout_result.rollout_path)

        return {
            "ok": True,
            "samples_path": samples_path or "",
            "rollout_path": str(rollout_result.rollout_path or ""),
            "artifacts_dir": str(rollout_result.artifacts_dir),
        }

    def evaluate(
        self,
        model_path: str | None = None,
        mode: str = "smoke",
    ) -> dict[str, Any]:
        """执行 FSM 评测。

        Args:
            model_path: 可选，覆盖 session 的模型路径
            mode: "smoke" 或 "full"

        Returns:
            dict 含 ok, score, metrics, reason 等字段
        """
        self._ensure_session()

        env_overrides = {}
        if model_path:
            env_overrides = self._build_env_overrides(model_path)

        llm_arg = Path(model_path) if model_path else None

        try:
            eval_result = self._session.evaluate(
                llm=llm_arg,
                mode=mode,
                env_overrides=env_overrides if env_overrides else None,
                repair_iters=self.repair_iters,
            )
        except Exception as e:
            return {"ok": False, "reason": f"evaluate_exception: {e}"}

        if not eval_result.ok:
            verify = eval_result.verify
            failed = getattr(verify, "failed_stage", "unknown") if verify else "unknown"
            return {"ok": False, "reason": f"evaluate_failed: stage={failed}"}

        metrics = eval_result.metrics or {}
        score = metrics.get("score")
        improvement = metrics.get("improvement")

        return {
            "ok": True,
            "score": score,
            "improvement": improvement,
            "best_score": metrics.get("best_score"),
            "metrics": metrics,
            "metrics_path": str(eval_result.metrics_path or ""),
        }

    def teardown(self):
        """释放 FSM session 资源。"""
        self._session = None

    def _build_env_overrides(self, model_path: str) -> dict[str, str]:
        """构建传递给 FSM stage 脚本的环境变量。"""
        overrides: dict[str, str] = {}

        overrides["TRAINED_MODEL_PATH"] = str(model_path)
        overrides["DEPLOY_ENGINE"] = self.deploy_engine

        if self.data_path:
            overrides["DATA_PATH"] = self.data_path
        if self.output_dir:
            overrides["OUTPUT_DIR"] = self.output_dir

        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
                     "MODEL_PATH", "CUDA_VISIBLE_DEVICES"):
            val = os.environ.get(key, "")
            if val:
                overrides[key] = val

        if "DATA_PATH" not in overrides:
            val = os.environ.get("DATA_PATH", "")
            if val:
                overrides["DATA_PATH"] = val
        if "OUTPUT_DIR" not in overrides:
            val = os.environ.get("OUTPUT_DIR", "")
            if val:
                overrides["OUTPUT_DIR"] = val

        return overrides

    def _extract_samples_path(self, rollout_path: Path | None) -> str | None:
        """从 rollout.json 中提取 samples.jsonl 路径。"""
        if rollout_path is None:
            return None

        try:
            with open(rollout_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            return None

        if not isinstance(obj, dict):
            return None

        paths = obj.get("paths")
        if isinstance(paths, dict):
            raw = paths.get("samples_jsonl")
            if isinstance(raw, str) and raw.strip():
                p = Path(raw.strip())
                if p.exists():
                    return str(p)
                repo = Path(self.target_repo).resolve()
                candidate = (repo / p).resolve()
                if candidate.exists():
                    return str(candidate)

        return None
