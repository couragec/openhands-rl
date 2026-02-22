"""Pipeline 主循环：编排 code_gen → training → FSM deploy/rollout/evaluate → 迭代。"""

import json
import os
import time
from pathlib import Path

from .phases import (
    create_llm,
    phase_code_generation,
    phase_evaluation,
    phase_fix_training,
    phase_training,
)
from .types import IterationResult
from .utils import get_data_stats, get_gpu_info, get_rollout_samples_stats


def _try_fsm_deploy_and_rollout(
    workspace: str,
    model_path: str,
    fsm_config: dict,
    data_path: str = "",
    output_dir: str = "",
) -> dict | None:
    """尝试通过 FSM-Runner 部署模型并执行 rollout 采样。

    返回 dict 含 samples_path / score / metrics，或 None 表示跳过/失败。
    """
    if not fsm_config.get("enabled"):
        return None

    try:
        from fsm_bridge import FSMBridge
    except ImportError:
        print("  FSM bridge not available, skipping FSM rollout")
        return None

    print(f"\n{'='*60}")
    print(f"  Phase FSM: Deploy + Rollout")
    print(f"{'='*60}")

    bridge = FSMBridge(
        target_repo=fsm_config.get("target_repo", "."),
        deploy_engine=fsm_config.get("deploy_engine", "vllm"),
        repair_iters=fsm_config.get("repair_iters", 3),
        opencode_url=fsm_config.get("opencode_url", ""),
        opencode_model=fsm_config.get("opencode_model", ""),
        data_path=data_path,
        output_dir=output_dir,
    )

    try:
        result = bridge.deploy_and_rollout(
            model_path=model_path,
            mode=fsm_config.get("mode", "smoke"),
            require_samples=True,
        )
        if result and result.get("ok"):
            print(f"  FSM rollout OK: samples={result.get('samples_path')}")
            return result
        else:
            reason = result.get("reason", "unknown") if result else "bridge returned None"
            print(f"  FSM rollout failed: {reason}")
            return result
    except Exception as e:
        print(f"  FSM rollout error: {e}")
        return None


def _try_fsm_evaluate(
    workspace: str,
    model_path: str,
    fsm_config: dict,
    data_path: str = "",
    output_dir: str = "",
) -> dict | None:
    """尝试通过 FSM-Runner 执行评测。

    返回 dict 含 score / metrics，或 None 表示跳过/失败。
    """
    if not fsm_config.get("enabled"):
        return None

    try:
        from fsm_bridge import FSMBridge
    except ImportError:
        return None

    print(f"\n{'='*60}")
    print(f"  Phase FSM: Evaluate")
    print(f"{'='*60}")

    bridge = FSMBridge(
        target_repo=fsm_config.get("target_repo", "."),
        deploy_engine=fsm_config.get("deploy_engine", "vllm"),
        repair_iters=fsm_config.get("repair_iters", 3),
        opencode_url=fsm_config.get("opencode_url", ""),
        opencode_model=fsm_config.get("opencode_model", ""),
        data_path=data_path,
        output_dir=output_dir,
    )

    try:
        result = bridge.evaluate(mode=fsm_config.get("mode", "smoke"))
        if result and result.get("ok"):
            print(f"  FSM evaluate OK: score={result.get('score')}")
            return result
        else:
            reason = result.get("reason", "unknown") if result else "bridge returned None"
            print(f"  FSM evaluate failed: {reason}")
            return result
    except Exception as e:
        print(f"  FSM evaluate error: {e}")
        return None


def run_pipeline(
    task: str,
    base_model: str,
    workspace: str,
    data_path: str = "",
    output_dir: str = "",
    max_iterations: int = 5,
    training_timeout: int = 3600,
    max_agent_steps: int = 25,
    fsm_config: dict | None = None,
):
    """运行固定阶段 Pipeline

    每轮迭代：
      Phase 1: Agent 写代码（探索数据 + 编写 train.py）
      Phase 2: Pipeline 执行训练（subprocess）
      Phase FSM-Deploy: （可选）FSM 部署模型 + rollout 采样
      Phase 3: Pipeline 提交评测（HTTP POST 或 FSM evaluate）
      Phase 4: 记录结果，注入到下轮 prompt
    """
    pipeline_start = time.time()
    grading_url = os.environ.get("GRADING_SERVER_URL", "http://localhost:5000")
    fsm_config = fsm_config or {}

    if not data_path:
        data_path = os.environ.get("DATA_PATH", "")
    if not output_dir:
        output_dir = os.environ.get("OUTPUT_DIR", str(Path(workspace) / "output"))

    print(f"{'#'*60}")
    print(f"  OpenHands RL Pipeline (Fixed-Stage)")
    print(f"{'#'*60}")
    print(f"  Task: {task}")
    print(f"  Model: {base_model}")
    print(f"  Workspace: {workspace}")
    print(f"  Data path: {data_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Training timeout: {training_timeout}s")
    print(f"  Agent steps/iter: {max_agent_steps}")
    print(f"  Grading Server: {grading_url}")
    print(f"  FSM enabled: {fsm_config.get('enabled', False)}")
    print(f"  LLM: {os.environ.get('LLM_MODEL', 'gpt-4.1')}")
    print()

    task_description = ""
    for fname in ["description.md", "instructions.md"]:
        fpath = Path(workspace) / fname
        if fpath.exists():
            task_description += fpath.read_text() + "\n\n"

    if not task_description.strip():
        print("WARNING: No description.md or instructions.md found in workspace")

    data_stats = get_data_stats(data_path)
    gpu_info = get_gpu_info()
    print(f"  Data: {data_stats['count']} samples")
    print(f"  GPU: {gpu_info['num_gpus']}x {gpu_info['gpu_name']}")

    llm = create_llm()
    print(f"  LLM initialized: {llm.model}\n")

    history: list[IterationResult] = []
    best_score: float | None = None
    best_iteration = -1
    max_fix_retries = 2
    last_rollout_stats: dict | None = None

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
                rollout_stats=last_rollout_stats,
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
                phase_fix_training(llm, code_path, stdout, data_path, workspace)
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

        # Phase FSM: Deploy + Rollout（训练成功后）
        if exit_code == 0 and fsm_config.get("enabled"):
            out_path = Path(output_dir)
            subdirs = [d for d in out_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if subdirs:
                trained_model = str(max(subdirs, key=lambda d: d.stat().st_mtime))
            else:
                trained_model = str(out_path)

            fsm_result = _try_fsm_deploy_and_rollout(
                workspace, trained_model, fsm_config,
                data_path=data_path, output_dir=output_dir,
            )
            if fsm_result and fsm_result.get("ok"):
                samples_path = fsm_result.get("samples_path", "")
                result.samples_path = samples_path
                if samples_path:
                    last_rollout_stats = get_rollout_samples_stats(samples_path)

        # Phase 3: Evaluation
        if exit_code == 0:
            eval_result = None

            if fsm_config.get("enabled"):
                out_path = Path(output_dir)
                subdirs = [d for d in out_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
                trained_model = str(max(subdirs, key=lambda d: d.stat().st_mtime)) if subdirs else str(out_path)
                fsm_eval = _try_fsm_evaluate(
                    workspace, trained_model, fsm_config,
                    data_path=data_path, output_dir=output_dir,
                )
                if fsm_eval and fsm_eval.get("ok"):
                    eval_result = {
                        "score": fsm_eval.get("score"),
                        "improvement": fsm_eval.get("improvement"),
                        "best": {"score": fsm_eval.get("best_score")},
                        "model_path": trained_model,
                    }

            if eval_result is None:
                eval_result = phase_evaluation(workspace, grading_url)

            if eval_result:
                result.score = eval_result.get("score")
                result.improvement = eval_result.get("improvement")
                result.best_score = eval_result.get("best", {}).get("score")
                result.model_path = eval_result.get("model_path", "")

                if best_score is None or (result.score is not None and result.score > best_score):
                    best_score = result.score
                    best_iteration = iteration

        # Phase 4: Record
        history.append(result)

        iter_elapsed = time.time() - iter_start
        print(f"\n  --- Iteration {iteration} Summary ---")
        print(f"  Exit code: {result.exit_code}")
        print(f"  Score: {result.score}")
        print(f"  Best so far: {best_score} (iter {best_iteration})")
        print(f"  Iteration time: {iter_elapsed:.0f}s")

    total_time = time.time() - pipeline_start
    print(f"\n{'#'*60}")
    print(f"  Pipeline Complete")
    print(f"  Best score: {best_score}")
    print(f"  Best iteration: {best_iteration}")
    print(f"  Total iterations: {len(history)}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"{'#'*60}")

    summary = {
        "task": task,
        "base_model": base_model,
        "best_score": best_score,
        "best_iteration": best_iteration,
        "total_time": total_time,
        "fsm_enabled": fsm_config.get("enabled", False),
        "iterations": [
            {
                "iteration": r.iteration,
                "exit_code": r.exit_code,
                "training_time": r.training_time,
                "score": r.score,
                "improvement": r.improvement,
                "samples_path": r.samples_path,
            }
            for r in history
        ],
    }
    summary_path = Path(workspace) / "pipeline_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  Results saved: {summary_path}")
