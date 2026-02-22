#!/usr/bin/env python3
"""
OpenHands RL Post-training Pipeline (Fixed-Stage)

每轮迭代：代码生成 → 训练执行 → (FSM 部署/rollout) → 评测提交 → 反馈注入。
"""

import argparse
import os
import sys
import time
from pathlib import Path

from benchmarks.registry import get_benchmark, list_benchmarks
from pipeline.runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="OpenHands RL Pipeline (Fixed-Stage)")
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        help="Benchmark 名称（对应 benchmarks/ 下的子目录）")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--run-dir", type=str, default="",
                        help="指定运行目录（默认自动生成 runs/{benchmark}_{timestamp}）")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--training-timeout", type=int, default=3600)
    parser.add_argument("--max-agent-steps", type=int, default=25)

    parser.add_argument("--fsm-enabled", action="store_true",
                        help="启用 FSM-Runner 自动部署/rollout/评测")
    parser.add_argument("--fsm-target-repo", type=str, default=".")
    parser.add_argument("--fsm-deploy-engine", type=str, default="vllm",
                        choices=["vllm", "tgi", "local"])
    parser.add_argument("--fsm-repair-iters", type=int, default=3)
    parser.add_argument("--fsm-mode", type=str, default="smoke",
                        choices=["smoke", "full"])

    parser.add_argument("--list-benchmarks", action="store_true",
                        help="列出所有可用 benchmark 并退出")

    args = parser.parse_args()

    if args.list_benchmarks:
        names = list_benchmarks()
        if not names:
            print("No benchmarks found in benchmarks/ directory.")
        else:
            print(f"Available benchmarks ({len(names)}):")
            for n in names:
                b = get_benchmark(n)
                print(f"  {n:<20s} [{b.task_type}]  {b.description}")
        sys.exit(0)

    bench = get_benchmark(args.benchmark)
    data_dir = str(bench.data_dir.resolve())

    if args.run_dir:
        run_dir = str(Path(args.run_dir).resolve())
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = str((Path("runs") / f"{args.benchmark}_{ts}").resolve())

    output_dir = str(Path(run_dir) / "output")
    code_dir = str(Path(run_dir) / "code")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(code_dir).mkdir(parents=True, exist_ok=True)

    os.environ["DATA_PATH"] = data_dir
    os.environ["OUTPUT_DIR"] = output_dir

    fsm_config = {
        "enabled": args.fsm_enabled or os.environ.get("FSM_ENABLED", "").lower() in ("1", "true", "yes"),
        "target_repo": args.fsm_target_repo or os.environ.get("FSM_TARGET_REPO", "."),
        "deploy_engine": args.fsm_deploy_engine or os.environ.get("FSM_DEPLOY_ENGINE", "vllm"),
        "repair_iters": args.fsm_repair_iters,
        "mode": args.fsm_mode,
        "opencode_url": os.environ.get("OPENCODE_URL", ""),
        "opencode_model": os.environ.get("OPENCODE_MODEL", ""),
    }

    run_pipeline(
        task=args.benchmark,
        base_model=args.base_model,
        workspace=run_dir,
        data_path=data_dir,
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        training_timeout=args.training_timeout,
        max_agent_steps=args.max_agent_steps,
        fsm_config=fsm_config,
    )


if __name__ == "__main__":
    main()
