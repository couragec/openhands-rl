"""Pipeline prompt 构建逻辑。"""

import os
from pathlib import Path

from .types import IterationResult
from .utils import load_data_preview


def build_code_prompt(
    iteration: int,
    workspace: str,
    base_model: str,
    task_description: str,
    history: list[IterationResult],
    data_stats: dict | None = None,
    gpu_info: dict | None = None,
    rollout_stats: dict | None = None,
) -> str:
    """构建代码生成阶段的 prompt

    关键设计：
    - 第 1 轮：给 GRPO 参考代码 + 数据统计，让 agent 快速写出能跑的代码
    - 后续轮：注入上轮结果（score/error），引导定向优化
    - 若有 rollout 样本统计，注入到 prompt 帮助 agent 优化 reward function
    """
    model_path = os.environ.get("MODEL_PATH", "")
    data_path = os.environ.get("DATA_PATH", "")
    output_dir = os.environ.get("OUTPUT_DIR", "")

    gpu_section = ""
    if gpu_info:
        gpu_section = f"""
## 硬件环境
- GPU: {gpu_info['num_gpus']}x {gpu_info['gpu_name']}
- CUDA_VISIBLE_DEVICES={gpu_info['cuda_devices']}
- Pipeline 会用 `accelerate launch` 执行你的代码，自动启用多卡 DDP，你不需要手动处理分布式
- 参数建议：{gpu_info['num_gpus']} 卡时 per_device_train_batch_size=4, gradient_accumulation_steps=2
"""

    data_stats_section = ""
    if data_stats and data_stats.get("count", 0) > 0:
        data_stats_section = f"""
## 数据统计
- 样本数: {data_stats['count']}
- 平均 prompt 长度: {data_stats['avg_prompt_len']} 字符
- 平均 answer 长度: {data_stats['avg_answer_len']} 字符
"""

    rollout_section = ""
    if rollout_stats:
        rollout_section = f"""
## 上轮 Rollout 采样统计
- 总样本数: {rollout_stats['total_samples']}
- 平均 reward: {rollout_stats['avg_reward']}
- reward > 0 比例: {rollout_stats['reward_positive_ratio']}
- 平均 prompt 长度: {rollout_stats['avg_prompt_len']} 字符
- 平均 completion 长度: {rollout_stats['avg_completion_len']} 字符

**提示**：可根据 rollout 采样结果优化 reward 函数设计。若 reward > 0 比例过低，考虑降低 reward 门槛或改进 prompt 格式。
"""

    history_section = ""
    if history:
        rows = []
        for h in history:
            score_s = f"{h.score:.2f}" if h.score is not None else "-"
            imp_s = f"{h.improvement:+.2f}" if h.improvement is not None else "-"
            status = "OK" if h.exit_code == 0 else f"FAIL({h.exit_code})"
            rows.append(f"| {h.iteration} | {status} | {h.training_time:.0f}s | {score_s} | {imp_s} |")

        history_section += "\n## 历史实验结果\n"
        history_section += "| 轮次 | 状态 | 耗时 | Score | vs Baseline |\n"
        history_section += "|------|------|------|-------|-------------|\n"
        history_section += "\n".join(rows) + "\n"

        last = history[-1]
        if last.exit_code != 0 and last.stdout:
            tail = "\n".join(last.stdout.strip().splitlines()[-60:])
            history_section += f"\n### 上轮错误日志（最后 60 行）\n```\n{tail}\n```\n"
            history_section += "\n**请根据错误信息修复代码。**\n"
        elif last.score is not None:
            history_section += f"\n### 上轮评测\n- Score: {last.score}\n- Improvement: {last.improvement}\n"
            history_section += "\n**请调整策略（reward 函数、超参数等）提升 score。**\n"
        elif last.exit_code == 0:
            history_section += "\n### 上轮训练成功但无评测结果\n"
            history_section += "可能原因：模型未保存到 $OUTPUT_DIR，或输出目录为空。\n"
            history_section += "**请确保代码中有 `trainer.save_model(OUTPUT_DIR)` 或等效保存操作。**\n"

    prev_code_section = ""
    if history:
        code_file = Path(history[-1].code_path)
        if code_file.exists():
            code_text = code_file.read_text()
            if len(code_text) > 4000:
                code_text = code_text[:4000] + "\n# ... (truncated)"
            prev_code_section = f"\n## 上轮代码\n```python\n{code_text}\n```\n"

    grpo_template = ""
    if iteration == 1:
        grpo_template = _build_grpo_template()

    if iteration == 1:
        task_instruction = f"""## 你的任务（第 1 轮：链路打通）
1. 用 terminal 快速查看数据格式：`head -3 {data_path}/train.jsonl`
2. 阅读 {workspace}/description.md 了解任务要求
3. 在 {workspace}/code/ 下编写 train.py（可参考上面的代码示例）
4. 重点工作：
   - 根据数据格式调整 prompt 构造
   - 根据任务设计 reward 函数（参考 description.md）
   - 路径通过 os.environ 获取（MODEL_PATH, DATA_PATH, OUTPUT_DIR）
   - 训练完成后保存模型到 $OUTPUT_DIR
5. 建议：可以先用少量样本验证链路能跑通，后续轮再全量训练
6. 完成后调用 finish 工具结束

**重要**：你只负责写代码，不要自己执行训练脚本。pipeline 会用 accelerate 自动运行。"""
    else:
        task_instruction = f"""## 你的任务（第 {iteration} 轮：迭代优化）
1. 分析上轮结果（见历史实验和错误日志）
2. 修改 {workspace}/code/train.py
3. 改进方向：
   - 如果上轮失败：修复错误
   - 如果 score 为空：确保模型保存到 $OUTPUT_DIR
   - 如果有 score：优化 reward 函数、调整超参数、尝试不同策略
   - 如果上轮用了数据子集：可以尝试增大数据量或训练步数
4. 完成后调用 finish 工具结束

**重要**：你只负责写代码，不要自己执行训练脚本。pipeline 会用 accelerate 自动运行。"""

    return f"""你是 RL 后训练专家。

## 安全限制
- 只能在 {workspace} 内操作
- 禁止 pip install 或任何包管理命令
- 预装库：transformers, trl, torch, vllm, datasets, accelerate, peft
{gpu_section}
## 目录结构
- 代码区: {workspace}/code/
- 训练数据: {data_path}（只读）
- 基础模型: {model_path}（只读，{base_model}）
- 模型输出: {output_dir}

## 环境变量（代码中用 os.environ 读取）
- MODEL_PATH={model_path}
- DATA_PATH={data_path}
- OUTPUT_DIR={output_dir}
- CUDA_VISIBLE_DEVICES={gpu_info['cuda_devices'] if gpu_info else ''}
{data_stats_section}{rollout_section}
## 任务描述
{task_description}
{grpo_template}{history_section}{prev_code_section}{task_instruction}"""


def build_fix_prompt(
    code_path: str,
    error_log: str,
    data_path: str,
    workspace: str,
) -> str:
    """构造训练失败后的修复 prompt。"""
    code_text = ""
    if Path(code_path).exists():
        code_text = Path(code_path).read_text()
        if len(code_text) > 6000:
            code_text = code_text[:6000] + "\n# ... (truncated)"

    data_preview = load_data_preview(data_path, num_samples=3)
    error_tail = "\n".join(error_log.strip().splitlines()[-100:]) if error_log else "（无日志）"

    return f"""训练脚本执行失败，请分析错误并修复。

## 错误日志（最后 100 行）
```
{error_tail}
```

## 当前 train.py
```python
{code_text}
```

## 数据预览（前 3 条）
```json
{data_preview}
```

## 要求
1. 仔细分析错误原因
2. 修复 {workspace}/code/train.py
3. 使用 file_editor 工具保存修复后的代码
4. 不要执行脚本，pipeline 会自动运行
5. 完成后调用 finish 工具结束
"""


def _build_grpo_template() -> str:
    return """
## GRPO 参考代码模板（TRL 0.28+）

以下是一个可参考的 GRPO 训练代码示例，可以根据需要自行调整：

```python
import os, json, re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]

# 1. 加载数据 → Dataset 格式，必须有 "prompt" 字段
with open(os.path.join(DATA_PATH, "train.jsonl")) as f:
    raw = [json.loads(l) for l in f if l.strip()]

# 根据实际数据格式调整 prompt 构造（先 head -3 看数据格式）
prompts = []
for item in raw:
    question = item.get("question") or item.get("prompt") or item.get("instruction")
    prompts.append([{"role": "user", "content": question}])

dataset = Dataset.from_dict({"prompt": prompts})

# 2. Reward 函数
# 注意：completions 可能是 chat 格式（list of dicts）或纯字符串，必须先提取文本
def reward_fn(completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        if isinstance(comp, list):
            text = comp[-1]["content"] if comp else ""
        else:
            text = str(comp)
        # 简单示例：有数字答案 +1，否则 0；请根据任务替换
        rewards.append(1.0 if re.search(r"\\d+", text) else 0.0)
    return rewards

# 3. 训练配置
config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=8,
    learning_rate=5e-6,
    max_completion_length=256,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    report_to="none",
)

# 4. 训练
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
```

**注意事项**：
- TRL 版本 0.28+，GRPOTrainer 签名：`GRPOTrainer(model, reward_funcs, args, train_dataset, processing_class=tokenizer)`
- 注意是 `processing_class` 不是 `tokenizer`（旧版用 tokenizer 会报错）
- `reward_funcs` 是第二个位置参数
- completions 是 chat 格式（list of dicts），reward 函数里要用 `comp[-1]["content"]` 提取文本
- prompt 必须是 chat 格式（list of dict），不是纯字符串
- `num_generations` 必须能被 `per_device_train_batch_size * num_gpus` 整除
- 可以先用少量样本快速验证链路能跑通，确认无误后再全量训练
"""
