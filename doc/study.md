https://github.com/couragec/openhands-rl

## Git Push

```bash
cd /Data/home/v-wanyichen/cwy/program/cwy/openhands-rl
git add .
git status
git commit -m "update"
git push origin main
```

## openhands-rl 改进方案

参考 openhands-magic（SFT）的实现，结合首次运行（20260212T081001）暴露的问题。

### 1. GPU 信息注入

当前 prompt 没提 GPU，agent 不知道硬件能力，写出低效代码。

改法（`build_code_prompt()`）：
- 注入 GPU 数量、型号（如 4x B200 80GB）、`CUDA_VISIBLE_DEVICES`
- `phase_training()` 里用 `accelerate launch train.py` 替代 `python train.py`，pipeline 层面自动多卡，agent 不用操心
- 给参数选择建议表（GPU 数量 → batch_size / gradient_accumulation）

SFT 做法：LlamaFactory 自动处理多卡，prompt 里告知 GPU 数量和 ID，agent 据此选 LoRA vs Full SFT。

### 2. GRPO 参考代码模板

首次运行 agent 浪费 30 步研究 TRL API，根因是 prompt 里没给代码骨架。

改法（`build_code_prompt()`）：
- 加约 50 行最小 GRPO 代码模板（数据加载 + reward 函数 + GRPOTrainer + 保存）
- 标注 TRL 版本号（0.27+）和可用 API
- 强调"先基于模板写最简版跑通，再迭代优化"
- 提示可以先用少量样本快速验证，确认无误后再全量训练

SFT 做法：给了 `call_llm` 完整实现、并发模板、数据预览代码等参考代码。训练本身封装在 LlamaFactoryTool 里，agent 只传参数。

### 3. 错误反馈增强 + 重试

当前训练失败只给 error log 最后 40 行，且没有重试机制。

改法（`run_pipeline()` Phase 2）：
- 训练失败后开新 Conversation（与 SFT 一致，不复用上下文），构造 `fix_prompt`：
  - 错误日志（最后 4000 字符）
  - 当前 train.py 完整内容
  - 数据预览（前 3 条样本）
- 最多重试 2 次，同一轮内完成，不消耗迭代次数
- 提交最新 checkpoint 评测

SFT 做法：脚本失败 → 新 Conversation + fix_prompt（错误日志 + 脚本 + 数据预览）→ 最多 3 次重试。还用 LLM 实时分析运行日志检测错误。

### 4. 上下文处理

当前每轮迭代开新 Conversation，靠 prompt 注入历史，这点已经和 SFT 一致。

需要补充的：
- 数据统计注入：样本数、prompt/answer 平均长度 → 帮选 batch_size / max_length
- 历史信息丰富化：除了迭代表 + 上轮详情，还可以加上轮训练耗时、显存使用情况

SFT 做法：每次都新建 `LLM → Agent → Conversation`，所有信息塞进 prompt（历史表格、最近 2 轮详情、数据统计、参数表、错误样本）。

### 已完成

- [x] `run.py` 添加 signal handler（SIGTERM/SIGINT），被 kill 时写结束日志
- [x] `build_code_prompt()`: GPU 信息注入（数量/型号/CUDA_VISIBLE_DEVICES）+ 参数建议
- [x] `build_code_prompt()`: GRPO 参考代码模板（~50行可运行代码）+ 少量样本验证提示
- [x] `build_code_prompt()`: 数据统计注入（样本数、prompt/answer 平均长度）
- [x] `phase_training()`: `accelerate launch` 替代 `python`，自动多卡 DDP
- [x] `run_pipeline()`: 训练失败重试（新 Conversation + fix_prompt，最多 2 次）
- [x] `build_fix_prompt()`: 新增修复 prompt 构造（错误日志 + 代码 + 数据预览）
- [x] 新增工具函数：`get_data_stats()`, `load_data_preview()`, `get_gpu_info()`
