# OpenHands RL

使用 OpenHands SDK 进行 RL 后训练的自动化 Pipeline。

## 功能

- 自动分析任务和数据
- 设计 reward function
- 使用 trl 库进行 GRPO/DPO/PPO 训练
- 自动评测和迭代优化

## 安装

```bash
# 使用现有的 openhands 环境
conda activate openhands

# 或安装依赖
pip install openhands-sdk trl transformers datasets
```

## 使用

### 独立运行

```bash
python main.py \
    --benchmark gsm8k \
    --base-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --workspace ./workspace
```

### 与 AutoRL-Bench 集成

在 AutoRL-Bench 中注册为 agent：

```bash
# 在 autorl_bench/agents/ 下创建 openhands/ 目录
# 配置 config.yaml 指向本项目
```

## 配置

配置文件示例 (`configs/gsm8k.yaml`):

```yaml
benchmark: gsm8k
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
workspace: ./workspace

llm_model: gpt-4o
max_iterations: 50
timeout: 7200
```

## 项目结构

```
openhands-rl/
├── main.py           # 主入口
├── config.py         # 配置管理
├── tools/
│   ├── trl_training.py      # TRL 训练工具
│   └── benchmark_eval.py    # 评测工具
├── configs/          # 配置文件
└── README.md
```

## 与 openhands-magic (SFT) 的区别

| 方面 | openhands-magic | openhands-rl |
|------|-----------------|--------------|
| 任务 | Supervised Fine-tuning | RL Post-training |
| 训练框架 | LlamaFactory | trl (GRPO/DPO/PPO) |
| 数据格式 | instruction-output | prompt + reward |
