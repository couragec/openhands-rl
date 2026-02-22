# OpenHands RL

使用 OpenHands SDK 进行 RL 后训练的自动化 Pipeline，集成 FSM-Runner 合同驱动执行框架实现模型部署、rollout 采样和评测的全流程自动化。

## 功能

- 固定阶段式 Pipeline：代码生成 → 训练执行 → 模型部署 → Rollout 采样 → 评测
- 支持 GRPO/DPO/PPO 训练（基于 trl）
- FSM-Runner 合同驱动：通过 `pipeline.yml` 定义 deploy/rollout/evaluation 各阶段
- 阶段缓存：部署成功后自动缓存，下次跳过重复部署（SHA256 校验 + TTL + Health Check）
- Benchmark 注册表：自动发现 `benchmarks/` 下的所有 benchmark，新增只需一个 `config.yaml`
- 运行隔离：每次运行产物存放在 `runs/{benchmark}_{timestamp}/`

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

### 查看可用 Benchmark

```bash
python main.py --list-benchmarks
```

### 基础 Pipeline（不启用 FSM）

```bash
export LLM_API_KEY="your-api-key"
export LLM_MODEL="openai/glm-4.7"
export LLM_BASE_URL="https://open.bigmodel.cn/api/coding/paas/v4"

python main.py --benchmark gsm8k --base-model Qwen/Qwen2.5-Coder-0.5B-Instruct
```

### 启用 FSM Deploy + Rollout + Evaluate

```bash
python main.py \
    --benchmark gsm8k \
    --base-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --fsm-enabled \
    --fsm-deploy-engine local \
    --fsm-mode smoke
```

### 环境变量

| 变量 | 用途 |
|------|------|
| `LLM_API_KEY` | LLM API 密钥 |
| `LLM_MODEL` | LLM 模型名（如 `openai/glm-4.7`） |
| `LLM_BASE_URL` | LLM API 地址 |
| `TRAINED_MODEL_PATH` | 训练好的模型目录（FSM deploy 使用） |
| `FSM_ENABLED` | 是否启用 FSM（`1`/`true`） |
| `AIDER_FSM_CACHE_TTL` | 缓存 TTL 秒数（默认 3600） |

## 项目结构

```
openhand/
├── main.py                     # 主入口
├── config.py                   # 配置管理
├── fsm_bridge.py               # FSM-Runner 桥接层
├── pipeline.yml                # FSM 合同定义
├── requirements.txt
│
├── benchmarks/                 # Benchmark 注册表 + 数据
│   ├── registry.py             #   自动发现/路径解析
│   ├── gsm8k/
│   │   ├── config.yaml         #   元信息（名称/类型/描述）
│   │   └── data/
│   │       └── train.jsonl
│   ├── humaneval/
│   │   ├── config.yaml
│   │   └── data/
│   │       └── train.jsonl
│   ├── mbpp/
│   │   ├── config.yaml
│   │   └── data/
│   │       └── train.jsonl
│   └── _template/              #   新 benchmark 模板
│       └── config.yaml
│
├── pipeline/                   # Pipeline 核心逻辑
│   ├── runner.py               #   Pipeline 编排器
│   ├── phases.py               #   各阶段实现
│   ├── prompts.py              #   Agent prompt 模板
│   ├── types.py                #   数据类型定义
│   └── utils.py                #   工具函数
│
├── runner_fsm/                 # FSM-Runner 合同驱动执行框架
│   ├── env.py                  #   EnvSession API
│   ├── core/                   #   核心执行引擎
│   │   ├── stage_cache.py      #     阶段缓存
│   │   └── ...
│   ├── contract/               #   合同验证 + 修复
│   ├── opencode/               #   OpenCode 客户端
│   └── utils/                  #   子进程/安全/评测审计
│
├── .aider_fsm/                 # FSM 运行时目录
│   ├── stages/                 #   各阶段 shell 脚本
│   └── cache/                  #   阶段缓存文件
│
└── runs/                       # 运行产物（自动生成，不提交）
    └── {benchmark}_{timestamp}/
        ├── code/               #   生成的训练代码
        │   └── train.py
        ├── output/             #   模型输出
        └── pipeline_results.json
```

## 新增 Benchmark

1. 创建目录 `benchmarks/{name}/`
2. 添加 `config.yaml`：
   ```yaml
   name: my_benchmark
   task_type: math
   description: "My custom benchmark"
   ```
3. 放入数据 `benchmarks/{name}/data/train.jsonl`
4. 运行 `python main.py --benchmark my_benchmark`

## FSM 执行流程

```
pipeline.yml 定义
       │
       ▼
  ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌────────────┐
  │  tests   │ ──▶ │  deploy  │ ──▶ │ rollout  │ ──▶ │ evaluation │
  │(torch ok)│     │  setup   │     │(生成样本) │     │ (计算指标)  │
  └─────────┘     │  health  │     └──────────┘     └────────────┘
                  └──────────┘                             │
                       │                                   ▼
                  缓存写入                           metrics.json
                  (.aider_fsm/cache/)              {ok, score, accuracy}
```
