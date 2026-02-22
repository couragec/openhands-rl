#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ENV_JSON="${AIDER_RUNTIME_ENV_PATH:-.aider_fsm/runtime_env.json}"
METRICS_JSON=".aider_fsm/metrics.json"
ROLLOUT_JSON=".aider_fsm/rollout.json"
DATA_PATH="${DATA_PATH:-}"
EVAL_MODE="${AIDER_EVAL_MODE:-smoke}"
EVAL_LIMIT="${AIDER_EVAL_LIMIT:-64}"

if [ "$EVAL_MODE" = "smoke" ]; then
    EVAL_LIMIT="${AIDER_EVAL_LIMIT:-8}"
fi

echo "Evaluation: mode=$EVAL_MODE limit=$EVAL_LIMIT"

python -c "
import json, os, re, sys

metrics_path = '$METRICS_JSON'
rollout_path = '$ROLLOUT_JSON'
data_path = os.environ.get('DATA_PATH', '$DATA_PATH')
eval_limit = int('$EVAL_LIMIT')

# Load ground truth answers from training data
answers = {}
train_jsonl = os.path.join(data_path, 'train.jsonl') if os.path.isdir(data_path) else data_path
if os.path.exists(train_jsonl):
    with open(train_jsonl, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = item.get('question') or item.get('prompt') or item.get('instruction') or ''
            a = item.get('answer') or item.get('response') or item.get('output') or ''
            if q and a:
                answers[q.strip()] = a.strip()

# Load rollout samples
samples = []
samples_path = None
if os.path.exists(rollout_path):
    with open(rollout_path) as f:
        rollout = json.load(f)
    sp = rollout.get('paths', {}).get('samples_jsonl', '')
    if sp and os.path.exists(sp):
        samples_path = sp

if samples_path and os.path.exists(samples_path):
    with open(samples_path) as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass

if not samples:
    print('WARNING: no rollout samples found, writing minimal metrics')
    metrics = {'ok': False, 'score': 0.0, 'reason': 'no_samples'}
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    sys.exit(1)

# Evaluate: compare completions against ground truth
def extract_number(text):
    nums = re.findall(r'-?[\d,]+\.?\d*', text.replace(',', ''))
    return nums[-1] if nums else None

correct = 0
total = 0
rewards_sum = 0.0

for s in samples[:eval_limit]:
    prompt = s.get('prompt', '').strip()
    completion = s.get('completion', '').strip()
    reward = s.get('reward', 0.0)
    rewards_sum += float(reward)
    total += 1

    gt = answers.get(prompt, '')
    if not gt:
        continue

    gt_num = extract_number(gt)
    pred_num = extract_number(completion)
    if gt_num is not None and pred_num is not None and gt_num == pred_num:
        correct += 1

accuracy = correct / max(total, 1)
avg_reward = rewards_sum / max(total, 1)
score = max(accuracy, avg_reward)

metrics = {
    'ok': True,
    'score': round(score, 4),
    'accuracy': round(accuracy, 4),
    'avg_reward': round(avg_reward, 4),
    'total_evaluated': total,
    'correct': correct,
    'eval_mode': '$EVAL_MODE',
}

with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f'Evaluation complete:')
print(f'  score={score:.4f} accuracy={accuracy:.4f} avg_reward={avg_reward:.4f}')
print(f'  total={total} correct={correct}')
print(f'  metrics: {metrics_path}')
"

echo "Evaluation stage done"
