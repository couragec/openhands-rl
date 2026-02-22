#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ENV_JSON="${AIDER_RUNTIME_ENV_PATH:-.aider_fsm/runtime_env.json}"
ROLLOUT_JSON=".aider_fsm/rollout.json"
SAMPLES_JSONL=".aider_fsm/samples.jsonl"
DATA_PATH="${DATA_PATH:-}"
EVAL_LIMIT="${AIDER_EVAL_LIMIT:-64}"
EVAL_MODE="${AIDER_EVAL_MODE:-smoke}"

if [ "$EVAL_MODE" = "smoke" ]; then
    EVAL_LIMIT="${AIDER_EVAL_LIMIT:-8}"
fi

if [ ! -f "$RUNTIME_ENV_JSON" ]; then
    echo "ERROR: runtime_env.json not found at $RUNTIME_ENV_JSON" >&2
    exit 1
fi

MODEL_PATH=$(python -c "
import json
with open('$RUNTIME_ENV_JSON') as f:
    obj = json.load(f)
print(obj.get('model_path', ''))
")

echo "Rollout: model=$MODEL_PATH limit=$EVAL_LIMIT mode=$EVAL_MODE"

python -c "
import json, os, re, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '$MODEL_PATH'
data_path = os.environ.get('DATA_PATH', '$DATA_PATH')
limit = int('$EVAL_LIMIT')
samples_path = '$SAMPLES_JSONL'
rollout_json = '$ROLLOUT_JSON'

print(f'Loading model from {model_path}...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()
print('Model loaded.')

prompts = []
train_jsonl = os.path.join(data_path, 'train.jsonl') if os.path.isdir(data_path) else data_path
if os.path.exists(train_jsonl):
    with open(train_jsonl, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = item.get('question') or item.get('prompt') or item.get('instruction') or ''
            if q:
                prompts.append(q)
            if len(prompts) >= limit:
                break

if not prompts:
    print('ERROR: no prompts found', file=sys.stderr)
    sys.exit(1)

print(f'Generating completions for {len(prompts)} prompts...')

samples = []
errors = 0
for i, prompt in enumerate(prompts):
    try:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        reward = 1.0 if completion.strip() else 0.0
        samples.append({'prompt': prompt, 'completion': completion, 'reward': reward})
        if (i + 1) % 2 == 0:
            print(f'  {i+1}/{len(prompts)} done')
    except Exception as e:
        errors += 1
        samples.append({'prompt': prompt, 'completion': '', 'reward': 0.0})
        if errors <= 3:
            print(f'  Sample {i+1} error: {e}', file=sys.stderr)

with open(samples_path, 'w') as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

rollout = {
    'paths': {'samples_jsonl': samples_path},
    'counts': {'samples': len(samples), 'errors': errors},
    'model': os.path.basename(model_path),
}
with open(rollout_json, 'w') as f:
    json.dump(rollout, f, indent=2, ensure_ascii=False)

print(f'Rollout complete: {len(samples)} samples, {errors} errors')
"

echo "Rollout stage done"
