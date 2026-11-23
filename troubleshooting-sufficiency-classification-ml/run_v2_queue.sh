#!/bin/zsh
# Waits for the running Qwen3 prompt_variants.py job, then runs prompts_v2.py on Qwen2.5 and Qwen3 (one model at a time).
cd "$(dirname "$0")"
while pgrep -f "prompt_variants.py|llm_judge.py" >/dev/null; do sleep 20; done
../.venv/bin/python prompts_v2.py > logs/prompts_v2_q25.log 2>&1
../.venv/bin/python prompts_v2.py mlx-community/Qwen3-8B-4bit q3 > logs/prompts_v2_q3.log 2>&1
echo queue done
