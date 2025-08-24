#!/usr/bin/env bash


# 실행 순서
# conda activate nlp310
# 1. pip install -r requirements.txt
# (맥터미널)
# chmod +x scripts/run_all.sh
# ./scripts/run_all.sh

# 그냥 한개만 : python -m src/main.py --config configs/bert_ebs256.yaml



set -e

CFG_DIR=configs

runs=(
  bert_ebs256.yaml
  bert_ebs512.yaml
  bert_ebs1024.yaml
  modernbert_ebs256.yaml
  modernbert_ebs512.yaml
  modernbert_ebs1024.yaml
  bert_ebs256_adamw.yaml
  bert_ebs512_adamw.yaml
  bert_ebs1024_adamw.yaml
  modernbert_ebs256_adamw.yaml
  modernbert_ebs512_adamw.yaml
  modernbert_ebs1024_adamw.yaml
)

for cfg in "${runs[@]}"; do
  echo ">>> Running $cfg"
  python -m src.main --config "$CFG_DIR/$cfg"
done
