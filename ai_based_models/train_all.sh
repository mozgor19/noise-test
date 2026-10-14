#!/usr/bin/env bash
# v3/v4 modellerini (VER=v3|v4|v5, varsayılan v5) sırayla eğitir (yarıda kalan varsa kaldığı epoch'tan devam eder).
# Canlı izleme:   tail -f ai_based_models/logs/train_<config>.log
# Epoch özeti:    grep "^Epoch" ai_based_models/logs/train_<config>.log
set -e
cd "$(dirname "$0")/src"
PY=${PY:-/home/mustafa/Projeler/dncnn/dncnn_env/bin/python}
mkdir -p ../logs
VER=${VER:-v5}
for c in dncnn_sp_$VER seconv_sp_$VER dncnn_$VER; do
  echo "### $c başladı $(date '+%F %T')"
  "$PY" -u train.py --config "../configs/$c.yaml" --resume >> "../logs/train_$c.log" 2>&1
  grep -E "^Epoch [0-9]+:" "../logs/train_$c.log" | tail -1
done
echo "### hepsi bitti $(date '+%F %T')"
