#!/usr/bin/env bash
# v3/v4 modelleriyle (VER=v3|v4|v5, varsayılan v5) Tiny ve 480p gürültülü setlerini temizler.
# Çıktılar: Tiny-Imagenet/Tiny_Cleaned_{DnCNN,DnCNN_SP,SeConvUNet}_{10..90}/
#           480p-Test/Cleaned_{DnCNN,DnCNN_SP,SeConvUNet}_{10..90}/
set -e
cd "$(dirname "$0")/src"
PY=${PY:-/home/mustafa/Projeler/dncnn/dncnn_env/bin/python}
VER=${VER:-v5}
for c in dncnn_$VER dncnn_sp_$VER seconv_sp_$VER; do
  echo "### $c (tiny) $(date '+%F %T')"
  "$PY" infer.py --config "../configs/$c.yaml" --dataset tiny \
      --noisy_root ../../Tiny-Imagenet --out_root ../../Tiny-Imagenet
  echo "### $c (480p) $(date '+%F %T')"
  "$PY" infer.py --config "../configs/$c.yaml" --dataset 480p \
      --noisy_root ../../480p-Test --out_root ../../480p-Test
done
echo "### hepsi bitti $(date '+%F %T')"
