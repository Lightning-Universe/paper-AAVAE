#!/bin/sh

WEIGHT_DECAY=(0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 1)
LR=1e-4
KL=1e-3

for i in {0..8}; do
    screen -dmS "run$i" bash -c "source /home/ananya/env/bin/activate; CUDA_VISIBLE_DEVICES=$i python src/vae.py --seed $(date +%s) --denoising --gpus 1 --online_ft --batch_size 256 --warmup_epochs 10 --kl_coeff $KL --learning_rate $LR --weight_decay ${WEIGHT_DECAY[$i]}; exec sh"
    sleep 5
done