#!/bin/sh

# can set min, max and n
LOG_SCALE=$(awk -v n=8 -v min=-5 -v max=2 -v seed="$RANDOM" 'BEGIN { srand(seed); for (i=0; i<n; ++i) print rand() * (max - min) + min }')
LOG_SCALE=($(echo ${LOG_SCALE=[*]}| tr " " "\n" | sort -n))
LR=1e-4
KL=1e-3

for i in {0..7}; do
    screen -dmS "run$i" bash -c "source /home/ananya/env/bin/activate; python setup.py install; CUDA_VISIBLE_DEVICES=$i python src/vae.py --seed $(date +%s) --denoising --gpus 1 --batch_size 256 --warmup_epochs 10 --val_samples 16 --weight_decay 0 --learning_rate $LR --kl_coeff $KL --log_scale ${LOG_SCALE[$i]}; exec sh"
    sleep 5
done
