#!/bin/sh

# can set min, max and n
WEIGHT_DECAY=(0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 1)
LR=1e-4
BATCH_SIZE=256
WARMUP_EPOCHS=10

for i in {0..7}; do
    screen -dmS "run$i" bash -c "source /home/ananya/env/bin/activate; python setup.py install; CUDA_VISIBLE_DEVICES=$i python src/ae.py --seed $(date +%s) --online_ft --denoising --gpus 1 --batch_size $BATCH_SIZE --warmup_epochs $WARMUP_EPOCHS --learning_rate $LR --weight_decay ${WEIGHT_DECAY[$i]}; exec sh"
    sleep 5
done
