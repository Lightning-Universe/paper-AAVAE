#!/bin/sh

WEIGHT_DECAY=(0 1e-6 1e-5 1e-4)
LR=1e-4
BATCH_SIZE=256
WARMUP_EPOCHS=10

for i in {0..3}; do
    screen -dmS "run$i" bash -c "source /home/ananya/env/bin/activate; python setup.py install; CUDA_VISIBLE_DEVICES=$i python src/ae.py --seed $(date +%s) --online_ft --gpus 1 --max_epochs 4000 --batch_size $BATCH_SIZE --warmup_epochs $WARMUP_EPOCHS --learning_rate $LR --weight_decay ${WEIGHT_DECAY[$i]}; exec sh"
    sleep 5
done

WEIGHT_DECAY=(-1 -1 -1 -1 0 1e-6 1e-5 1e-4)
LR=2.5e-4
BATCH_SIZE=256
WARMUP_EPOCHS=10

for i in {4..7}; do
    screen -dmS "run$i" bash -c "source /home/ananya/env/bin/activate; python setup.py install; CUDA_VISIBLE_DEVICES=$i python src/ae.py --seed $(date +%s) --online_ft --gpus 1 --max_epochs 4000 --batch_size $BATCH_SIZE --warmup_epochs $WARMUP_EPOCHS --learning_rate $LR --weight_decay ${WEIGHT_DECAY[$i]}; exec sh"
    sleep 5
done
