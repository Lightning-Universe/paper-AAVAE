#!/bin/sh

# TODO: add ckpt path
for i in {0..7}; do
    screen -dmS "run$i" bash -c "source /home/ananya/env/bin/activate; python setup.py install; CUDA_VISIBLE_DEVICES=$i python src/linear_eval.py --ckpt_path lightning_logs/version_'$i'/checkpoints/last.ckpt; exec sh"
    sleep 5
done
