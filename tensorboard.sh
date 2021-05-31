#!/bin/sh

NAME="ae-cifar10-lr-wd-baselines"

screen -dmS "logging" bash -c "source /home/ananya/env/bin/activate; tensorboard dev upload --logdir lightning_logs/ --name $NAME; exec sh"