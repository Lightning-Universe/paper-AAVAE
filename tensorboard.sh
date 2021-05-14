#!/bin/sh

NAME="cifar10-kl-1e-3-lr-1e-4-wd-sweep"

screen -dmS "logging" bash -c "source /home/ananya/env/bin/activate; tensorboard dev upload --logdir lightning_logs/ --name $NAME; exec sh"