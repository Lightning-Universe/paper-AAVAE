#!/bin/sh

NAME="cifar10-lr-1e-4-kl-1e-3-logscale-sweep"

screen -dmS "logging" bash -c "source /home/ananya/env/bin/activate; tensorboard dev upload --logdir lightning_logs/ --name $NAME; exec sh"