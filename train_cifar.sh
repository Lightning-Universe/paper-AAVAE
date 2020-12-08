# train CIFAR10 on one V100 GPU, should take ~28 hours
grid train --grid_instance_type p3.2xlarge --grid_gpus 1 src/train.py --gpus 1
