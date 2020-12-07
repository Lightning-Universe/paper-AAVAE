# train CIFAR10 on one V100 GPU, should take ~28 hours
grid train --grid_instance p3.2xlarge src/train.py