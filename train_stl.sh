# train STL10 on four V100 GPU
grid train --grid_instance_type p3.8xlarge src/train.py --grid_gpus 4 --dataset stl10 --batch_size 56 --gpus 4
