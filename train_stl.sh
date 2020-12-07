# train STL10 on four V100 GPU
grid train --grid_instance p3.8xlarge src/train.py --dataset stl10 --batch_size 56