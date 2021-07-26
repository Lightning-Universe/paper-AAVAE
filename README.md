# AAVAE

Official implementation of the paper "AAVAE: Augmentation-AugmentedVariational Autoencoders"

To train the AAVAE model

1. Create a python virtual environment.
2. ``python setup.py install``.
3. Train using ``python src/vae.py --denoising``.

To reproduce the results from the paper on CIFAR-10:

```
python src/vae.py \
    --gpus 1 \
    --max_epochs 3200 \
    --batch_size 256 \
    --warmup_epochs 10 \
    --val_samples 16 \
    --weight_decay 0 \
    --logscale 0 \
    --kl_coeff 0 \
    --learning_rate 2.5e-4
```

To evaluate the pretrained encoder

```
python src/linear_eval.py --ckpt_path "path\to\saved\file.ckpt"
```

### Saved checkpoints

| Model | Dataset | Checkpoint | Downstream acc. |
| --- | --- | --- | --- |
| AAVAE | CIFAR-10 | [checkpoint](https://aavae.s3.us-east-2.amazonaws.com/checkpoints/aavae_cifar10.ckpt) | 87.14 |
| AAVAE | STL-10 | [checkpoint](https://aavae.s3.us-east-2.amazonaws.com/checkpoints/aavae_stl10.ckpt) | 84.72 |
