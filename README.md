# AAVAE

Official implementation of the paper ["AAVAE: Augmentation-AugmentedVariational Autoencoders"](https://arxiv.org/pdf/2107.12329.pdf)

![AAVAE](https://aavae.s3.us-east-2.amazonaws.com/images/model.png)

### Abstract

Recent methods for self-supervised learning can be grouped into two paradigms: contrastive and non-contrastive approaches. Their success can largely be attributed to data augmentation pipelines which generate multiple views of a single input that preserve the underlying semantics. In this work, we introduce augmentation-augmented variational autoencoders (AAVAE), a third approach to self-supervised learning based on autoencoding. We derive AAVAE starting from the conventional variational autoencoder (VAE), by replacing the KL divergence regularization, which is agnostic to the input domain, with data augmentations that explicitly encourage the internal representations to encode domain-specific invariances and equivariances. We empirically evaluate the proposed AAVAE on image classification, similar to how recent contrastive and non-contrastive learning algorithms have been evaluated. Our experiments confirm the effectiveness of data augmentation as a replacement for KL divergence regularization. The AAVAE outperforms the VAE by 30\% on CIFAR-10 and 40\% on STL-10. The results for AAVAE are largely comparable to the state-of-the-art for self-supervised learning.

### Training

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
