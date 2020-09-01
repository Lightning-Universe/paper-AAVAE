# VAE

```bash
python vae/vae.py --prior normal --posterior normal
python vae/finetune.py path/to/checkpoint
```

 * `vae/vae.py` - VAE python module
 * `vae/finetune.py` - Fintune python module
 * `vae/resnet.py` - ResNet based encoder and decoder architecture

## ResNet VAE Architecture

#### Encoder

Encoder is standard ResNet18 architecture, but `conv1` is places with a 3x3
kernel with stride 1, padding 1. The output of the final average pool is
512x1x1.

#### Decoder

Decoder is essentially the encoder in reverse order, where the average pool is
replaced with a linear layer, and stride 2 convolutions are replaced with a 2x
upscale followed by a stride 1 convolution.

## Pre-train Results

#### Input and Reconstruction

![input](figures/input.png)
![reconst](figures/reconst.png)

#### Random Sample

![sample](figures/sample.png)


#### TSNE

![tsne](figures/tsne.png)
