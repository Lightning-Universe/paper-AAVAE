# VAE

## TODO

- [ ] marginal log p in `metrics.py` (then uncomment in `vae.py`)
- [ ] use validation accuracy in `online_eval.py` (not as urgent)

## INTERNAL ONLY: Run with grid

```
# needs to be run from root 
grid train --gpus 1 --instance_type p3.2xlarge vae/vae.py
```

## Usage

```bash
# train vae
python vae/vae.py --prior normal --posterior laplace

# train vae, construct original from SimCLR view
python vae/vae.py --simclr True

```

 * `vae/vae.py` - VAE python module
 * `vae/finetune.py` - Fintune python module
 * `vae/resnet.py` - ResNet based encoder and decoder architecture

## Results

Model     | Prior (P) | Posterior (Q) | Reconstruct View | Val. Gini | Finetune Val. Acc. 
---       | ---       | ---           | ---              | ---       | ---
ResNet18  | Normal    | Normal        | None             | 0.437     | 46.03
ResNet18  | Laplace   | Normal        | None             | 0.445     | 45.13
ResNet18  | Normal    | Laplace       | None             | 0.476     | 44.76
ResNet18  | Normal    | Normal        | SimCLR           |           |
ResNet18  | Laplace   | Normal        | SimCLR           |           |
ResNet18  | Normal    | Laplace       | SimCLR           |           |

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
