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
 * `vae/resnet.py` - ResNet based encoder and decoder architecture

