# VAE

## TODO

- [ ] marginal log p in `metrics.py` (then uncomment in `vae.py`)
- [ ] use validation accuracy in `online_eval.py` (not as urgent)

## INTERNAL ONLY: Run with grid

```
# needs to be run from root 
grid train --name baseline --gpus 1 --instance_type p3.2xlarge vae/vae.py --recon_transform "['original','global','local']" --input_transform "['original','global','local']"
grid train --name laplace_post --gpus 1 --instance_type p3.2xlarge vae/vae.py --posterior laplace --recon_transform "['original','global','local']" --input_transform "['original','global','local']"
grid train --name laplace_prior --gpus 1 --instance_type p3.2xlarge vae/vae.py --prior laplace --recon_transform "['original','global','local']" --input_transform "['original','global','local']"
```

## Usage

```bash
# train vae
python vae/vae.py --prior normal --posterior laplace

# train vae, construct original from local transform
python vae/vae.py --recon_transform original --input_transform local

```

 * `vae/vae.py` - VAE python module
 * `vae/resnet.py` - ResNet based encoder and decoder architecture

