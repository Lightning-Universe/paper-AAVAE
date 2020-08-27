## Experiments

 - [ ] "Fine-tune" untrained encoder:
```bash
python vae.py --finetune
```

 - [ ] Train normal VAE, fine tune encoder for classification:
```bash
# train VAE
python vae.py
# finetune
python vae.py --pretrained <checkpoint_path> --finetune
```

 - [ ] Train normal VAE with extra cosine term in loss (see
     `vae.training_step`):
```bash
# train VAE
python vae.py --cosine
# finetune
python vae.py --pretrained <checkpoint_path> --finetune
```

## Results

WIP
