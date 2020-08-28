## Experiments

https://tensorboard.dev/experiment/Q120HK9PSymvkJEdKgJ6yA/#scalars

 - [X] "Fine-tune" untrained encoder:
```bash
# tensorboard version_0
python vae.py --finetune --max_epochs 200
```

 - [X] Train normal VAE, fine tune encoder for classification:
```bash
# train VAE (tensorboard version_1)
python vae.py
# finetune (tensorboard version_2)
python vae.py --finetune --max_epochs 200 --pretrained <checkpoint_path>
```

 - [X] Train normal VAE with extra cosine term in loss (see
     `vae.training_step`):
```bash
# train VAE (tensorboard version_3)
python vae.py --cosine
# finetune (tensorboard version_4)
python vae.py --finetune --max_epochs 200 --pretrained <checkpoint_path>
```

## Results

`encoder = [conv, batch_norm, relu] * 4 + fc`

Model                                                               | Max Valid Accuracy CIFAR10
---                                                                 | ---
encoder, no pretraining                                             | 0.83
encoder, VAE pretraining                                            | 0.83
encoder, VAE pretraining + cosine similarity loss from SimCLR pairs | 0.83

