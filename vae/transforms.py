import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class Identity:
    def __call__(self, x):
        return x


class GlobalTransform:
    def __init__(self, flip: bool = False, jitter_strength: float = 1.0):
        color_jitter = T.ColorJitter(
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.2 * jitter_strength,
        )

        transforms = []
        if flip:
            transforms.append(T.RandomHorizontalFlip())

        transforms.append(T.RandomApply([color_jitter], p=0.8))
        transforms.append(T.RandomGrayscale(p=0.2))
        self.transforms = T.Compose(transforms)

    def __call__(self, x):
        return self.transforms(x)


class LocalTransform:
    def __init__(self, size=32):
        self.transforms = T.Compose([GlobalTransform(), T.RandomResizedCrop(size=size)])

    def __call__(self, x):
        return self.transforms(x)


class Transforms:
    transform_map = {
        "original": Identity,
        "global": GlobalTransform,
        "local": LocalTransform,
    }

    def __init__(
        self, input_transform="original", recon_transform="original", normalize_fn=None
    ):
        input_transform = [self.transform_map[input_transform](), T.ToTensor()]
        recon_transform = [self.transform_map[recon_transform](), T.ToTensor()]

        if normalize_fn is not None:
            input_transform.append(normalize_fn)
            recon_transform.append(normalize_fn)

        self.input_transform = T.Compose(input_transform)
        self.recon_transform = T.Compose(recon_transform)

    def __call__(self, x):
        return self.input_transform(x), self.recon_transform(x)
