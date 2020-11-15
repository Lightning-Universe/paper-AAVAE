import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from pl_bolts.models.self_supervised.simclr.transforms import GaussianBlur


class Identity:
    def __init__(self, size=32, **kwargs):
        self.size = size

    def __call__(self, x):
        return x


class GlobalTransform:
    def __init__(self, size=32, flip: bool = False, jitter_strength: float = 1.0):
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
    def __init__(self, size=32, **kwargs):
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
        self,
        size,
        flip=False,
        jitter_strength=1.,
        input_transform="original",
        recon_transform="original",
        normalize_fn=None
    ):

        input_transform = [self.transform_map[input_transform](
            size, flip=flip, jitter_strength=jitter_strength
        ), T.ToTensor()]
        recon_transform = [self.transform_map[recon_transform](
            size, flip=flip, jitter_strength=jitter_strength
        ), T.ToTensor()]

        original_transform = [Identity(), T.ToTensor()]

        if normalize_fn is not None:
            input_transform.append(normalize_fn)
            recon_transform.append(normalize_fn)
            original_transform.append(normalize_fn)

        self.input_transform = T.Compose(input_transform)
        self.recon_transform = T.Compose(recon_transform)
        self.original_transform = T.Compose(original_transform)

    def __call__(self, x):
        return (
            self.input_transform(x),
            self.recon_transform(x),
            self.original_transform(x),
        )


class MultiViewTrainTransform:
    def __init__(self, normalization, num_views: int, input_height: int, s: int = 1):

        self.s = s
        self.num_views = num_views
        self.input_height = input_height
        color_jitter = T.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = T.Compose([
            T.RandomResizedCrop(size=self.input_height),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.5),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * self.input_height)),
            T.ToTensor(),
            normalization
        ])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        results = [transform(sample) for _ in range(self.num_views)]
        return results


class MultiViewEvalTransform:
    def __init__(self, normalization, num_views: int, input_height: int, s: int = 1):

        self.s = s
        self.num_views = num_views
        self.input_height = input_height
        self.test_transform = T.Compose([
            T.Resize(input_height + 10, interpolation=3),
            T.CenterCrop(input_height),
            T.ToTensor(),
            normalization
        ])

    def __call__(self, sample):
        transform = self.test_transform
        results = [transform(sample) for _ in range(self.num_views)]
        return results
