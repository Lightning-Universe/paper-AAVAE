import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class SimCLRTransform(object):
    """
        x = sample()
        (xi, xj, x) = transform(x)
    """

    def __init__(
        self,
        input_height: int = 224,
        gaussian_blur: bool = True,
        flip: bool = False,
        jitter_strength: float = 1.0,
        normalize=None,
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.flip = flip
        self.normalize = normalize

        self.color_jitter = T.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        data_transforms = []

        if self.flip:
            data_transforms.append(T.RandomHorizontalFlip())

        data_transforms.append(T.RandomApply([self.color_jitter], p=0.8))
        data_transforms.append(T.RandomGrayscale(p=0.2))

        """
        if self.gaussian_blur:
            data_transforms.append(GaussianBlur(kernel_size=int(0.1 * self.input_height, p=0.5)))
        """

        data_transforms.append(T.ToTensor())
        eval_transform = [T.Resize(self.input_height), T.ToTensor()]

        if self.normalize:
            data_transforms.append(normalize)
            eval_transform.append(normalize)

        self.train_transform = T.Compose(data_transforms)
        self.eval_transform = T.Compose(eval_transform)

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj, self.eval_transform(sample)


class EvalTransform:
    def __init__(self, normalize=None):
        transforms = [T.ToTensor()]
        if normalize is not None:
            transforms.append(normalize)
        self.transforms = T.Compose(transforms)

    def __call__(self, x):
        return self.transforms(x), self.transforms(x), self.transforms(x)
