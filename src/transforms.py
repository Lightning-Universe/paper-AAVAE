import torchvision.transforms as transforms
import cv2
import numpy as np


class LocalTransform:
    def __init__(
        self,
        input_height: int = 224,
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
        normalize=None,
    ) -> None:

        self.color_jitter = transforms.ColorJitter(
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.2 * jitter_strength,
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if gaussian_blur:
            kernel_size = int(0.1 * input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(
                transforms.RandomApply([GaussianBlur(kernel_size=kernel_size)], p=0.5)
            )

        data_transforms = transforms.Compose(data_transforms)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose(
                [transforms.ToTensor(), normalize]
            )

        self.transform = transforms.Compose([data_transforms, self.final_transform])

    def __call__(self, x):
        return self.transform(x)


class OriginalTransform(object):
    def __init__(self, input_height: int = 224, normalize=None) -> None:

        self.input_height = input_height
        self.normalize = normalize

        data_transforms = [
            transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
            transforms.CenterCrop(self.input_height),
        ]

        data_transforms = transforms.Compose(data_transforms)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose(
                [transforms.ToTensor(), normalize]
            )

        self.transform = transforms.Compose([data_transforms, self.final_transform])

    def __call__(self, x):
        return self.transform(x)


class GaussianBlur:
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min_sigma=0.1, max_sigma=2.0):
        self.min = min_sigma
        self.max = max_sigma
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(
                sample, (self.kernel_size, self.kernel_size), sigma
            )

        return sample
