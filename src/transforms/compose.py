from src.transforms.transforms import (
    SimCLRTransform,
    LinearEvalTrainTransform,
    LinearEvalValidTransform,
    OriginalTransform
)


class TrainTransform:
    """
    TrainTransform returns a transformed image along with the original
    """

    def __init__(
        self,
        denoising: bool = True,
        input_height: int = 32,
        dataset="cifar10",
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
        normalize=None,
        online_ft: bool = False,
    ) -> None:
        self.online_ft = online_ft

        if denoising:
            self.input_transform = SimCLRTransform(
                input_height=input_height,
                jitter_strength=jitter_strength,
                gaussian_blur=gaussian_blur,
                normalize=normalize,
            )
        else:
            self.input_transform = OriginalTransform(
                dataset=dataset, normalize=normalize
            )

        self.original_transform = OriginalTransform(
            dataset=dataset, normalize=normalize
        )

        if self.online_ft:
            self.finetune_transform = LinearEvalTrainTransform(
                dataset=dataset, normalize=normalize
            )

    def __call__(self, x):
        train_transforms = [self.input_transform(x), self.original_transform(x)]

        if self.online_ft:
            train_transforms.append(self.finetune_transform(x))

        return train_transforms


class EvalTransform:
    """
    EvalTransform returns the orginial image twice
    """

    def __init__(
        self,
        denoising: bool = True,
        input_height: int = 224,
        dataset="cifar10",
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
        normalize=None,
        online_ft: bool = False,
    ) -> None:
        self.online_ft = online_ft

        if denoising:
            self.input_transform = SimCLRTransform(
                input_height=input_height,
                jitter_strength=jitter_strength,
                gaussian_blur=gaussian_blur,
                normalize=normalize,
            )
        else:
            self.input_transform = OriginalTransform(
                dataset=dataset, normalize=normalize
            )

        self.original_transform = OriginalTransform(
            dataset=dataset, normalize=normalize
        )

        if self.online_ft:
            self.finetune_transform = LinearEvalValidTransform(
                dataset=dataset, normalize=normalize
            )

    def __call__(self, x):
        val_transforms = [self.input_transform(x), self.original_transform(x)]

        if self.online_ft:
            val_transforms.append(self.finetune_transform(x))

        return val_transforms
