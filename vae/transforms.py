import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class RandomRotate:
    def __init__(self, degrees=[0, 90, 120, 270]):
        self.degrees = degrees

    def __call__(self, x):
        angle = random.choice(self.degrees)
        return TF.rotate(x, angle)


class TrainTransforms:
    def __init__(self, normalize=None):
        self.transforms = T.Compose([RandomRotate()])
        post = [T.ToTensor()]
        if normalize is not None:
            post.append(normalize)
        self.post = T.Compose(post)

    def __call__(self, x):
        return self.post(self.transforms(x)), self.post(x)


class EvalTransforms:
    def __init__(self, normalize=None):
        post = [T.ToTensor()]
        if normalize is not None:
            post.append(normalize)
        self.post = T.Compose(post)

    def __call__(self, x):
        return self.post(x), self.post(x)
