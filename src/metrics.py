import torch
import torch.nn as nn


def metric1(x1, x2):
    """
        pairwise cosine similarity among the representations of the
        validation examples.

        x1: (batch, dim)
        x2: (batch, dim)
    """
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cosine_similarity(x1, x2)


def metric2():
    """
        log p(z), where z = mean of q(z|x)
    """
    pass


def metric3():
    """
        log p(x|z) where z = mean of q(z|x)
    """
    pass


def metric4():
    """
        log p(x|z) + log p(z), where z = mean of q(z|x)
    """
    pass
