import torch


def gini_score(x):
    """
    Measures the sparsity of a tensor
    The closer to zero the score is, the LESS sparse the vector is
    ie: 0 ---> 1   less sparse ------> more sparse
    """
    # sort first + abs value
    x, _ = torch.sort(torch.abs(x), dim=1)

    N = x.size(1)
    b = x.size(0)

    # init the weight vector
    weight = torch.arange(1, N + 1).type_as(x)
    weight = (N - weight + 1 / 2) / N

    # normalize the vectors
    eps = 1e-8
    one_norms = torch.norm(x, p=1, dim=1).view(b, 1)
    x = x / (one_norms + eps)

    # calculate scores
    scores = torch.mv(x, weight).squeeze(-1)
    scores = 1 - (2 * scores)

    # (b) -> (b, 1)
    scores = scores.view(b, 1)

    return scores
