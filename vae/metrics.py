import torch


def marginal_logpx(z, x_hat, p, q, N):
    """
    Evaluate marginal log p(x) using importance sampling

    \begin{align}
    \log p(x) &= E_p[p(x|z)]\\
    &=\log(\int p(x|z) p(z) dz) \\
    &= \log(\int p(x|z) p(z) / q(z|x) q(z|x) dz) \\
    &= E_q[p(x|z) p(z) / q(z|x)] \\
    &\approx \log(1/n * \sum_i p(x|z_i) p(z_i)/q(z_i)) \\
    &\approx \log(1/N * \sum_i \exp({\log p(x|z_i) + \log p(z_i) - \log q(z_i)}) \\
    &\approx \text{logsumexp}(\log p(x|z_i) + \log p(z_i) - \log q(z_i)) - \log N
    \end{align}

    Example:

        mu, log_var = encoder(x)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        x_hat = decoder(z)

        logpx = marginal_logpx(z, x_hat, p, q, N=1000)
    """
    log_pz = p.log_prob(z).sum(dim=1)
    log_qz = q.log_prob(z).sum(dim=1)
    log_pxz = torch.log(x_hat).sum(dim=(1, 2, 3))

    return torch.logsumexp(log_pxz + log_pz - log_qz) - torch.log(N)


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
