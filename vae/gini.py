import torch
import torchvision.transforms as T
import argparse

from pl_bolts.datamodules import CIFAR10DataModule

from vae import VAE


@torch.no_grad()
def gini_sparsity(model, dl):

    model.eval()
    results = []
    for batch in dl:
        x, _ = batch
        x = x.type_as(model.encoder.conv1.weight)
        z = model(x)[0]
        scores = gini_score(z)
        results.append(scores)
    results = torch.cat(results).view(-1)
    score = results.mean()
    return results, score


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrained")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    dm = CIFAR10DataModule(data_dir="data", batch_size=args.batch_size, num_workers=6)
    dm.test_transforms = T.ToTensor()
    dm.val_transforms = T.ToTensor()

    vae = VAE.load_from_checkpoint(args.pretrained).cuda()
    results, score = gini_sparsity(vae, dm.val_dataloader())
    print(f"gini sparsity: {score}")
