import math


# TODO: add step decay
def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
    """
    Linear warmup for warmup_steps, optionally with cosine annealing or
    linear decay to 0 at total_steps
    """
    # check if both decays are not True at the same time
    assert not (linear and cosine)

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn
