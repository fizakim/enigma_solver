import math
import torch

# The tables in language/ngram/ store log10 probabilities (see language/ngram/ngrams.py).
# We convert to natural log so the n-gram loss magnitude (and therefore its gradient scale)
# matches the existing cross-entropy path, which uses natural-log probabilities.
_LN10 = math.log(10.0)


def load_ngram_logprobs(path, alphabet_size=26, device="cpu"):
    """Load a saved n-gram log-probability table as natural-log probs.

    Args:
        path: path to a `{n}grams.pth` tensor of shape [alphabet_size] * n.
        alphabet_size: expected size of each dimension (26 for the full A-Z table).
        device: device to place the returned tensor on.

    Returns:
        A float32 tensor of shape [alphabet_size] * n holding natural-log probabilities.
    """
    log10_probs = torch.load(path, map_location="cpu")

    expected = tuple([alphabet_size] * log10_probs.ndim)
    if tuple(log10_probs.shape) != expected:
        raise ValueError(
            f"N-gram table at {path} has shape {tuple(log10_probs.shape)}, "
            f"expected {expected} for alphabet_size={alphabet_size}."
        )

    return (log10_probs.float() * _LN10).contiguous().to(device)
