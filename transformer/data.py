import torch
import torch.nn.functional as F


def load_corpus(path, char_to_idx, val_frac=0.0005):
    """Read the corpus and return (train_ids, val_ids) as 1-D LongTensors.

    Only characters present in the alphabet are kept and the whole corpus is
    concatenated into a single id stream (mirrors the q_net trainers' `"".join`).
    A small tail is held out for validation.
    """
    with open(path, "r", encoding="utf-8") as f:
        ids = [char_to_idx[c] for c in f.read() if c in char_to_idx]
    data = torch.tensor(ids, dtype=torch.long)
    n_val = max(1, int(len(data) * val_frac))
    return data[:-n_val], data[-n_val:]


def get_batch(data, block_size, batch_size, device):
    """Sample `batch_size` contiguous windows. Returns (x, y) of shape [B, block_size].

    y is x shifted by one position (standard next-character targets).
    """
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def noise_inputs(ids, vocab_size, max_noise=0.6, full_uniform_prob=0.05):
    """Turn hard token ids [B, T] into noised soft inputs [B, T, vocab].

    Each position's one-hot is blended toward the uniform distribution by a random
    amount `a ~ U(0, max_noise)`, and with probability `full_uniform_prob` a position
    is made fully uniform. Targets stay the hard next char, so the LM learns to
    predict through blur — making the frozen scorer robust to the high-entropy soft
    decode `d` the q_net produces early in training.
    """
    onehot = F.one_hot(ids, vocab_size).float()              # [B, T, V]
    a = torch.rand(ids.shape + (1,), device=ids.device) * max_noise
    if full_uniform_prob > 0:
        full = (torch.rand(ids.shape + (1,), device=ids.device) < full_uniform_prob)
        a = torch.where(full, torch.ones_like(a), a)
    uniform = 1.0 / vocab_size
    return (1.0 - a) * onehot + a * uniform
