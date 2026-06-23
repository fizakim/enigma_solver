import sys
import os
import math
import glob
from collections import Counter
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformer.loss import load_transformer_lm
from transformer.data import load_corpus, get_batch
from config.alphabet26 import alphabet26

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")
MODELS_DIR = os.path.join(_ROOT, "models")
_LN2 = math.log(2.0)

EVAL_BATCHES = 50
BATCH_SIZE = 64
NGRAM_TRAIN_CAP = 3_000_000   # chars used to estimate the n-gram baselines


def latest_checkpoint():
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "transformer_lm_*.pth")))
    if not paths:
        raise FileNotFoundError("No transformer_lm_*.pth in models/. Run transformer/train.py first.")
    return paths[-1]


@torch.no_grad()
def lm_bpc(model, val_data, block_size):
    total = 0.0
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(val_data, block_size, BATCH_SIZE, device)
        logits = model(x)
        total += F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size), y.reshape(-1)).item()
    return (total / EVAL_BATCHES) / _LN2


def ngram_bpc(train_data, val_data, k, vocab, alpha=1.0):
    """Conditional (k-1)-order n-gram bits-per-char with add-alpha smoothing."""
    train = train_data[:NGRAM_TRAIN_CAP].tolist()
    ctx_counts, joint_counts = Counter(), Counter()
    for i in range(len(train) - k + 1):
        ctx = 0
        for j in range(k - 1):
            ctx = ctx * vocab + train[i + j]
        ctx_counts[ctx] += 1
        joint_counts[ctx * vocab + train[i + k - 1]] += 1

    val = val_data.tolist()
    nll = 0.0
    count = 0
    for i in range(len(val) - k + 1):
        ctx = 0
        for j in range(k - 1):
            ctx = ctx * vocab + val[i + j]
        w = val[i + k - 1]
        p = (joint_counts[ctx * vocab + w] + alpha) / (ctx_counts[ctx] + alpha * vocab)
        nll += -math.log2(p)
        count += 1
    return nll / count


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else latest_checkpoint()
    print(f"Loading {ckpt}")
    model = load_transformer_lm(ckpt, device)
    alphabet = alphabet26.alphabet
    char_to_idx = {c: i for i, c in enumerate(alphabet)}

    train_data, val_data = load_corpus(CORPUS_PATH, char_to_idx)

    print("\n=== Bits-per-char on held-out validation (lower is better) ===")
    print(f"transformer LM : {lm_bpc(model, val_data, model.cfg.block_size):.3f}")
    for k in (1, 3, 4):
        print(f"{k}-gram baseline: {ngram_bpc(train_data, val_data, k, model.cfg.vocab_size):.3f}")

    print("\n=== Sample generation (greedy + temperature) ===")
    seed = "THE"
    idx = torch.tensor([[char_to_idx[c] for c in seed]], dtype=torch.long, device=device)
    for label, kw in (("greedy", dict(greedy=True)), ("temp=0.8", dict(temperature=0.8))):
        out = model.generate(idx.clone(), max_new_tokens=200, **kw)[0].tolist()
        print(f"[{label}] {''.join(alphabet[i] for i in out)}")


if __name__ == "__main__":
    main()
