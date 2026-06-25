import sys
import os
import math
import glob
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformer.loss import load_transformer_lm, TransformerLoss
from config.alphabet26 import alphabet26

STRING = "IWANTTOKNOWHOWITSGOING"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(_ROOT, "models")
_LN2 = math.log(2.0)

def latest_checkpoint():
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "transformer_lm_*.pth")))
    if not paths:
        raise FileNotFoundError("No checkpoint found.")
    return paths[-1]

def score(string, model, loss_fn, char_to_idx):
    ids = [char_to_idx[c] for c in string.upper() if c in char_to_idx]
    if len(ids) < 2:
        raise ValueError("Need at least 2 alphabet characters to score.")
    ids_t = torch.tensor(ids, dtype=torch.long, device=device)

    logits = (F.one_hot(ids_t, model.cfg.vocab_size).float() * 30.0).unsqueeze(0)
    with torch.no_grad():
        loss_nats = loss_fn(logits).item()
        lm_logits = model(ids_t.unsqueeze(0))
        ce = F.cross_entropy(lm_logits[0, :-1], ids_t[1:]).item()
    return loss_nats, ce / _LN2, len(ids)

def main():
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else STRING
    ckpt = latest_checkpoint()
    print(f"Loading {ckpt}")
    model = load_transformer_lm(ckpt, device)
    loss_fn = TransformerLoss(model, tau=0.5)
    char_to_idx = {c: i for i, c in enumerate(alphabet26.alphabet)}

    loss_nats, bpc, n_used = score(text, model, loss_fn, char_to_idx)
    print(f"\nstring : {text.upper()[:80]}{'...' if len(text) > 80 else ''}")
    print(f"chars  : {n_used} alphabet characters scored")
    print(f"score  : {loss_nats:.4f} nats/char   ({bpc:.3f} bits/char)")

if __name__ == "__main__":
    main()
