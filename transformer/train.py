import sys
import os
import math
from datetime import datetime
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformer.config import LMConfig
from transformer.model import CharTransformer
from transformer.data import load_corpus, get_batch, noise_inputs
from config.alphabet26 import alphabet26

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CONFIG = alphabet26
ALPHABET = CONFIG.alphabet

LM = LMConfig(
    vocab_size=len(ALPHABET),
    block_size=512,
    n_layer=12,
    n_head=8,
    d_model=512,
    dropout=0.1,
    tie_weights=True,
)

MAX_STEPS = 10_000
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 400
GRAD_CLIP = 1.0
EVAL_STEP = 500
EVAL_BATCHES = 20

P_SOFT = 0.5
MAX_NOISE = 0.6
FULL_UNIFORM_PROB = 0.05

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")
MODELS_DIR = os.path.join(_ROOT, "models")
_LN2 = math.log(2.0)

char_to_idx = {c: i for i, c in enumerate(ALPHABET)}

def lr_at(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.5 * LEARNING_RATE * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def eval_bpc(model, data, soft=False):
    model.eval()
    total = 0.0
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, LM.block_size, BATCH_SIZE, device)
        if soft:
            x = noise_inputs(x, LM.vocab_size, MAX_NOISE, FULL_UNIFORM_PROB)
        logits = model(x)
        total += F.cross_entropy(logits.reshape(-1, LM.vocab_size), y.reshape(-1)).item()
    model.train()
    return (total / EVAL_BATCHES) / _LN2

def main():
    train_data, val_data = load_corpus(CORPUS_PATH, char_to_idx)
    model = CharTransformer(LM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    model.train()
    for step in range(MAX_STEPS):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)

        x, y = get_batch(train_data, LM.block_size, BATCH_SIZE, device)
        if torch.rand(1).item() < P_SOFT:
            x = noise_inputs(x, LM.vocab_size, MAX_NOISE, FULL_UNIFORM_PROB)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, LM.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_STEP == 0 or step == MAX_STEPS - 1:
            hard_bpc = eval_bpc(model, val_data, soft=False)
            soft_bpc = eval_bpc(model, val_data, soft=True)
            print(f"step {step:>5d} | train loss {loss.item():.4f} | val bpc(hard) {hard_bpc:.3f} | val bpc(soft) {soft_bpc:.3f} | lr {lr_at(step):.2e}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODELS_DIR, f"transformer_lm_a{LM.vocab_size}_{timestamp}.pth")
    torch.save({"model": model.state_dict(), "config": LM.to_dict(), "alphabet": ALPHABET}, path)
    print(f"\nSaved checkpoint to '{path}'")

if __name__ == "__main__":
    main()
