import sys
import os
import glob
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from transformer import LMConfig, CharTransformer, TransformerLoss, load_transformer_lm
from enigma_net import LossFunction


def _tiny_model(vocab=8, block_size=16):
    cfg = LMConfig(vocab_size=vocab, block_size=block_size, n_layer=2, n_head=2,
                   d_model=32, dropout=0.0)
    return CharTransformer(cfg).eval()


def test_is_loss_function():
    model = _tiny_model()
    assert issubclass(TransformerLoss, LossFunction)
    assert TransformerLoss(model).requires_full_sequence is True
    print("test_is_loss_function passed")


def test_dual_input_shapes():
    """The model accepts both hard ids [B,T] and soft dists [B,T,vocab]."""
    model = _tiny_model(vocab=8, block_size=16)
    ids = torch.randint(0, 8, (3, 12))
    soft = torch.rand(3, 12, 8)
    soft = soft / soft.sum(-1, keepdim=True)
    assert model(ids).shape == (3, 12, 8)
    assert model(soft).shape == (3, 12, 8)
    print("test_dual_input_shapes passed")


def test_hard_soft_equivalence():
    """Feeding one_hot(ids) must equal feeding ids (the expected-embedding bridge)."""
    model = _tiny_model(vocab=8, block_size=16)
    ids = torch.randint(0, 8, (3, 12))
    with torch.no_grad():
        logits_hard = model(ids)
        logits_soft = model(F.one_hot(ids, 8).float())
    assert torch.allclose(logits_hard, logits_soft, atol=1e-5), \
        (logits_hard - logits_soft).abs().max().item()
    print("test_hard_soft_equivalence passed")


def test_loss_shape_grad_and_freezing():
    """Loss returns finite [B]; grad flows to the input logits but not to LM params,
    and the long-T windowing path (T > block_size) runs."""
    vocab, block = 8, 16
    model = _tiny_model(vocab, block)
    loss_fn = TransformerLoss(model, tau=0.5)

    B, T = 3, 40  # T > block_size -> exercises the unfold/windowing path
    logits = torch.randn(B, T, vocab, requires_grad=True)
    out = loss_fn(logits)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()

    out.mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0, "no gradient reached the input decode"
    # The frozen scorer must not accumulate gradient on its own parameters.
    assert all(p.grad is None for p in loss_fn.lm.parameters())
    print("test_loss_shape_grad_and_freezing passed")


def test_english_ranking():
    """With a pretrained checkpoint, real English must score lower than shuffled text."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpts = sorted(glob.glob(os.path.join(root, "models", "transformer_lm_*.pth")))
    if not ckpts:
        print("test_english_ranking SKIPPED (no transformer_lm_*.pth checkpoint)")
        return

    from config.alphabet26 import alphabet26
    from transformer.data import load_corpus

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = load_transformer_lm(ckpts[-1], device)
    loss_fn = TransformerLoss(lm, tau=0.5)
    n = lm.cfg.vocab_size

    char_to_idx = {c: i for i, c in enumerate(alphabet26.alphabet)}
    corpus_path = os.path.join(root, "language", "fineweb", "fineweb.txt")
    _, val = load_corpus(corpus_path, char_to_idx)
    T = min(512, len(val) - 1)
    eng = val[:T].to(device)
    shuf = eng[torch.randperm(T, device=device)]

    # Near-one-hot logits so softmax(.) reproduces the chosen sequence.
    def logits_of(ids):
        return (F.one_hot(ids, n).float() * 10.0).unsqueeze(0)

    with torch.no_grad():
        eng_loss = loss_fn(logits_of(eng)).item()
        shuf_loss = loss_fn(logits_of(shuf)).item()
    print(f"test_english_ranking: english={eng_loss:.3f} < shuffled={shuf_loss:.3f}")
    assert eng_loss < shuf_loss


if __name__ == "__main__":
    test_is_loss_function()
    test_dual_input_shapes()
    test_hard_soft_equivalence()
    test_loss_shape_grad_and_freezing()
    test_english_ranking()
    print("All tests passed!")
