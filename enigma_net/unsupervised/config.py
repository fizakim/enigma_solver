from config.alphabet3 import alphabet3
from enigma_net import NgramLoss
from enigma_net.train_config import TrainConfig
from n_gram.generator import load_ngram_counts

NGRAM_COUNTS_PATH = "n_gram/counts3.pth"

ngram_counts = load_ngram_counts(NGRAM_COUNTS_PATH)

alphabet3_unsupervised = TrainConfig(
    enigma_config=alphabet3,
    loss_fn=NgramLoss(ngram_counts),
    trainable_rotors=None,
    trainable_reflector=True,
)
