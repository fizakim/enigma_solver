from config.alphabet3 import alphabet3
from enigma_net.permutation.plackett_luce.loss import PlackettLuceLoss
from enigma_net.train_config import TrainConfig

alphabet3_plackett_luce = TrainConfig(
    enigma_config=alphabet3,
    loss_fn=PlackettLuceLoss(),
    trainable_rotors=None,
    trainable_reflector=True,
)
