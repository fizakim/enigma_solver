from config.alphabet3 import alphabet3
from enigma_net.permutation.basic.permutation_loss import PermutationLoss
from enigma_net.train_config import TrainConfig

alphabet3_permutation = TrainConfig(
    enigma_config=alphabet3,
    loss_fn=PermutationLoss(),
    trainable_rotors=None,
    trainable_reflector=True,
)
