from config.alphabet3 import alphabet3
from config.alphabet26 import alphabet26
from enigma_net import CrossEntropyLoss
from enigma_net.train_config import TrainConfig

dft_config = TrainConfig(
    enigma_config=alphabet3,
    loss_fn=CrossEntropyLoss(),
    trainable_rotors=None,
    trainable_reflector=False,
)

alphabet26_config = TrainConfig(
    enigma_config=alphabet26,
    loss_fn=CrossEntropyLoss(),
    trainable_rotors=None,
    trainable_reflector=False,
)