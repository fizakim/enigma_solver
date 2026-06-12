from config.alphabet3 import alphabet3
from enigma_net import CrossEntropyLoss
from enigma_net.train_config import TrainConfig

alphabet3_supervised = TrainConfig(
    enigma_config=alphabet3,
    loss_fn=CrossEntropyLoss(),
    trainable_rotors=None,
    trainable_reflector=True,
)
