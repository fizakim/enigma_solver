import torch
from config.config3 import config3
from enigma_net.train_config import TrainingConfig

config3_supervised = TrainingConfig(
    enigma_config=config3,
    loss_fn=torch.nn.CrossEntropyLoss(),
    trainable_rotors=None,
    trainable_reflector=True,
)
