import torch
from config.base import EnigmaConfig

class TrainingConfig:
    def __init__(
        self,
        enigma_config: EnigmaConfig,
        loss_fn=torch.nn.CrossEntropyLoss(),
        trainable_rotors=None,
        trainable_reflector: bool = False,
    ):
        self.enigma_config = enigma_config
        self.loss_fn = loss_fn
        self.trainable_rotors = trainable_rotors
        self.trainable_reflector = trainable_reflector
