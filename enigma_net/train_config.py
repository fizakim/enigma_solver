import torch
from config.base import EnigmaConfig
from .supervised.cross_entropy import CrossEntropyLoss

_DEFAULT = object()

class TrainConfig:
    def __init__(
        self,
        enigma_config: EnigmaConfig,
        loss_fn=_DEFAULT,
        trainable_rotors=None,
        trainable_reflector: bool = False,
    ):
        self.enigma_config = enigma_config
        self.loss_fn = CrossEntropyLoss() if loss_fn is _DEFAULT else loss_fn
        self.trainable_rotors = trainable_rotors
        self.trainable_reflector = trainable_reflector
