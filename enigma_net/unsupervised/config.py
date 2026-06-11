from config.config3 import config3
from enigma_net.train_config import TrainingConfig

config3_unsupervised = TrainingConfig(
    enigma_config=config3,
    loss_fn=None,  
    trainable_rotors=None,
    trainable_reflector=False,
)
