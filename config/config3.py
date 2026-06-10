import os
from .base import EnigmaConfig, RotorConfig
from language import Language

# Get the path to abc.txt relative to the config file
config_dir = os.path.dirname(__file__)
abc_path = os.path.join(config_dir, "..", "language", "abc.txt")

config3 = EnigmaConfig(
    alphabet="ABC",
    rotors=[
        RotorConfig(wiring="BAC", notch="B"),
        RotorConfig(wiring="ACB", notch="B"),
        RotorConfig(wiring="CBA", notch="B"),
    ],
    reflector="BAC",
    plugboard_pairs=[],
    language=Language(abc_path)
)

