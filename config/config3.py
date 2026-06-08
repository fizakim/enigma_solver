import os
from .base import EnigmaConfig, RotorConfig
from language import Language

# Get the path to abc.txt relative to the config file
config_dir = os.path.dirname(__file__)
abc_path = os.path.join(config_dir, "..", "language", "abc.txt")

config3 = EnigmaConfig(
    alphabet="ABC",
    rotors=[
        RotorConfig(wiring="BCA", notch="B"),
        RotorConfig(wiring="CAB", notch="B"),
        RotorConfig(wiring="ABC", notch="B"),
    ],
    reflector="BAC",
    plugboard_pairs=[],
    language=Language(abc_path)
)

