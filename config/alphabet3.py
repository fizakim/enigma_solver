import os
from .base import EnigmaConfig, RotorConfig
from language import Language

config_dir = os.path.dirname(__file__)
words3_path = os.path.join(config_dir, "..", "language", "words3.txt")

alphabet3 = EnigmaConfig(
    alphabet="ABC",
    rotors=[
        RotorConfig(wiring="BAC", notch="B"),
        RotorConfig(wiring="ACB", notch="B"),
        RotorConfig(wiring="CBA", notch="B"),
    ],
    reflector="BAC",
    plugboard_pairs=[],
    language=Language(words3_path)
)
