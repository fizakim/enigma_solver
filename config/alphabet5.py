import os
from .base import EnigmaConfig, RotorConfig
from language import Language

config_dir = os.path.dirname(__file__)
words5_path = os.path.join(config_dir, "..", "language", "words5.txt")

alphabet5 = EnigmaConfig(
    alphabet="ABCDE",
    rotors=[
        RotorConfig(wiring="ECADB", notch="C"),
        RotorConfig(wiring="ABEDC", notch="D"),
        RotorConfig(wiring="DBCEA", notch="E"),
    ],
    reflector="BADCE",
    plugboard_pairs=[],
    language=Language(words5_path)
)
