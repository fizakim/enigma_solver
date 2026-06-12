import os
from .base import EnigmaConfig, RotorConfig
from language import Language

config_dir = os.path.dirname(__file__)
words10_path = os.path.join(config_dir, "..", "language", "words10.txt")

alphabet10 = EnigmaConfig(
    alphabet="ABCDEFGHIJ",
    rotors=[
        RotorConfig(wiring="EIBFJDGCHA", notch="C"),
        RotorConfig(wiring="CHJGDIEFBA", notch="F"),
        RotorConfig(wiring="JAFHCEGIDB", notch="I"),
    ],
    reflector="BADCFEHGJI",
    plugboard_pairs=[],
    language=Language(words10_path)
)
