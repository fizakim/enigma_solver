import os
from .base import EnigmaConfig, RotorConfig
from language import Language

config_dir = os.path.dirname(__file__)
abcdefghij_path = os.path.join(config_dir, "..", "language", "abcdefghij.txt")

config10 = EnigmaConfig(
    alphabet="ABCDEFGHIJ",
    rotors=[
        RotorConfig(wiring="EIBFJDGCHA", notch="C"),
        RotorConfig(wiring="CHJGDIEFBA", notch="F"),
        RotorConfig(wiring="JAFHCEGIDB", notch="I"),
    ],
    reflector="BADCFEHGJI",
    plugboard_pairs=[],
    language=Language(abcdefghij_path)
)

