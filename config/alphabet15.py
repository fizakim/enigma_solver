import os
from .base import EnigmaConfig, RotorConfig
from language import Language

config_dir = os.path.dirname(__file__)
words15_path = os.path.join(config_dir, "..", "language", "words15.txt")

alphabet15 = EnigmaConfig(
    alphabet="ABCDEFGHIJKLMNO",
    rotors=[
        RotorConfig(wiring="HMAGFBNCDKJEOIL", notch="D"),
        RotorConfig(wiring="EJBINGMCADHOLKF", notch="H"),
        RotorConfig(wiring="DFJHLGENMAKOICB", notch="L"),
    ],
    reflector="BADCFEHGJILKMNO",
    plugboard_pairs=[],
    language=Language(words15_path)
)
