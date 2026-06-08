from .base import EnigmaConfig, RotorConfig

config26 = EnigmaConfig(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    rotors=[
        RotorConfig(wiring="EKMFLGDQVZNTOWYHXUSPAIBRCJ", notch="Q"),
        RotorConfig(wiring="AJDKSIRUXBLHWTMCQGZNPYFVOE", notch="E"),
        RotorConfig(wiring="BDFHJLCPRTXVZNYEIWGAKMUSQO", notch="V"),
    ],
    reflector="YRUHQSLDPXNGOKMIEBFZCWVJAT",
    plugboard_pairs=[]
)

def build(positions=None):
    return config26.build(positions)

def make_enigma(positions=None):
    return config26.build(positions)
