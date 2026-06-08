from .base import EnigmaConfig, RotorConfig

config3 = EnigmaConfig(
    alphabet="ABC",
    rotors=[
        RotorConfig(wiring="BCA", notch="B"),
        RotorConfig(wiring="CAB", notch="B"),
        RotorConfig(wiring="ABC", notch="B"),
    ],
    reflector="BAC",
    plugboard_pairs=[]
)

def build(positions=None):
    return config3.build(positions)

def make_enigma(positions=None):
    return config3.build(positions)
