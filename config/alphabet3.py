from .base import EnigmaConfig, RotorConfig

alphabet3 = EnigmaConfig(
    alphabet="ABC",
    rotors=[
        RotorConfig(wiring="BAC", notch="B"),
        RotorConfig(wiring="ACB", notch="B"),
        RotorConfig(wiring="CBA", notch="B"),
    ],
    reflector="BAC",
    plugboard_pairs=[]
)
