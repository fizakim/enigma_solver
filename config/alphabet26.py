from .base import EnigmaConfig, RotorConfig

alphabet26 = EnigmaConfig(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    rotors=[
        RotorConfig(wiring="EKMFLGDQVZNTOWYHXUSPAIBRCJ", notch="Q"),
        RotorConfig(wiring="AJDKSIRUXBLHWTMCQGZNPYFVOE", notch="E"),
        RotorConfig(wiring="BDFHJLCPRTXVZNYEIWGAKMUSQO", notch="V"),
    ],
    reflector="YRUHQSLDPXNGOKMIEBFZCWVJAT",
    plugboard_pairs=[]
)
