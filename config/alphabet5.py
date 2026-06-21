from .base import EnigmaConfig, RotorConfig

alphabet5 = EnigmaConfig(
    alphabet="ABCDE",
    rotors=[
        RotorConfig(wiring="ECADB", notch="C"),
        RotorConfig(wiring="ABEDC", notch="D"),
        RotorConfig(wiring="DBCEA", notch="E"),
    ],
    reflector="BADCE",
    plugboard_pairs=[]
)
