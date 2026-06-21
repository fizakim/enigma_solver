from .base import EnigmaConfig, RotorConfig

alphabet10 = EnigmaConfig(
    alphabet="ABCDEFGHIJ",
    rotors=[
        RotorConfig(wiring="EIBFJDGCHA", notch="C"),
        RotorConfig(wiring="CHJGDIEFBA", notch="F"),
        RotorConfig(wiring="JAFHCEGIDB", notch="I"),
    ],
    reflector="BADCFEHGJI",
    plugboard_pairs=[]
)
