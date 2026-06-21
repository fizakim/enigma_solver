from .base import EnigmaConfig, RotorConfig

alphabet15 = EnigmaConfig(
    alphabet="ABCDEFGHIJKLMNO",
    rotors=[
        RotorConfig(wiring="HMAGFBNCDKJEOIL", notch="D"),
        RotorConfig(wiring="EJBINGMCADHOLKF", notch="H"),
        RotorConfig(wiring="DFJHLGENMAKOICB", notch="L"),
    ],
    reflector="BADCFEHGJILKMNO",
    plugboard_pairs=[]
)
