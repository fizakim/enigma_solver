from dataclasses import dataclass, asdict

@dataclass
class LMConfig:
    vocab_size: int = 26
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    dropout: float = 0.1
    tie_weights: bool = True
    causal: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})
