from dataclasses import dataclass, asdict


@dataclass
class LMConfig:
    """Architecture/config for the character-level transformer language model.

    The full config is saved inside each checkpoint (see transformer/train.py) so
    the loader can rebuild an identical model without guessing hyper-parameters.

    vocab_size is the alphabet size (n). For the unsupervised q_net loss this must
    match len(config.alphabet) — 26 for the full A-Z setup.
    """

    vocab_size: int = 26
    block_size: int = 128          # context length the LM is trained/scored on
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    dropout: float = 0.1
    tie_weights: bool = True       # share the output head with the token embedding

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        # Tolerate extra keys from future checkpoints; ignore unknown fields.
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})
