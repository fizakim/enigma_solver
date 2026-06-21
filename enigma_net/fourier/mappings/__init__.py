from .softmax import SoftmaxMapping
from .linear import LinearMapping

def get_mapping(mapping_type, size):
    if mapping_type == "softmax":
        return SoftmaxMapping(size)
    if mapping_type == "linear":
        return LinearMapping(size)
    raise ValueError(f"Unknown mapping type: '{mapping_type}'")
