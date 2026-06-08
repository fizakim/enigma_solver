from __future__ import annotations
import numpy as np
from .transformation import Transformation

class Plugboard(Transformation):
    def __init__(
        self,
        wiring_matrix: np.ndarray | None = None,
        size: int = 26,
    ) -> None:
        if wiring_matrix is None:
            wiring_matrix = np.eye(size, dtype=int)
        super().__init__(wiring_matrix)

    def swap(self, v: np.ndarray) -> np.ndarray:
        return self.apply(v)

