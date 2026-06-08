import numpy as np

class Transformation:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=int)
        self.size = self.matrix.shape[0]

    def apply(self, v):
        return self.matrix @ v
