import numpy as np
from .transformation import Transformation

class Rotor(Transformation):
    def __init__(self, matrix, notch, position=0):
        super().__init__(matrix)
        self.notch = notch
        self.position = position

    def step(self):
        at_notch = self.position == self.notch
        self.position = (self.position + 1) % self.size
        return at_notch

    def forward(self, v):
        return np.roll(self.apply(np.roll(v, self.position)), -self.position)

    def backward(self, v):
        return np.roll(self.matrix.T @ np.roll(v, self.position), -self.position)
