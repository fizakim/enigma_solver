import sys
sys.path.append(".")
import numpy as np
from enigma.rotor import Rotor

m = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
])

r = Rotor(m, notch=1, position=0)

assert r.step() is False
assert r.position == 1

assert r.step() is True
assert r.position == 2

assert r.step() is False
assert r.position == 0

r = Rotor(m, notch=1, position=0)
v = np.array([1, 0, 0])
assert np.all(r.forward(v) == np.array([0, 1, 0]))
assert np.all(r.backward(np.array([0, 1, 0])) == v)

r.step()
assert np.all(r.forward(v) == np.array([0, 1, 0]))
assert np.all(r.backward(np.array([0, 1, 0])) == v)

print("Passed")
