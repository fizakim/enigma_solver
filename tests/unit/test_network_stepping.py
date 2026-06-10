import sys
sys.path.append(".")
from config.config3 import config3
from enigma_net.enigma_net import EnigmaNet

net = EnigmaNet(config3)

net.reset([0, 0, 0])

net.step()
assert net.positions == [0, 0, 1]

net.step()
assert net.positions == [0, 1, 2]

net.step()
assert net.positions == [0, 1, 0]

net.step()
assert net.positions == [0, 1, 1]

net.step()
assert net.positions == [1, 2, 2]

print("Passed")
