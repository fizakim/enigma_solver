import torch
from enigma_net.loss import LossFunction

class NoFixedPointLoss(LossFunction):
    def forward(self, model, inputs, positions):
        model.reset(positions)
        outputs = []
        for v in inputs:
            outputs.append(model(v))

        inputs_t = torch.stack(inputs)
        outputs_t = torch.stack(outputs)
        return (inputs_t * outputs_t).sum(dim=-1).mean()
