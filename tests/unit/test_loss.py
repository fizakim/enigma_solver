import sys
sys.path.append(".")
import torch
import torch.nn as nn
from enigma_net import LossFunction, CrossEntropyLoss, NoFixedPointLoss

def test_loss_interface():
    # Verify inheritance and class definitions
    assert issubclass(CrossEntropyLoss, LossFunction)
    assert issubclass(NoFixedPointLoss, LossFunction)
    assert issubclass(LossFunction, nn.Module)

def test_cross_entropy_loss_values():
    # Setup inputs
    predictions = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    targets = torch.tensor([0], dtype=torch.long)
    
    # Custom loss
    custom_loss_fn = CrossEntropyLoss()
    custom_loss = custom_loss_fn(predictions, targets)
    
    # Native loss
    native_loss_fn = nn.CrossEntropyLoss()
    native_loss = native_loss_fn(predictions, targets)
    
    # Check output values are identical
    assert torch.allclose(custom_loss, native_loss)
    
    # Check gradients propagate
    custom_loss.backward()
    assert predictions.grad is not None
    assert predictions.grad.shape == predictions.shape
    print("test_cross_entropy_loss_values passed")

def test_no_fixed_point_loss():
    class MockModel:
        def __init__(self):
            self.outputs = [
                torch.tensor([0.1, 0.8, 0.1], requires_grad=True),
                torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
            ]
            self.call_count = 0
            
        def reset(self, positions):
            pass
            
        def __call__(self, v):
            out = self.outputs[self.call_count]
            self.call_count += 1
            return out

    model = MockModel()
    inputs = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0])
    ]
    positions = [0, 0]
    
    loss_fn = NoFixedPointLoss()
    
    # Expect loss = (0.1 + 0.5) / 2 = 0.3
    loss = loss_fn(model, inputs, positions)
    assert torch.allclose(loss, torch.tensor(0.3))
    
    # Check gradients propagate
    loss.backward()
    assert model.outputs[0].grad is not None
    assert torch.allclose(model.outputs[0].grad, torch.tensor([0.5, 0.0, 0.0]))
    print("test_no_fixed_point_loss passed")

if __name__ == "__main__":
    test_loss_interface()
    test_cross_entropy_loss_values()
    test_no_fixed_point_loss()
    print("All tests passed!")
