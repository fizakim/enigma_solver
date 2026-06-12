import sys
sys.path.append(".")
import torch
import torch.nn as nn
from enigma_net import LossFunction, CrossEntropyLoss

def test_loss_interface():
    # Verify inheritance and class definitions
    assert issubclass(CrossEntropyLoss, LossFunction)
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

if __name__ == "__main__":
    test_loss_interface()
    test_cross_entropy_loss_values()
    print("All tests passed!")
