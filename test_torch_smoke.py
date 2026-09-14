# tests/test_torch_smoke.py
import torch

def test_backward_gives_expected_grad():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x**2).sum()
    y.backward()
    assert torch.allclose(x.grad, torch.tensor([2.0, 4.0, 6.0]))