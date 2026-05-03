import pytest
import torch

from jepa.loss import jepa_loss, vicreg_loss


def test_vicreg_collapse_signal():
    """Constant inputs (collapse) produce a much larger penalty than diverse inputs."""
    z_collapsed = torch.zeros(32, 64)
    torch.manual_seed(0)
    z_healthy = torch.randn(32, 64)
    assert vicreg_loss(z_collapsed) > vicreg_loss(z_healthy) * 5


def test_jepa_loss_predicted_flows_gradient():
    """vicreg_target='predicted' must let gradients flow into predicted."""
    predicted = torch.randn(8, 16, 32, requires_grad=True)
    target = torch.randn(8, 16, 32)
    mse, reg = jepa_loss(predicted, target, vicreg_target="predicted")
    (mse + reg).backward()
    assert predicted.grad is not None
    assert predicted.grad.abs().sum() > 0


def test_jepa_loss_predicted_vicreg_alone_flows_gradient():
    """The reg term itself must produce gradient (not just because MSE does)."""
    predicted = torch.randn(8, 16, 32, requires_grad=True)
    target = torch.randn(8, 16, 32)
    _, reg = jepa_loss(predicted, target, vicreg_target="predicted")
    reg.backward()
    assert predicted.grad is not None
    assert predicted.grad.abs().sum() > 0


def test_jepa_loss_target_mode_no_gradient_from_reg():
    """vicreg_target='target' makes reg purely diagnostic — `reg` has no
    grad_fn (entire input detached), so backward through it is impossible.
    """
    predicted = torch.randn(8, 16, 32, requires_grad=True)
    target = torch.randn(8, 16, 32, requires_grad=True)
    _, reg = jepa_loss(predicted, target, vicreg_target="target")
    assert not reg.requires_grad
    assert reg.grad_fn is None


def test_jepa_loss_target_never_receives_mse_gradient():
    """target is always detached; MSE shouldn't propagate into it."""
    predicted = torch.randn(8, 16, 32, requires_grad=True)
    target = torch.randn(8, 16, 32, requires_grad=True)
    mse, _ = jepa_loss(predicted, target)
    mse.backward()
    assert target.grad is None or target.grad.abs().sum() == 0


def test_jepa_loss_invalid_vicreg_target_raises():
    predicted = torch.randn(8, 16, 32)
    target = torch.randn(8, 16, 32)
    with pytest.raises(ValueError):
        jepa_loss(predicted, target, vicreg_target="bogus")
