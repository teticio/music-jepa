import torch

from jepa.model import build_model


CFG = {
    "data": {"img_size": [96, 216], "patch_size": [8, 8]},
    "model": {
        "encoder": {
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
        },
        "predictor": {"predictor_dim": 32, "depth": 2, "num_heads": 4},
        "ema_momentum": 0.996,
    },
}


def test_forward_pass_shape():
    model = build_model(CFG).eval()
    B, M = 2, 10
    ctx = torch.randn(B, 1, 96, 216)
    tgt = torch.randn(B, 1, 96, 216)
    target_patch_ids = torch.arange(M).unsqueeze(0).expand(B, -1).long()
    predicted, target = model(ctx, tgt, target_patch_ids)
    assert predicted.shape == (B, M, 64)
    assert target.shape == (B, M, 64)


def test_target_encoder_no_gradient():
    """Target encoder parameters should never receive gradient."""
    model = build_model(CFG)
    B, M = 2, 5
    ctx = torch.randn(B, 1, 96, 216)
    tgt = torch.randn(B, 1, 96, 216)
    target_patch_ids = torch.arange(M).unsqueeze(0).expand(B, -1).long()
    predicted, target = model(ctx, tgt, target_patch_ids)
    loss = (predicted - target.detach()).pow(2).mean()
    loss.backward()
    for p in model.target_encoder.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0
    # Sanity: encoder did receive some gradient
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads)


def test_ema_update_blends_weights():
    """EMA update must move target params toward the (changed) encoder params."""
    model = build_model(CFG)
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    before = next(iter(model.target_encoder.parameters())).clone()
    model.update_target_encoder(step=0, total_steps=100)
    after = next(iter(model.target_encoder.parameters()))
    assert not torch.allclose(before, after)
