import torch

from jepa.playlist_head import PlaylistHead, save_head, load_head


def test_head_forward_shape():
    head = PlaylistHead(embed_dim=128, hidden_dim=256)
    context = torch.randn(4, 128 * 3)
    out = head(context)
    assert out.shape == (4, 128)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_head_residual_source_validates():
    import pytest
    with pytest.raises(ValueError):
        PlaylistHead(embed_dim=8, hidden_dim=16, residual_source="bogus")


def test_head_save_load_roundtrip(tmp_path):
    head = PlaylistHead(embed_dim=16, hidden_dim=32, dropout=0.0, residual_source="first")
    head.eval()
    cfg = {"model": {"hidden_dim": 32, "dropout": 0.0}}
    out_path = str(tmp_path / "head.pt")
    save_head(out_path, head, cfg)
    loaded, _ = load_head(out_path, device="cpu")
    context = torch.randn(2, 16 * 3)
    with torch.no_grad():
        a = head(context)
        b = loaded(context)
    assert torch.allclose(a, b, atol=1e-6)


def test_head_save_load_residual_third(tmp_path):
    head = PlaylistHead(embed_dim=16, hidden_dim=32, dropout=0.0, residual_source="third")
    head.eval()
    cfg = {"model": {"hidden_dim": 32, "dropout": 0.0}}
    out_path = str(tmp_path / "head.pt")
    save_head(out_path, head, cfg)
    loaded, _ = load_head(out_path, device="cpu")
    assert loaded.residual_source == "third"
