import copy
import math
import torch
import torch.nn as nn

from .vit import VisionTransformer, TransformerBlock


class CrossTrackPredictor(nn.Module):
    """
    Given context spectrogram encodings, predicts representations for
    specified patch positions in a target spectrogram.

    Implements the I-JEPA predictor pattern: context tokens + positional query
    tokens are concatenated, self-attended, and the query outputs are predictions.
    """
    def __init__(
        self,
        num_patches=324,
        embed_dim=384,
        predictor_dim=192,
        depth=4,
        num_heads=6,
    ):
        super().__init__()
        self.predictor_dim = predictor_dim

        self.ctx_proj = nn.Linear(embed_dim, predictor_dim)

        # Positional embeddings for each possible target patch position
        self.target_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, embed_dim)

        nn.init.trunc_normal_(self.target_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)

    def forward(self, ctx_tokens, target_patch_ids):
        """
        ctx_tokens:       (B, N, D)  -- context encoder output
        target_patch_ids: (B, M)     -- patch indices to predict in target

        Returns: (B, M, D) predicted representations
        """
        B, M = target_patch_ids.shape

        ctx = self.ctx_proj(ctx_tokens)  # (B, N, P)

        # Gather positional embeddings for queried target positions
        pos = self.target_pos_embed.expand(B, -1, -1)  # (B, num_patches, P)
        idx = target_patch_ids.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        query_tokens = torch.gather(pos, 1, idx)  # (B, M, P)

        # Self-attend over context + query tokens
        x = torch.cat([ctx, query_tokens], dim=1)  # (B, N+M, P)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Extract only query outputs
        preds = x[:, ctx.shape[1]:]      # (B, M, P)
        return self.out_proj(preds)       # (B, M, D)


class MusicJEPA(nn.Module):
    """
    Music JEPA: cross-track predictive architecture.

    Given a context spectrogram (track N) and a target spectrogram (track N+1
    from the same playlist), the encoder processes the context and the predictor
    predicts patch-level representations of the target in latent space.

    The target encoder is an EMA copy of the context encoder - it produces the
    prediction targets without gradient flow, preventing trivial collapse.
    """
    def __init__(self, encoder: VisionTransformer, predictor: CrossTrackPredictor, ema_momentum=0.996):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_target_encoder(self, step: int, total_steps: int):
        """Cosine-ramped EMA update: momentum rises from ema_momentum -> 0.999."""
        m = 1 - (1 - self.ema_momentum) * (math.cos(math.pi * step / total_steps) + 1) / 2
        for p_enc, p_tgt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(m).add_(p_enc.data, alpha=1 - m)

    def forward(self, ctx_spec, tgt_spec, target_patch_ids):
        """
        ctx_spec:          (B, 1, H, W)  context spectrogram (track N)
        tgt_spec:          (B, 1, H, W)  target spectrogram (track N+1)
        target_patch_ids:  (B, M)        patch indices to predict

        Returns:
            predicted:  (B, M, D)  predictor output
            target:     (B, M, D)  target encoder output (detached)
        """
        ctx_tokens = self.encoder(ctx_spec)  # (B, N, D)

        with torch.no_grad():
            tgt_tokens = self.target_encoder(tgt_spec)  # (B, N, D)

        # Gather target patches at queried positions
        B, M = target_patch_ids.shape
        D = tgt_tokens.shape[-1]
        idx = target_patch_ids.unsqueeze(-1).expand(-1, -1, D)
        tgt_patches = torch.gather(tgt_tokens, 1, idx)   # (B, M, D)

        predicted = self.predictor(ctx_tokens, target_patch_ids)  # (B, M, D)

        return predicted, tgt_patches


def build_model(cfg) -> MusicJEPA:
    img_size = tuple(cfg["data"]["img_size"])
    patch_size = tuple(cfg["data"]["patch_size"])
    num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

    encoder = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=1,
        **cfg["model"]["encoder"],
    )
    embed_dim = cfg["model"]["encoder"]["embed_dim"]

    predictor = CrossTrackPredictor(
        num_patches=num_patches,
        embed_dim=embed_dim,
        **cfg["model"]["predictor"],
    )

    return MusicJEPA(
        encoder=encoder,
        predictor=predictor,
        ema_momentum=cfg["model"]["ema_momentum"],
    )
