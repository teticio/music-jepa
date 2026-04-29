import torch
import torch.nn.functional as F


def vicreg_loss(z: torch.Tensor, std_coef=25.0, cov_coef=1.0, eps=1e-4) -> torch.Tensor:
    """
    VICReg regularization on representations z: (B, D).

    Two terms prevent mode collapse:
    - Variance: penalises any dimension whose std falls below 1.
    - Covariance: penalises off-diagonal entries of the covariance matrix,
      forcing dimensions to be decorrelated and carry distinct information.

    No contrastive negatives needed - the asymmetry between student encoder
    (gradient) and EMA target encoder (no gradient) already prevents collapse;
    VICReg gives an extra safety net especially early in training.
    """
    B, D = z.shape
    z = z - z.mean(dim=0)  # center per dimension

    # Variance term
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(1.0 - std).mean()

    # Covariance term
    cov = (z.T @ z) / (B - 1)
    off_diag_sq = cov.pow(2).fill_diagonal_(0).sum()
    cov_loss = off_diag_sq / D

    return std_coef * var_loss + cov_coef * cov_loss


def jepa_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    std_coef=25.0,
    cov_coef=1.0,
):
    """
    Combined JEPA training loss.

    predicted: (B, M, D)  predictor output
    target:    (B, M, D)  target encoder output (detached)

    Returns (mse_loss, reg_loss) — caller sums them.
    The split allows separate logging to track training dynamics.
    """
    mse = F.mse_loss(predicted, target.detach())

    # Apply VICReg on mean-pooled target representations (one vector per track)
    target_pooled = target.detach().mean(dim=1)  # (B, D)
    reg = vicreg_loss(target_pooled, std_coef=std_coef, cov_coef=cov_coef)

    return mse, reg
