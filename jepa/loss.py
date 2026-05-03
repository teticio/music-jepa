import torch
import torch.nn.functional as F


def vicreg_loss(z: torch.Tensor, std_coef=25.0, cov_coef=1.0, eps=1e-4) -> torch.Tensor:
    """
    VICReg regularization on representations z: (B, D).

    Two terms prevent mode collapse:
    - Variance: penalises any dimension whose std falls below 1.
    - Covariance: penalises off-diagonal entries of the covariance matrix,
      forcing dimensions to be decorrelated and carry distinct information.
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
    std_coef: float = 25.0,
    cov_coef: float = 1.0,
    vicreg_target: str = "predicted",
):
    """
    Combined JEPA training loss.

    predicted: (B, M, D)  predictor output (gradient flows)
    target:    (B, M, D)  target encoder output (always detached here)

    Returns (mse_loss, reg_loss) — caller sums them.
    The split allows separate logging to track training dynamics.

    `vicreg_target` controls where VICReg is applied:
    - "predicted": gradient-flowing regulariser on the predictor output. This
      makes VICReg a real anti-collapse term, complementing EMA asymmetry.
    - "target":    diagnostic only. Operates on the detached target encoder
      output, contributing zero gradient — useful as a monitor of target-side
      representation health (rising values = collapse warning).
    """
    target = target.detach()
    mse = F.mse_loss(predicted, target)

    if vicreg_target == "predicted":
        z = predicted.mean(dim=1)  # (B, D); gradient flows
    elif vicreg_target == "target":
        z = target.mean(dim=1)     # (B, D); detached, no gradient
    else:
        raise ValueError(
            f"vicreg_target must be 'predicted' or 'target', got {vicreg_target!r}"
        )

    reg = vicreg_loss(z, std_coef=std_coef, cov_coef=cov_coef)
    return mse, reg
