"""Training losses (scope.md 4.4).

The primary loss is binary cross-entropy against a *soft* target -- the AI fraction in
[0, 1] -- not against a hard label. It is minimised at sigma(z) = y, its gradient with
respect to the logit is the clean sigma(z) - y, and it is exactly right for a target the
UI treats as a continuous dial.

"Label smoothing 0.05" needs a stated reading, because the obvious one is wrong. Label
smoothing is defined for one-hot labels; applied to a soft target it is a *target-range
clamp* to [0.025, 0.975]. That caps |z| at about 3.7 and stops logit blow-up on the mass
piled at 0 and 1, which is a real regulariser.

It also has a cost worth being explicit about. This product lives in the far right tail
of the human score distribution, and symmetric smoothing compresses exactly the region
being resolved -- it tells the model never to say a pure-human window is below 0.025.
Hence the asymmetric mode, which clamps only the AI end and leaves human targets at 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias


def smooth_targets(target: torch.Tensor, eps: float, mode: str = "symmetric") -> torch.Tensor:
    if eps <= 0.0:
        return target
    if mode == "asymmetric":
        # Clamp only the AI end; leave pure-human targets at exactly 0 so the tail
        # the operating point lives in stays resolvable.
        return torch.where(target > 0.0, target.clamp(max=1.0 - eps), target)
    return target * (1.0 - eps) + eps / 2.0


def soft_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 0.05,
    mode: str = "symmetric",
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    smoothed = smooth_targets(target, eps, mode)
    loss = F.binary_cross_entropy_with_logits(
        logits.squeeze(-1).float(), smoothed.float(), reduction="none"
    )
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def target_entropy(target: torch.Tensor) -> torch.Tensor:
    """H(target), the floor that soft-target BCE cannot go below."""
    p = target.float().clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def excess_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Loss minus its irreducible floor. Goes to zero for a perfect model.

    Log this alongside the raw loss. With mixed-authorship rows in the corpus the raw
    number plateaus well above zero and never approaches it, which looks exactly like
    a training bug and is not one.
    """
    raw = F.binary_cross_entropy_with_logits(
        logits.squeeze(-1).float(), target.float(), reduction="none"
    )
    return (raw - target_entropy(target)).mean()


def multitask_loss(
    ai_logits: torch.Tensor,
    ai_target: torch.Tensor,
    genre_logits: torch.Tensor | None = None,
    genre_target: torch.Tensor | None = None,
    *,
    eps: float = 0.05,
    mode: str = "symmetric",
    genre_weight: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, ai_loss, genre_loss)."""
    ai_loss = soft_bce(ai_logits, ai_target, eps=eps, mode=mode)
    genre_loss = torch.zeros((), device=ai_logits.device, dtype=ai_loss.dtype)
    if genre_logits is not None and genre_target is not None and genre_weight > 0:
        genre_loss = F.cross_entropy(genre_logits.float(), genre_target.long())
    return ai_loss + genre_weight * genre_loss, ai_loss, genre_loss
