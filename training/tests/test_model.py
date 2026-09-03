"""Loss semantics and layerwise LR decay. Torch only -- no backbone download."""

from __future__ import annotations

from itertools import pairwise

import pytest

torch = pytest.importorskip("torch")

from slopmarker.model.losses import (  # noqa: E402
    excess_bce,
    multitask_loss,
    smooth_targets,
    soft_bce,
    target_entropy,
)
from slopmarker.model.optim import (  # noqa: E402
    HEAD_DEPTH,
    build_llrd_param_groups,
    parameter_depth,
)


class TestSoftBce:
    def test_minimised_when_sigmoid_matches_target(self) -> None:
        target = torch.tensor([0.0, 0.3, 0.7, 1.0])
        exact = torch.logit(target.clamp(1e-6, 1 - 1e-6)).unsqueeze(-1)
        worse = torch.zeros_like(exact)
        assert soft_bce(exact, target, eps=0.0) < soft_bce(worse, target, eps=0.0)

    def test_gradient_is_sigmoid_minus_target(self) -> None:
        logits = torch.tensor([[0.4], [-1.2]], requires_grad=True)
        target = torch.tensor([0.25, 0.8])
        soft_bce(logits, target, eps=0.0).backward()
        expected = (torch.sigmoid(logits.detach().squeeze(-1)) - target) / target.numel()
        assert torch.allclose(logits.grad.squeeze(-1), expected, atol=1e-6)

    def test_accepts_fractional_targets(self) -> None:
        loss = soft_bce(torch.zeros(3, 1), torch.tensor([0.1, 0.5, 0.9]))
        assert torch.isfinite(loss)


class TestSmoothing:
    def test_symmetric_clamps_both_ends(self) -> None:
        out = smooth_targets(torch.tensor([0.0, 1.0]), 0.05, "symmetric")
        assert out.tolist() == pytest.approx([0.025, 0.975])

    def test_asymmetric_leaves_pure_human_at_zero(self) -> None:
        """The tail the operating point lives in must stay resolvable."""
        out = smooth_targets(torch.tensor([0.0, 0.5, 1.0]), 0.05, "asymmetric")
        assert out[0].item() == 0.0
        assert out[2].item() == pytest.approx(0.95)

    def test_zero_eps_is_a_no_op(self) -> None:
        target = torch.tensor([0.0, 0.5, 1.0])
        assert torch.equal(smooth_targets(target, 0.0), target)

    def test_smoothing_bounds_the_logit(self) -> None:
        """The point of the clamp: it caps |z| instead of letting it run away."""
        target = torch.tensor([1.0])
        big, bigger = torch.tensor([[8.0]]), torch.tensor([[20.0]])
        # Without smoothing, bigger is always better; with it, there is an optimum.
        assert soft_bce(bigger, target, eps=0.0) < soft_bce(big, target, eps=0.0)
        assert soft_bce(bigger, target, eps=0.05) > soft_bce(big, target, eps=0.05)


class TestExcessBce:
    def test_floor_is_target_entropy(self) -> None:
        """Soft-target BCE plateaus above zero; excess BCE is what goes to zero."""
        target = torch.tensor([0.3, 0.5, 0.7])
        perfect = torch.logit(target).unsqueeze(-1)
        assert soft_bce(perfect, target, eps=0.0).item() > 0.5
        assert excess_bce(perfect, target).abs().item() < 1e-5

    def test_entropy_zero_at_hard_labels(self) -> None:
        assert target_entropy(torch.tensor([0.0, 1.0])).max().item() < 1e-4


class TestMultitaskLoss:
    def test_genre_weight_scales_the_auxiliary_term(self) -> None:
        ai_logits = torch.zeros(4, 1)
        ai_target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        genre_logits = torch.randn(4, 9)
        genre_target = torch.tensor([0, 1, 2, 3])
        total_a, ai_a, genre_a = multitask_loss(
            ai_logits, ai_target, genre_logits, genre_target, genre_weight=0.2
        )
        total_b, ai_b, _ = multitask_loss(
            ai_logits, ai_target, genre_logits, genre_target, genre_weight=0.0
        )
        assert ai_a == ai_b
        assert total_a == pytest.approx((ai_a + 0.2 * genre_a).item())
        assert total_b == pytest.approx(ai_b.item())

    def test_works_without_genre_labels(self) -> None:
        total, ai, genre = multitask_loss(torch.zeros(2, 1), torch.tensor([0.0, 1.0]))
        assert genre.item() == 0.0
        assert total == ai


class TestLayerwiseDecay:
    @pytest.mark.parametrize(
        ("name", "depth"),
        [
            ("model.embeddings.tok_embeddings.weight", 0),
            ("model.layers.0.attn.Wqkv.weight", 1),
            ("model.layers.21.mlp.Wi.weight", 22),
            ("model.final_norm.weight", HEAD_DEPTH),
            ("classifier.weight", HEAD_DEPTH),
            ("genre_head.proj.weight", HEAD_DEPTH),
        ],
    )
    def test_depth_assignment(self, name: str, depth: int) -> None:
        assert parameter_depth(name) == depth

    def test_rates_decay_with_depth(self) -> None:
        model = torch.nn.ModuleDict(
            {
                "embeddings": torch.nn.Linear(4, 4),
                "layers": torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(22)]),
                "classifier": torch.nn.Linear(4, 1),
            }
        )
        groups = build_llrd_param_groups(model, 3e-5, decay=0.9)
        by_depth = {g["depth"]: g["lr"] for g in groups}
        assert by_depth[HEAD_DEPTH] == pytest.approx(3e-5)
        assert by_depth[0] == pytest.approx(3e-5 * 0.9**HEAD_DEPTH)
        depths = sorted(by_depth)
        for shallow, deep in pairwise(depths):
            assert by_depth[shallow] < by_depth[deep]

    def test_survives_lambdalr_scheduling(self) -> None:
        """LambdaLR scales each group's base_lr; a scheduler that *sets* lr would
        silently erase layerwise decay with no error anywhere."""
        model = torch.nn.ModuleDict(
            {
                "embeddings": torch.nn.Linear(4, 4),
                "layers": torch.nn.ModuleList([torch.nn.Linear(4, 4)]),
                "classifier": torch.nn.Linear(4, 1),
            }
        )
        groups = build_llrd_param_groups(model, 3e-5, decay=0.9)
        optimizer = torch.optim.AdamW(groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / 10))

        def ratio() -> float:
            rates = [g["lr"] for g in optimizer.param_groups]
            return max(rates) / max(min(rates), 1e-12)

        for _ in range(5):
            optimizer.step()
            scheduler.step()
        assert ratio() == pytest.approx(1 / 0.9**HEAD_DEPTH, rel=1e-6)

    def test_norm_weights_are_excluded_from_decay(self) -> None:
        model = torch.nn.ModuleDict({"final_norm": torch.nn.LayerNorm(4)})
        groups = build_llrd_param_groups(model, 3e-5, weight_decay=0.01)
        assert all(g["weight_decay"] == 0.0 for g in groups)
