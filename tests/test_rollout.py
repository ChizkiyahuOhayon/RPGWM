import torch

from rpgwm.models.gaussians import GaussianState
from rpgwm.models.rollout import RolloutW

B, N, C, D = 2, 64, 17, 32


def make_state(seed=0):
    g = torch.Generator().manual_seed(seed)
    return GaussianState.random(B, N, C, D, generator=g)


def make_model():
    torch.manual_seed(0)
    return RolloutW(dim=64, layers=2, heads=4, knn=8, num_classes=C, feat_dim=D)


def test_shapes_and_identity_preserved():
    w, s = make_model(), make_state()
    actions = torch.randn(B, 6, 4) * 0.1
    frames = w.rollout(s, actions)
    assert len(frames) == 6
    for f in frames:
        # slot count and ordering never change: identity is positional
        assert f.mu.shape == (B, N, 3)
        assert f.opacity.shape == (B, N)
        assert f.sem.shape == (B, N, C)


def test_near_identity_at_init():
    """Zero-initialized heads: the rollout starts as 'everything stays put'."""
    w, s = make_model(), make_state()
    out = w(s, torch.zeros(B, 4))
    assert torch.allclose(out.mu, s.mu, atol=1e-5)
    assert torch.allclose(out.log_scale, s.log_scale, atol=1e-5)
    assert torch.allclose(out.opacity, s.opacity, atol=1e-5)


def test_quaternions_stay_normalized():
    w, s = make_model(), make_state()
    for p in w.parameters():  # break the zero init so rotations actually move
        torch.nn.init.normal_(p, std=0.02)
    out = w(s, torch.randn(B, 4))
    norms = out.quat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_gradients_flow_through_full_rollout():
    w, s = make_model(), make_state()
    actions = torch.randn(B, 3, 4, requires_grad=True) * 0.1
    frames = w.rollout(s, actions)
    frames[-1].mu.sum().backward()
    grads = [p.grad for p in w.parameters() if p.grad is not None]
    assert len(grads) > 0 and any(g.abs().sum() > 0 for g in grads)


def test_action_changes_prediction():
    w, s = make_model(), make_state()
    for p in w.parameters():
        torch.nn.init.normal_(p, std=0.05)
    out_a = w(s, torch.tensor([[5.0, 0.0, 0.0, 10.0]] * B))
    out_b = w(s, torch.tensor([[-5.0, 2.0, 1.0, 3.0]] * B))
    assert not torch.allclose(out_a.mu, out_b.mu)


def test_step_bounds_respected():
    w, s = make_model(), make_state()
    for p in w.parameters():
        torch.nn.init.normal_(p, std=10.0)  # adversarial weights
    out = w(s, torch.randn(B, 4) * 100)
    assert (out.mu - s.mu).norm(dim=-1).max() <= RolloutW.MAX_STEP_M * (3 ** 0.5) + 1e-4
    assert (out.opacity >= 0).all() and (out.opacity <= 1).all()
