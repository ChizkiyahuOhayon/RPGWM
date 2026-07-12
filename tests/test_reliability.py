import torch

from rpgwm.models.reliability import (
    QuantileNormalizer, RhoHead, expected_calibration_error,
    partition_and_inflate, propagated_cov_summary, realized_error,
)


def test_quantile_transform_range_and_monotonicity():
    qn = QuantileNormalizer(num_steps=6)
    torch.manual_seed(0)
    train_errors = torch.rand(10000) * 3.0
    qn.fit(0, train_errors)
    x = torch.tensor([0.0, 0.5, 1.5, 2.9, 10.0])
    q = qn.transform(0, x)
    assert (q >= 0).all() and (q <= 1).all()
    assert (q[1:] >= q[:-1]).all()          # monotone
    assert q[0] < 0.05 and q[-1] > 0.99     # extremes map to extremes


def test_realized_error_direction():
    """Worse overlap and larger attribute error => larger e (UNreliability)."""
    good = realized_error(local_iou=torch.tensor([[0.9]]), sem_err=torch.tensor([[0.1]]),
                          center_err=torch.tensor([[0.1]]), has_gt_support=torch.tensor([[True]]))
    bad = realized_error(local_iou=torch.tensor([[0.1]]), sem_err=torch.tensor([[0.8]]),
                         center_err=torch.tensor([[0.9]]), has_gt_support=torch.tensor([[True]]))
    assert bad > good


def test_realized_error_no_gt_support_keeps_semantic_term_only():
    e = realized_error(local_iou=torch.tensor([[0.0]]), sem_err=torch.tensor([[0.3]]),
                       center_err=torch.tensor([[5.0]]), has_gt_support=torch.tensor([[False]]))
    assert torch.allclose(e, torch.tensor([[0.3]]))


def test_rho_head_output_range_and_shapes():
    head = RhoHead(feat_dim=32, num_steps=6, width=64)
    feat = torch.randn(2, 10, 32)
    cov = propagated_cov_summary(3, torch.tensor([0.04, 0.04, 0.01]))
    step = torch.full((2, 10), 3, dtype=torch.long)
    rho = head(feat, cov, step)
    assert rho.shape == (2, 10)
    assert (rho >= 0).all() and (rho <= 1).all()


def test_propagated_cov_grows_with_step():
    var = torch.tensor([0.04, 0.04, 0.01])
    tr = [propagated_cov_summary(k, var)[3] for k in range(6)]  # log trace
    assert all(tr[i] < tr[i + 1] for i in range(5))


def test_ece_perfect_and_broken():
    torch.manual_seed(0)
    target = torch.rand(5000)
    assert expected_calibration_error(target, target) < 1e-6
    assert expected_calibration_error(1.0 - target, target) > 0.3


def test_partition_and_inflate():
    log_scale = torch.zeros(1, 4, 3)
    rho = torch.tensor([[0.0, 0.4, 0.8, 1.0]])
    trusted, new_ls = partition_and_inflate(log_scale, rho, theta=0.5, lam=2.0)
    assert trusted.tolist() == [[True, True, False, False]]
    # trusted slots untouched
    assert torch.allclose(new_ls[0, :2], log_scale[0, :2])
    # untrusted: Sigma scaled by (1 + lam*rho) -> log_scale += 0.5*log(f)
    expected = 0.5 * torch.log(torch.tensor(1 + 2.0 * 0.8))
    assert torch.allclose(new_ls[0, 2], torch.full((3,), expected.item()), atol=1e-6)
    # the least reliable Gaussian is inflated the most
    assert new_ls[0, 3, 0] > new_ls[0, 2, 0]
