import torch
from model.losses import weighted_channel_mse, occupancy_weighted_mse


def test_zero_loss_when_pred_equals_target():
    pred = torch.randn(2, 3, 10, 10)
    target = pred.clone()
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss = weighted_channel_mse(pred, target, weights)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_channel_weighting_scales_contribution():
    pred = torch.zeros(1, 3, 10, 10)
    target = torch.zeros(1, 3, 10, 10)
    target[:, 0, :, :] = 1.0  # error only in channel 0
    weights_a = torch.tensor([1.0, 0.0, 0.0])
    weights_b = torch.tensor([2.0, 0.0, 0.0])
    loss_a = weighted_channel_mse(pred, target, weights_a)
    loss_b = weighted_channel_mse(pred, target, weights_b)
    assert torch.isclose(loss_b, loss_a * 2.0)


def test_bg_weight_zero_ignores_background_error():
    target = torch.zeros(1, 3, 10, 10)
    target[:, 0, 5, 5] = 1.0  # occupied pixel
    source = target.clone()
    weights = torch.tensor([1.0, 0.0, 0.0])

    pred_bg_err = target.clone()
    pred_bg_err[:, 0, 0, 0] = 5.0  # error only in a background pixel
    loss_bg = occupancy_weighted_mse(pred_bg_err, target, source, weights, bg_weight=0.0, peak_weight=0.0)
    assert torch.isclose(loss_bg, torch.tensor(0.0), atol=1e-6)

    pred_occ_err = target.clone()
    pred_occ_err[:, 0, 5, 5] = 5.0  # error only at the occupied pixel
    loss_occ = occupancy_weighted_mse(pred_occ_err, target, source, weights, bg_weight=0.0, peak_weight=0.0)
    assert loss_occ > 0.0


def test_bg_weight_only_discounts_prob_channel():
    target = torch.zeros(1, 3, 10, 10)
    target[:, 0, 5, 5] = 1.0  # occupied pixel; rest of grid is background
    source = target.clone()

    pred_vel_err = target.clone()
    pred_vel_err[:, 1, 0, 0] = 5.0  # VX error at a background pixel
    weights = torch.tensor([0.0, 1.0, 0.0])
    loss_vel_bg = occupancy_weighted_mse(pred_vel_err, target, source, weights, bg_weight=0.05, peak_weight=0.0)
    loss_vel_uniform = weighted_channel_mse(pred_vel_err, target, weights)
    assert torch.isclose(loss_vel_bg, loss_vel_uniform)


def test_bg_weight_one_matches_weighted_channel_mse():
    pred = torch.randn(2, 3, 10, 10)
    target = torch.randn(2, 3, 10, 10)
    source = torch.randn(2, 3, 10, 10)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss_occ = occupancy_weighted_mse(pred, target, source, weights, bg_weight=1.0, peak_weight=0.0)
    loss_uniform = weighted_channel_mse(pred, target, weights)
    assert torch.isclose(loss_occ, loss_uniform)


def test_background_error_less_diluted_than_grid_wide_mean():
    # Regression test for the root-cause fix: previously, background PROB
    # error was averaged over the *entire* grid (occ+bg combined), diluting
    # a widespread small background error (e.g. a growing periodic artifact)
    # into near-invisibility once background pixels dominate the grid (as
    # they do in practice: ~10% occupancy). The fixed formula normalizes
    # background error by the background pixel count itself, so the same
    # absolute error produces a much larger, size-appropriate contribution.
    n = 50
    occ_frac = 0.1
    num_occ = int(occ_frac * n * n)
    target = torch.zeros(1, 3, n, n)
    flat = target[0, 0].view(-1)
    flat[:num_occ] = 1.0
    source = target.clone()
    pred = target.clone()
    pred[:, 0, :, :] += 0.05  # small widespread error, e.g. a periodic ripple
    pred_flat = pred[0, 0].view(-1)
    pred_flat[:num_occ] = 1.0  # occupied cells predicted exactly

    weights = torch.tensor([1.0, 0.0, 0.0])
    bg_weight = 0.05
    fixed_loss = occupancy_weighted_mse(pred, target, source, weights, bg_weight=bg_weight)

    occ_count = num_occ
    bg_count = n * n - num_occ
    bg_sum_err = bg_count * (0.05 ** 2)
    old_grid_wide_loss = (bg_weight * bg_sum_err) / (n * n)
    new_formula_expected = (bg_weight * bg_sum_err) / (occ_count + bg_weight * bg_count)

    assert torch.isclose(fixed_loss, torch.tensor(new_formula_expected), rtol=1e-4)
    assert fixed_loss > old_grid_wide_loss * 5


def test_vacated_cell_keeps_full_weight_via_source_occupancy():
    source = torch.zeros(1, 3, 10, 10)
    source[:, 0, 5, 5] = 1.0  # occupied at t
    target = torch.zeros(1, 3, 10, 10)  # empty at t+1 (vacated)
    weights = torch.tensor([1.0, 0.0, 0.0])

    pred_correct = target.clone()  # correctly predicts the cell clears
    pred_wrong = target.clone()
    pred_wrong[:, 0, 5, 5] = 5.0  # incorrectly predicts mass stayed

    loss_correct = occupancy_weighted_mse(pred_correct, target, source, weights, bg_weight=0.0, peak_weight=0.0)
    loss_wrong = occupancy_weighted_mse(pred_wrong, target, source, weights, bg_weight=0.0, peak_weight=0.0)
    assert torch.isclose(loss_correct, torch.tensor(0.0), atol=1e-6)
    assert loss_wrong > 0.0


def test_peak_term_prefers_sharp_shifted_over_diffuse_centered():
    # Regression test for the root-cause fix in
    # docs/debugging/findings-peak-decay-dissolution.md: per-pixel
    # occupancy-weighted MSE alone (peak_weight=0) prefers a diffuse,
    # mass-matched blob over a sharp-but-slightly-mispositioned ball once
    # position uncertainty exceeds about one ball radius -- confirmed
    # directly below -- which the peak term must reverse.
    n = 20
    weights = torch.tensor([1.0, 0.0, 0.0])

    def disk(cx, cy, radius):
        ii, jj = torch.meshgrid(torch.arange(n).float(), torch.arange(n).float(), indexing="ij")
        d = torch.hypot(ii - cx, jj - cy)
        w = torch.where(d <= radius, 1.0 - d / radius, torch.zeros_like(d))
        return 1.0 - torch.exp(-w)

    target = torch.zeros(1, 3, n, n)
    target[0, 0] = disk(10, 10, 0.75)
    source = target.clone()

    diffuse = disk(10, 10, 6.0)
    diffuse = diffuse * (target[0, 0].sum() / diffuse.sum())
    pred_diffuse = torch.zeros(1, 3, n, n)
    pred_diffuse[0, 0] = diffuse

    pred_sharp_shifted = torch.zeros(1, 3, n, n)
    pred_sharp_shifted[0, 0] = disk(12, 10, 0.75)

    loss_diffuse_no_peak = occupancy_weighted_mse(pred_diffuse, target, source, weights, peak_weight=0.0)
    loss_sharp_no_peak = occupancy_weighted_mse(pred_sharp_shifted, target, source, weights, peak_weight=0.0)
    assert loss_diffuse_no_peak < loss_sharp_no_peak  # confirms the bug exists without the fix

    loss_diffuse_with_peak = occupancy_weighted_mse(pred_diffuse, target, source, weights, peak_weight=0.1)
    loss_sharp_with_peak = occupancy_weighted_mse(pred_sharp_shifted, target, source, weights, peak_weight=0.1)
    assert loss_sharp_with_peak < loss_diffuse_with_peak  # the fix reverses the preference


def test_mass_term_prefers_commit_over_giveup_in_dense_high_drift_tile():
    # Regression test for docs/debugging/findings-mass-conservation-loss.md:
    # a per-tile summed-mass term should favor committing to a drifted-but-
    # present position over giving up entirely, even in the dense/
    # high-drift regime that defeated every peak-based regional
    # formulation tested there (a shared max lets a confidently-correct
    # neighbor mask a give-up ball's missing mass; a shared sum can't,
    # since the tile total is directly short by the missing mass either
    # way).
    n = 32
    weights = torch.tensor([1.0, 0.0, 0.0])

    def disk(cx, cy, radius=0.75):
        ii, jj = torch.meshgrid(torch.arange(n).float(), torch.arange(n).float(), indexing="ij")
        d = torch.hypot(ii - cx, jj - cy)
        w = torch.where(d <= radius, 1.0 - d / radius, torch.zeros_like(d))
        return 1.0 - torch.exp(-w)

    # 8 "certain" balls crowded into one 16x16 tile (rows/cols 0-15),
    # simulating high local density; correctly predicted in both candidates.
    certain_centers = [(2, 2), (4, 10), (8, 4), (10, 12), (3, 14), (12, 2), (6, 8), (14, 6)]
    # One "uncertain" ball, also in that tile, true position (7, 7).
    uncertain_true = (7.0, 7.0)
    drift = 8.0  # large drift, still inside the same tile

    target = torch.zeros(1, 3, n, n)
    for cx, cy in certain_centers:
        target[0, 0] += disk(cx, cy)
    target[0, 0] += disk(*uncertain_true)
    source = target.clone()

    give_up = torch.zeros(1, 3, n, n)
    for cx, cy in certain_centers:
        give_up[0, 0] += disk(cx, cy)
    # uncertain ball omitted entirely

    commit = give_up.clone()
    commit[0, 0] += disk(uncertain_true[0] + drift, uncertain_true[1])

    loss_giveup_no_mass = occupancy_weighted_mse(
        give_up, target, source, weights, peak_weight=0.0, mass_weight=0.0)
    loss_commit_no_mass = occupancy_weighted_mse(
        commit, target, source, weights, peak_weight=0.0, mass_weight=0.0)
    assert loss_giveup_no_mass < loss_commit_no_mass  # confirms the bug exists without the fix

    loss_giveup_with_mass = occupancy_weighted_mse(
        give_up, target, source, weights, peak_weight=0.0, mass_weight=0.1, mass_tile=16)
    loss_commit_with_mass = occupancy_weighted_mse(
        commit, target, source, weights, peak_weight=0.0, mass_weight=0.1, mass_tile=16)
    assert loss_commit_with_mass < loss_giveup_with_mass  # the fix reverses the preference


def test_mass_term_zero_when_tile_mass_matches():
    n = 32
    weights = torch.tensor([1.0, 0.0, 0.0])
    target = torch.zeros(1, 3, n, n)
    target[:, 0, 5, 5] = 1.0
    source = target.clone()
    # Prediction spreads the same total tile mass to a different pixel
    # within the same tile -- tile-level mass matches exactly, so the
    # mass term alone should contribute nothing even though pred != target.
    pred = torch.zeros(1, 3, n, n)
    pred[:, 0, 6, 6] = 1.0

    loss_mass_only = occupancy_weighted_mse(
        pred, target, source, weights, bg_weight=0.0, peak_weight=0.0, mass_weight=1.0, mass_tile=16)
    loss_no_mass = occupancy_weighted_mse(
        pred, target, source, weights, bg_weight=0.0, peak_weight=0.0, mass_weight=0.0)
    assert torch.isclose(loss_mass_only, loss_no_mass, atol=1e-6)


def test_mass_term_handles_grid_not_divisible_by_tile_size():
    # n=50 with mass_tile=16 doesn't divide evenly (production grid size) --
    # confirm this doesn't error and doesn't spuriously penalize a correct
    # prediction because of the partial boundary tile.
    n = 50
    weights = torch.tensor([1.0, 0.0, 0.0])
    pred = torch.zeros(1, 3, n, n)
    pred[:, 0, 48, 48] = 1.0  # inside the partial boundary tile
    target = pred.clone()
    source = pred.clone()
    loss = occupancy_weighted_mse(
        pred, target, source, weights, bg_weight=0.0, peak_weight=0.0, mass_weight=1.0, mass_tile=16)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
