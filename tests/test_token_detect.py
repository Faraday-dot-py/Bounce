import numpy as np
import torch

import bounce
from model.token_detect import centroid_near, find_token_positions, territory_mask


def _prob_channel(balls, n, radius):
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    return torch.from_numpy(np.array(G, dtype=np.float32)[:, :, bounce.PROB])


def test_centroid_near_recovers_isolated_ball_center():
    balls = [{"x": 10.5, "y": 7.5, "vx": 0.0, "vy": 0.0}]
    prob = _prob_channel(balls, 20, 0.75)
    result = centroid_near(prob, torch.tensor([10.0, 7.0]), radius=0.75)
    assert torch.allclose(result, torch.tensor([10.5, 7.5]), atol=0.05)


def test_centroid_near_returns_input_position_when_window_is_empty():
    prob = torch.zeros((20, 20))
    pos = torch.tensor([5.0, 5.0])
    result = centroid_near(prob, pos, radius=0.75)
    assert torch.equal(result, pos)


def test_find_token_positions_recovers_well_separated_balls():
    balls = [
        {"x": 5.0, "y": 5.0, "vx": 0.0, "vy": 0.0},
        {"x": 15.0, "y": 8.0, "vx": 0.0, "vy": 0.0},
        {"x": 8.0, "y": 16.0, "vx": 0.0, "vy": 0.0},
    ]
    prob = _prob_channel(balls, 20, 0.75)
    detected = find_token_positions(prob, radius=0.75)
    assert detected.shape[0] == 3

    expected = torch.tensor([[b["x"], b["y"]] for b in balls])
    for row in detected:
        dists = torch.norm(expected - row, dim=1)
        assert dists.min() < 0.2


def test_find_token_positions_merges_heavily_overlapping_balls():
    # Accepted edge case (design spec): balls closer together than one
    # ball's footprint are detected as a single token, not two.
    balls = [
        {"x": 10.0, "y": 10.0, "vx": 0.0, "vy": 0.0},
        {"x": 10.3, "y": 10.0, "vx": 0.0, "vy": 0.0},
    ]
    prob = _prob_channel(balls, 20, 0.75)
    detected = find_token_positions(prob, radius=0.75)
    assert detected.shape[0] < 2


def test_find_token_positions_breaks_ties_at_half_integer_center():
    # A ball centered exactly between two grid cells splats equal mass
    # onto both, so raw `prob == pooled` non-max suppression doesn't
    # break the tie and both cells register as peaks for one ball (see
    # docs/debugging/experiment-log.md). Reproduced directly at radius
    # 1.5, x=10.5.
    balls = [{"x": 10.5, "y": 5.0, "vx": 0.0, "vy": 0.0}]
    prob = _prob_channel(balls, 20, 1.5)
    detected = find_token_positions(prob, radius=1.5)
    assert detected.shape[0] == 1


def test_find_token_positions_does_not_let_a_neighbour_hijack_a_weaker_peak():
    # Reproduces a real dropout case directly (seed 4750, see
    # docs/debugging/experiment-log.md): two balls ~2.85 cells apart,
    # each correctly detected as its own raw NMS peak, but one sits
    # closer to its own grid cell (larger own-cell mass) than the
    # other. Without gating, `centroid_near`'s window from the weaker
    # ball's peak reaches into the stronger one's mass and gets pulled
    # there entirely, losing the weaker ball's detection.
    balls = [
        {"x": 14.952715875383788, "y": 9.078687174437743, "vx": 0.0, "vy": 0.0},
        {"x": 13.241050987763971, "y": 6.6910605299027734, "vx": 0.0, "vy": 0.0},
    ]
    prob = _prob_channel(balls, 20, 0.75)
    detected = find_token_positions(prob, radius=0.75)
    assert detected.shape[0] == 2
    dists = torch.cdist(detected, torch.tensor([[b["x"], b["y"]] for b in balls]))
    # every true ball has its own nearby detection (not both detections
    # collapsed onto the same one, leaving the other ball uncovered)
    assert dists.min(dim=0).values.max() < 1.5


def test_find_token_positions_handles_empty_grid():
    prob = torch.zeros((20, 20))
    detected = find_token_positions(prob, radius=0.75)
    assert detected.shape == (0, 2)


def test_centroid_near_returns_position_for_far_off_grid_query():
    # Clamping an off-grid window used to leave i_lo > i_hi, which both
    # sliced the grid with wrapped negative indices and made the arange
    # over the window raise; the `total <= 1e-6` guard never got a chance
    # to fire. Mass in the corner is what made the wrapped slice nonempty.
    prob = torch.zeros((20, 20))
    prob[0:2, 0:2] = 1.0
    for coords in ([-50.0, -50.0], [-21.0, -21.0], [500.0, 500.0], [-21.0, 500.0]):
        pos = torch.tensor(coords)
        assert torch.equal(centroid_near(prob, pos, radius=0.75), pos)


def test_territory_mask_splits_window_at_midline():
    # Two tokens 4 cells apart on the x-axis; a window spanning both must
    # be split down the middle, each token owning its own half.
    ii = torch.arange(0, 10, dtype=torch.float32).view(-1, 1).expand(10, 1)
    jj = torch.zeros(10, 1)
    positions = torch.tensor([[3.0, 0.0], [7.0, 0.0]])
    mask0 = territory_mask(ii, jj, positions, self_idx=0)
    mask1 = territory_mask(ii, jj, positions, self_idx=1)
    # cell x=4 is closer to token 0 (dist 1) than token 1 (dist 3)
    assert bool(mask0[4, 0])
    assert not bool(mask1[4, 0])
    # cell x=6 is closer to token 1 (dist 1) than token 0 (dist 3)
    assert bool(mask1[6, 0])
    assert not bool(mask0[6, 0])


def test_territory_mask_no_op_for_single_token():
    ii = torch.arange(0, 5, dtype=torch.float32).view(-1, 1).expand(5, 1)
    jj = torch.zeros(5, 1)
    positions = torch.tensor([[2.0, 0.0]])
    mask = territory_mask(ii, jj, positions, self_idx=0)
    assert bool(mask.all())


def test_territory_mask_ties_favor_self():
    # Cell exactly equidistant from both tokens (x=5, tokens at x=3 and
    # x=7) must belong to whichever token's mask is being computed.
    ii = torch.tensor([[5.0]])
    jj = torch.tensor([[0.0]])
    positions = torch.tensor([[3.0, 0.0], [7.0, 0.0]])
    mask0 = territory_mask(ii, jj, positions, self_idx=0)
    mask1 = territory_mask(ii, jj, positions, self_idx=1)
    assert bool(mask0[0, 0])
    assert bool(mask1[0, 0])


def test_territory_mask_handles_coincident_tokens():
    # Two tokens at the exact same position -- degenerate but must not
    # produce an all-False mask for either.
    ii = torch.tensor([[5.0]])
    jj = torch.tensor([[5.0]])
    positions = torch.tensor([[5.0, 5.0], [5.0, 5.0]])
    mask0 = territory_mask(ii, jj, positions, self_idx=0)
    mask1 = territory_mask(ii, jj, positions, self_idx=1)
    assert bool(mask0[0, 0])
    assert bool(mask1[0, 0])
