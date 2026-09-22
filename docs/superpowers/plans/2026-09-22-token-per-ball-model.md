# Token-Per-Ball Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a next-frame prediction model that represents each ball as a persistent, tracked token (position/velocity/hidden state) with proximity-graph attention for collision dynamics, as a replacement candidate for the grid-based flow-warp architecture (`stage2_flownet_h12_v6.pt`).

**Architecture:** Detect tokens from the first two frames of a sequence; each step, a radius-graph attention network predicts a per-token position/velocity delta from current state and same-neighborhood tokens; a predictive occlusion gate decides whether to blend that prediction with a fresh grid observation or run open-loop through a contact event; the final per-token state is rasterized back into a grid using `bounce.py`'s exact splat formula so loss/eval stay comparable to existing baselines.

**Tech Stack:** PyTorch (no new dependencies — no scipy, matches existing `requirements.txt`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-token-per-ball-model-design.md`

## Global Constraints

- No dependence on ball count N anywhere in the token pipeline (tokens are a variable-length set) — spec's Constraints section.
- No dependence on grid size `n` anywhere except token-init detection and the final rasterization step — spec's I/O contract.
- Neighbor-graph edges must be built from absolute continuous positions and must be able to cross tile boundaries — spec's Proximity-graph attention section, "Tiling requirement."
- Occlusion gate uses the fixed, known ball radius (hard threshold `2 * radius`), not a learned gate — spec's Occlusion gate section.
- Output grid must be produced by rasterizing tokens through `bounce.py`'s existing splat/saturation formula, not a learned decoder — spec's Rasterization head section.
- Default sim params for all tests/training: `radius=0.75`, `dt=0.15`, `gravity=9.0`, `stiffness=400.0`, `substeps=8`, `vy=2.3` (matches `bounce.py` defaults).
- Seed 4738 for any test or run needing a fixed seed (per project convention).

---

## Task 1: Differentiable token rasterizer

**Files:**
- Create: `model/token_rasterize.py`
- Test: `tests/test_token_rasterize.py`

**Interfaces:**
- Produces: `rasterize_tokens(positions: Tensor[N,2], velocities: Tensor[N,2], n: int, radius: float) -> Tensor[3,n,n]`, channel order `(PROB, VX, VY)` matching `bounce.PROB/VX/VY`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_token_rasterize.py
import numpy as np
import torch

import bounce
from model.token_rasterize import rasterize_tokens


def _numpy_ground_truth(balls, n, radius):
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    return np.array(G, dtype=np.float32).transpose(2, 0, 1)


def test_rasterize_tokens_matches_numpy_ground_truth_non_overlapping():
    balls = [
        {"x": 5.0, "y": 5.0, "vx": 1.0, "vy": -2.0},
        {"x": 15.0, "y": 12.0, "vx": -0.5, "vy": 0.3},
    ]
    n, radius = 20, 0.75
    expected = _numpy_ground_truth(balls, n, radius)

    positions = torch.tensor([[b["x"], b["y"]] for b in balls])
    velocities = torch.tensor([[b["vx"], b["vy"]] for b in balls])
    actual = rasterize_tokens(positions, velocities, n, radius).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_rasterize_tokens_matches_numpy_ground_truth_overlapping():
    balls = [
        {"x": 10.0, "y": 10.0, "vx": 2.0, "vy": 0.0},
        {"x": 10.6, "y": 10.0, "vx": -2.0, "vy": 0.0},
    ]
    n, radius = 20, 0.75
    expected = _numpy_ground_truth(balls, n, radius)

    positions = torch.tensor([[b["x"], b["y"]] for b in balls])
    velocities = torch.tensor([[b["vx"], b["vy"]] for b in balls])
    actual = rasterize_tokens(positions, velocities, n, radius).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_rasterize_tokens_handles_zero_tokens():
    n, radius = 10, 0.75
    positions = torch.zeros((0, 2))
    velocities = torch.zeros((0, 2))
    out = rasterize_tokens(positions, velocities, n, radius)
    assert out.shape == (3, n, n)
    assert torch.all(out == 0.0)


def test_rasterize_tokens_is_differentiable():
    positions = torch.tensor([[5.0, 5.0]], requires_grad=True)
    velocities = torch.tensor([[1.0, 0.0]], requires_grad=True)
    out = rasterize_tokens(positions, velocities, 10, 0.75)
    out.sum().backward()
    assert positions.grad is not None
    assert torch.any(positions.grad != 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_token_rasterize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.token_rasterize'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/token_rasterize.py
import torch


def rasterize_tokens(positions, velocities, n, radius):
    """Differentiable equivalent of bounce.py's splat_all for a single
    frame's worth of tokens (no batch dimension -- token count varies per
    sample, so the token pipeline processes one sample at a time).

    positions: (N, 2) tensor, columns (x, y) in grid-cell coordinates.
    velocities: (N, 2) tensor, columns (vx, vy).
    Returns a (3, n, n) tensor in bounce.py's channel order
    (PROB, VX, VY). Matches bounce.py's splat_ball/splat_all exactly:
    each token contributes a linear-falloff disk weight
    `w = max(0, 1 - d/radius)`; the *raw* weight sum is used as the
    VX/VY averaging denominator before PROB is saturated via
    `1 - exp(-sum_w)` -- computing PROB from the saturated value instead
    would double-correct the averaging and not match ground truth.
    """
    device = positions.device
    dtype = positions.dtype
    ii, jj = torch.meshgrid(
        torch.arange(n, device=device, dtype=dtype),
        torch.arange(n, device=device, dtype=dtype),
        indexing="ij",
    )
    x = positions[:, 0].view(-1, 1, 1)
    y = positions[:, 1].view(-1, 1, 1)
    vx = velocities[:, 0].view(-1, 1, 1)
    vy = velocities[:, 1].view(-1, 1, 1)

    d = torch.sqrt((ii.unsqueeze(0) - x) ** 2 + (jj.unsqueeze(0) - y) ** 2 + 1e-12)
    w = torch.clamp(1.0 - d / radius, min=0.0)

    sum_w = w.sum(dim=0)
    vx_sum = (w * vx).sum(dim=0)
    vy_sum = (w * vy).sum(dim=0)
    denom = sum_w.clamp(min=1e-6)

    prob = 1.0 - torch.exp(-sum_w)
    return torch.stack([prob, vx_sum / denom, vy_sum / denom], dim=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_token_rasterize.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_rasterize.py tests/test_token_rasterize.py
git commit -m "feat: add differentiable token rasterizer"
```

---

## Task 2: Token detection (init + per-step centroid refinement)

**Files:**
- Create: `model/token_detect.py`
- Test: `tests/test_token_detect.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `centroid_near(prob: Tensor[n,n], position: Tensor[2], radius: float, margin: float = 1.0) -> Tensor[2]`; `find_token_positions(prob: Tensor[n,n], radius: float, threshold: float = 0.1) -> Tensor[N,2]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_token_detect.py
import numpy as np
import torch

import bounce
from model.token_detect import centroid_near, find_token_positions


def _prob_channel(balls, n, radius):
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    return torch.from_numpy(np.array(G, dtype=np.float32)[:, :, bounce.PROB])


def test_centroid_near_recovers_isolated_ball_center():
    balls = [{"x": 10.3, "y": 6.7, "vx": 0.0, "vy": 0.0}]
    prob = _prob_channel(balls, 20, 0.75)
    result = centroid_near(prob, torch.tensor([10.0, 7.0]), radius=0.75)
    assert torch.allclose(result, torch.tensor([10.3, 6.7]), atol=0.05)


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


def test_find_token_positions_handles_empty_grid():
    prob = torch.zeros((20, 20))
    detected = find_token_positions(prob, radius=0.75)
    assert detected.shape == (0, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_token_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.token_detect'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/token_detect.py
import math

import torch
import torch.nn.functional as F


def centroid_near(prob, position, radius, margin=1.0):
    """Intensity-weighted centroid of `prob` (n, n) within a square window
    around `position` (x, y) -- refines a coarse detection (or a token's
    predicted position) into a sub-pixel observation. Differentiable in
    `prob`'s values (not in the window's location, which is derived from
    a detached rounded position). Returns `position` unchanged if the
    window has no mass, e.g. the predicted position and the true ball
    have drifted further apart than the window covers."""
    n = prob.shape[0]
    half = int(math.ceil(radius + margin))
    cx = int(round(float(position[0])))
    cy = int(round(float(position[1])))
    i_lo, i_hi = max(0, cx - half), min(n - 1, cx + half)
    j_lo, j_hi = max(0, cy - half), min(n - 1, cy + half)
    window = prob[i_lo:i_hi + 1, j_lo:j_hi + 1]
    total = window.sum()
    if total <= 1e-6:
        return position

    ii = torch.arange(i_lo, i_hi + 1, device=prob.device, dtype=prob.dtype).view(-1, 1)
    jj = torch.arange(j_lo, j_hi + 1, device=prob.device, dtype=prob.dtype).view(1, -1)
    x = (window * ii).sum() / total
    y = (window * jj).sum() / total
    return torch.stack([x, y])


def find_token_positions(prob, radius, threshold=0.1):
    """Detect one token per isolated ball via non-max suppression over a
    window sized to one ball's footprint, refined by `centroid_near`.
    Used only for sequence-start initialization -- per-step tracking uses
    `centroid_near` directly against each token's predicted position
    instead of re-running full-grid detection. Two balls closer together
    than the window are detected as a single peak, an accepted rare
    failure mode (see design spec's Token initialization section)."""
    n = prob.shape[0]
    k = max(1, int(round(radius)) * 2 + 1)
    padded = prob.unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(padded, kernel_size=k, stride=1, padding=k // 2)
    pooled = pooled[0, 0, :n, :n]
    is_peak = (prob == pooled) & (prob > threshold)
    coords = torch.nonzero(is_peak, as_tuple=False).to(prob.dtype)
    if coords.shape[0] == 0:
        return torch.zeros((0, 2), dtype=prob.dtype, device=prob.device)
    return torch.stack([centroid_near(prob, c, radius) for c in coords])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_token_detect.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_detect.py tests/test_token_detect.py
git commit -m "feat: add token detection (init peak-finding + centroid refinement)"
```

---

## Task 3: Ground-truth token dataset

**Files:**
- Create: `model/token_dataset.py`
- Test: `tests/test_token_dataset.py`

**Interfaces:**
- Consumes: `model.dataset.make_scenario_uniform/clustered/settled` (unchanged, existing functions), `bounce.make_grid/splat_all/step`.
- Produces: `BounceTokenSequenceDataset(num_samples, n, ball_range, seed, horizon=3, dt=0.15, gravity=9.0, radius=0.75, stiffness=400.0, substeps=8, vy=2.3, cluster_radius=3.0, settle_steps=200)`. `__getitem__(idx) -> (grid_seq: Tensor[horizon+1,3,n,n], state_seq: list[dict])`, each `state_seq[t]` a dict with keys `"x","y","vx","vy"`, each a `Tensor[N]` (N constant across a sample's frames, varies per sample).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_token_dataset.py
import torch

from model.token_dataset import BounceTokenSequenceDataset


def test_token_dataset_state_seq_matches_grid_seq_length_and_ball_count():
    dataset = BounceTokenSequenceDataset(
        num_samples=6, n=20, ball_range=(3, 6), seed=4738, horizon=4
    )
    assert len(dataset) == 6
    for i in range(len(dataset)):
        grid_seq, state_seq = dataset[i]
        assert grid_seq.shape[0] == 5  # horizon + 1
        assert grid_seq.shape[1:] == (3, 20, 20)
        assert len(state_seq) == 5
        num_balls = state_seq[0]["x"].shape[0]
        assert 3 <= num_balls <= 6
        for frame_state in state_seq:
            for key in ("x", "y", "vx", "vy"):
                assert frame_state[key].shape == (num_balls,)


def test_token_dataset_state_seq_is_consistent_with_grid_seq():
    # The ball detected in state_seq[0] should land inside a nonzero
    # PROB region of grid_seq[0] at the same coordinates.
    dataset = BounceTokenSequenceDataset(
        num_samples=1, n=20, ball_range=(2, 2), seed=4738, horizon=2
    )
    grid_seq, state_seq = dataset[0]
    prob = grid_seq[0, 0]
    for x, y in zip(state_seq[0]["x"], state_seq[0]["y"]):
        assert prob[int(round(float(x))), int(round(float(y)))] > 0.0


def test_token_dataset_cycles_through_scenarios():
    dataset = BounceTokenSequenceDataset(
        num_samples=3, n=20, ball_range=(2, 4), seed=4738, horizon=2
    )
    assert len(dataset.samples) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_token_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.token_dataset'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/token_dataset.py
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import bounce
from model.dataset import make_scenario_uniform, make_scenario_clustered, make_scenario_settled


def _copy_states(balls):
    return [dict(b) for b in balls]


def generate_token_sequence(balls, n, dt, gravity, radius, stiffness, substeps, horizon):
    """Same stepping as model.dataset.generate_sequence, but also records
    each frame's exact ball states. model.dataset discards these after
    rendering; the token model needs them for initialization and for
    collision-specific eval metrics that the grid-only dataset can't
    supply (see design spec's Validation plan)."""
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    frames = [np.array(G, dtype=np.float32)]
    states = [_copy_states(balls)]
    for _ in range(horizon):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
        frames.append(np.array(G, dtype=np.float32))
        states.append(_copy_states(balls))
    return frames, states


class BounceTokenSequenceDataset(Dataset):
    SCENARIOS = ("uniform", "clustered", "settled")

    def __init__(self, num_samples, n, ball_range, seed, horizon=3, dt=0.15, gravity=9.0,
                 radius=0.75, stiffness=400.0, substeps=8, vy=2.3,
                 cluster_radius=3.0, settle_steps=200):
        self.samples = []
        rng = random.Random(seed)
        for i in range(num_samples):
            scenario = self.SCENARIOS[i % len(self.SCENARIOS)]
            num_balls = rng.randint(*ball_range)
            if scenario == "uniform":
                balls = make_scenario_uniform(num_balls, n, vy, rng)
            elif scenario == "clustered":
                balls = make_scenario_clustered(num_balls, n, vy, rng, cluster_radius)
            else:
                balls = make_scenario_settled(
                    num_balls, n, vy, rng, radius, gravity, stiffness, dt,
                    substeps, settle_steps,
                )
            frames, states = generate_token_sequence(
                balls, n, dt, gravity, radius, stiffness, substeps, horizon
            )
            self.samples.append((np.stack(frames), states))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, states = self.samples[idx]
        grid_seq = torch.from_numpy(frames.transpose(0, 3, 1, 2)).clone()
        state_seq = [
            {
                "x": torch.tensor([b["x"] for b in frame_states], dtype=torch.float32),
                "y": torch.tensor([b["y"] for b in frame_states], dtype=torch.float32),
                "vx": torch.tensor([b["vx"] for b in frame_states], dtype=torch.float32),
                "vy": torch.tensor([b["vy"] for b in frame_states], dtype=torch.float32),
            }
            for frame_states in states
        ]
        return grid_seq, state_seq
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_token_dataset.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_dataset.py tests/test_token_dataset.py
git commit -m "feat: add ground-truth token dataset"
```

---

## Task 4: Occlusion gate

**Files:**
- Create: `model/token_gate.py`
- Test: `tests/test_token_gate.py`

**Interfaces:**
- Produces: `occluding_mask(positions: Tensor[N,2], radius: float) -> Tensor[N] (bool)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_token_gate.py
import torch

from model.token_gate import occluding_mask


def test_occluding_mask_flags_close_pair_not_far_token():
    radius = 0.75
    positions = torch.tensor([
        [10.0, 10.0],
        [10.6, 10.0],   # 0.6 < 2*radius=1.5 from token 0 -> occluding
        [18.0, 18.0],   # far from everything -> not occluding
    ])
    mask = occluding_mask(positions, radius)
    assert mask.tolist() == [True, True, False]


def test_occluding_mask_single_token_is_never_occluding():
    positions = torch.tensor([[5.0, 5.0]])
    mask = occluding_mask(positions, radius=0.75)
    assert mask.tolist() == [False]


def test_occluding_mask_zero_tokens():
    positions = torch.zeros((0, 2))
    mask = occluding_mask(positions, radius=0.75)
    assert mask.shape == (0,)


def test_occluding_mask_boundary_uses_strict_inequality_at_exactly_2r():
    radius = 0.75
    positions = torch.tensor([[10.0, 10.0], [11.5, 10.0]])  # exactly 2*radius apart
    mask = occluding_mask(positions, radius)
    assert mask.tolist() == [False, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_token_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.token_gate'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/token_gate.py
import torch


def occluding_mask(positions, radius):
    """Per-token boolean mask: True if this token's position is within
    2*radius of any other token -- the same contact-overlap condition as
    bounce.py's ball_pair_forces (`2 * radius - dist`). Intended to be
    called on *predicted* (not yet observed) positions, per the design
    spec's Occlusion gate section, so the gate anticipates an upcoming
    overlap before it's observed rather than reacting to already-blended
    pixels."""
    n = positions.shape[0]
    if n < 2:
        return torch.zeros(n, dtype=torch.bool, device=positions.device)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)
    dist = dist + torch.eye(n, device=positions.device, dtype=positions.dtype) * 1e6
    min_dist, _ = dist.min(dim=1)
    return min_dist < (2.0 * radius)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_token_gate.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_gate.py tests/test_token_gate.py
git commit -m "feat: add predictive occlusion gate"
```

---

## Task 5: Radius-graph edge builder

**Files:**
- Create: `model/token_graph.py`
- Test: `tests/test_token_graph.py`

**Interfaces:**
- Produces: `build_radius_graph(positions: Tensor[N,2], neighbor_radius: float) -> Tensor[2,E] (long)`, `edge_index[0]`=source indices, `edge_index[1]`=destination indices, `i != j`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_token_graph.py
import torch

from model.token_graph import build_radius_graph


def test_build_radius_graph_connects_only_within_threshold():
    positions = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],   # within 2.0 of token 0
        [10.0, 0.0],  # far from everything
    ])
    edges = build_radius_graph(positions, neighbor_radius=2.0)
    pairs = set(zip(edges[0].tolist(), edges[1].tolist()))
    assert pairs == {(0, 1), (1, 0)}


def test_build_radius_graph_no_self_loops():
    positions = torch.tensor([[0.0, 0.0]])
    edges = build_radius_graph(positions, neighbor_radius=5.0)
    assert edges.shape == (2, 0)


def test_build_radius_graph_crosses_tile_boundary_when_positions_are_absolute():
    # "Tile A" token near x=9.5, "tile B" token near x=10.5: adjacent
    # tiles' tokens must connect when passed together in absolute
    # coordinates (design spec's tiling requirement).
    tile_a = torch.tensor([[9.5, 5.0]])
    tile_b = torch.tensor([[10.5, 5.0]])
    combined = torch.cat([tile_a, tile_b], dim=0)

    edges_combined = build_radius_graph(combined, neighbor_radius=2.0)
    assert edges_combined.shape[1] == 2  # (0,1) and (1,0)

    edges_tile_a_alone = build_radius_graph(tile_a, neighbor_radius=2.0)
    assert edges_tile_a_alone.shape[1] == 0  # the cross-boundary neighbor is invisible in isolation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_token_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.token_graph'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/token_graph.py
import torch


def build_radius_graph(positions, neighbor_radius):
    """Undirected edges (as directed pairs, both directions) between
    tokens within `neighbor_radius`, in absolute position space. Has no
    notion of tiles -- passing positions from two adjacent tiles together
    produces edges across the tile boundary exactly as if the tokens had
    come from one untiled scene. Building the graph from one tile's
    tokens in isolation loses any cross-boundary edge, so callers must
    always pass the full combined position set (design spec's tiling
    requirement)."""
    n = positions.shape[0]
    if n < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=positions.device)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)
    within = dist <= neighbor_radius
    within.fill_diagonal_(False)
    src, dst = torch.nonzero(within, as_tuple=True)
    return torch.stack([src, dst], dim=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_token_graph.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_graph.py tests/test_token_graph.py
git commit -m "feat: add radius-graph edge builder"
```

---

## Task 6: Token dynamics (recurrent state + proximity-graph attention)

**Files:**
- Create: `model/token_net.py`
- Test: `tests/test_token_net.py`

**Interfaces:**
- Consumes: `model.token_graph.build_radius_graph(positions, neighbor_radius) -> Tensor[2,E]`.
- Produces: `TokenDynamics(hidden_dim=32, neighbor_radius=3.0)`, `.forward(positions: Tensor[N,2], velocities: Tensor[N,2], hidden: Tensor[N,hidden_dim]) -> (delta_pos: Tensor[N,2], delta_vel: Tensor[N,2], new_hidden: Tensor[N,hidden_dim])`. `.hidden_dim` attribute.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_token_net.py
import torch

from model.token_net import TokenDynamics


def test_forward_shapes():
    torch.manual_seed(4738)
    model = TokenDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.rand(5, 2) * 20
    velocities = torch.randn(5, 2)
    hidden = torch.zeros(5, 8)
    delta_pos, delta_vel, new_hidden = model(positions, velocities, hidden)
    assert delta_pos.shape == (5, 2)
    assert delta_vel.shape == (5, 2)
    assert new_hidden.shape == (5, 8)


def test_delta_head_is_zero_at_init():
    torch.manual_seed(4738)
    model = TokenDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.rand(4, 2) * 20
    velocities = torch.randn(4, 2)
    hidden = torch.randn(4, 8)
    delta_pos, delta_vel, _ = model(positions, velocities, hidden)
    assert torch.allclose(delta_pos, torch.zeros_like(delta_pos))
    assert torch.allclose(delta_vel, torch.zeros_like(delta_vel))


def test_gradients_flow_to_attention_parameters():
    torch.manual_seed(4738)
    model = TokenDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.tensor([[5.0, 5.0], [6.0, 5.0]], requires_grad=True)
    velocities = torch.zeros(2, 2)
    hidden = torch.zeros(2, 8)
    delta_pos, delta_vel, new_hidden = model(positions, velocities, hidden)
    loss = new_hidden.sum()
    loss.backward()
    assert model.query.weight.grad is not None
    assert torch.any(model.query.weight.grad != 0.0)


def test_isolated_token_unaffected_by_distant_tokens():
    # Locality property required for train-small/tile-large transfer:
    # a token's output must not depend on tokens outside neighbor_radius.
    torch.manual_seed(4738)
    model = TokenDynamics(hidden_dim=8, neighbor_radius=3.0)
    hidden = torch.randn(3, 8)
    positions_a = torch.tensor([[5.0, 5.0], [5.5, 5.0], [50.0, 50.0]])
    positions_b = torch.tensor([[5.0, 5.0], [5.5, 5.0], [90.0, 90.0]])
    velocities = torch.zeros(3, 2)

    _, _, hidden_a = model(positions_a, velocities, hidden)
    _, _, hidden_b = model(positions_b, velocities, hidden)

    assert torch.allclose(hidden_a[0], hidden_b[0], atol=1e-6)
    assert torch.allclose(hidden_a[1], hidden_b[1], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_token_net.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.token_net'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/token_net.py
import torch
import torch.nn as nn

from model.token_graph import build_radius_graph


class TokenDynamics(nn.Module):
    """Predicts each token's next (position, velocity) delta and updates
    its hidden state, using a radius-graph attention layer so a token's
    update depends only on neighbors within `neighbor_radius` -- see
    design spec's Proximity-graph attention section. The delta head is
    zero-initialized so the model starts as an exact "coast at current
    velocity" identity (see TokenModel.step, which adds `velocities * dt`
    as the base prediction) -- the same zero-init-residual convention as
    model/net.py, for the same reason: a decent physics-agnostic identity
    is a strong starting point before any learned correction.
    """

    def __init__(self, hidden_dim=32, neighbor_radius=3.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.neighbor_radius = neighbor_radius
        input_dim = 4 + hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, positions, velocities, hidden):
        n = positions.shape[0]
        state = torch.cat([positions, velocities, hidden], dim=-1)
        q = self.query(state)
        k = self.key(state)
        v = self.value(state)

        edge_index = build_radius_graph(positions, self.neighbor_radius)
        attn_out = torch.zeros(n, self.hidden_dim, device=positions.device, dtype=positions.dtype)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            scores = (q[dst] * k[src]).sum(dim=-1) / (self.hidden_dim ** 0.5)
            weights = torch.exp(scores - scores.max())
            denom = torch.zeros(n, device=positions.device, dtype=positions.dtype)
            denom = denom.index_add(0, dst, weights)
            weights = weights / denom[dst].clamp(min=1e-6)
            attn_out = attn_out.index_add(0, dst, weights.unsqueeze(-1) * v[src])

        new_hidden = self.gru(attn_out, hidden)
        delta = self.delta_head(new_hidden)
        return delta[:, :2], delta[:, 2:], new_hidden
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_token_net.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_net.py tests/test_token_net.py
git commit -m "feat: add token dynamics (radius-graph attention + recurrent state)"
```

---

## Task 7: TokenModel assembly (init, step, occlusion-gated observation blend)

**Files:**
- Create: `model/token_model.py`
- Test: `tests/test_token_model.py`

**Interfaces:**
- Consumes: `TokenDynamics` (Task 6), `find_token_positions`/`centroid_near` (Task 2), `occluding_mask` (Task 4), `rasterize_tokens` (Task 1).
- Produces: `TokenModel(n, radius, dt, hidden_dim=32, neighbor_radius=3.0, detect_threshold=0.1, observation_weight=0.5)`.
  - `.init_tokens(first_frame: Tensor[3,n,n], second_frame: Tensor[3,n,n]) -> (positions: Tensor[N,2], velocities: Tensor[N,2], hidden: Tensor[N,hidden_dim])`
  - `.step(positions, velocities, hidden, observed_frame: Tensor[3,n,n]) -> (positions: Tensor[N,2], velocities: Tensor[N,2], hidden: Tensor[N,hidden_dim], pred_grid: Tensor[3,n,n])`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_token_model.py
import torch

from model.token_model import TokenModel
from model.token_rasterize import rasterize_tokens


def test_init_tokens_recovers_position_and_velocity():
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt)
    pos0 = torch.tensor([[5.0, 5.0]])
    vel = torch.tensor([[2.0, -1.0]])
    pos1 = pos0 + vel * dt
    frame0 = rasterize_tokens(pos0, vel, n, radius)
    frame1 = rasterize_tokens(pos1, vel, n, radius)

    positions, velocities, hidden = model.init_tokens(frame0, frame1)
    assert positions.shape == (1, 2)
    assert torch.allclose(positions[0], pos1[0], atol=0.1)
    assert torch.allclose(velocities[0], vel[0], atol=0.3)
    assert hidden.shape == (1, model.dynamics.hidden_dim)


def test_step_at_init_coasts_at_constant_velocity_when_observation_weight_is_zero():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=0.0)
    positions = torch.tensor([[5.0, 5.0]])
    velocities = torch.tensor([[2.0, -1.0]])
    hidden = torch.zeros(1, model.dynamics.hidden_dim)
    observed_frame = torch.zeros(3, n, n)  # unused when observation_weight=0.0

    new_pos, new_vel, new_hidden, pred_grid = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(new_pos, positions + velocities * dt, atol=1e-5)
    assert torch.allclose(new_vel, velocities, atol=1e-5)
    assert pred_grid.shape == (3, n, n)


def test_step_output_grid_matches_direct_rasterization():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=0.0)
    positions = torch.tensor([[5.0, 5.0]])
    velocities = torch.tensor([[2.0, -1.0]])
    hidden = torch.zeros(1, model.dynamics.hidden_dim)
    observed_frame = torch.zeros(3, n, n)

    new_pos, new_vel, _, pred_grid = model.step(positions, velocities, hidden, observed_frame)
    expected_grid = rasterize_tokens(new_pos, new_vel, n, radius)
    assert torch.allclose(pred_grid, expected_grid)


def test_occluding_tokens_ignore_observation_even_at_full_observation_weight():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0)
    # two tokens within 2*radius -> both flagged occluding after their
    # (zero-init, coast-only) predicted step
    positions = torch.tensor([[10.0, 10.0], [10.6, 10.0]])
    velocities = torch.zeros(2, 2)
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    observed_frame = torch.rand(3, n, n)  # arbitrary/irrelevant if gate works

    new_pos, new_vel, _, _ = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(new_pos, positions, atol=1e-5)  # coast (velocity=0), obs ignored
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_token_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.token_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/token_model.py
import torch

from model.token_net import TokenDynamics
from model.token_detect import find_token_positions, centroid_near
from model.token_gate import occluding_mask
from model.token_rasterize import rasterize_tokens


class TokenModel(torch.nn.Module):
    """Wires together token initialization, per-step dynamics, the
    occlusion gate, and rasterization -- see
    docs/superpowers/specs/2026-09-22-token-per-ball-model-design.md."""

    def __init__(self, n, radius, dt, hidden_dim=32, neighbor_radius=3.0,
                 detect_threshold=0.1, observation_weight=0.5):
        super().__init__()
        self.n = n
        self.radius = radius
        self.dt = dt
        self.detect_threshold = detect_threshold
        self.observation_weight = observation_weight
        self.dynamics = TokenDynamics(hidden_dim=hidden_dim, neighbor_radius=neighbor_radius)

    def init_tokens(self, first_frame, second_frame):
        """Detects tokens from `second_frame`; estimates velocity by
        finite difference against `first_frame`'s nearest detection to
        each token (detection order isn't stable across frames)."""
        pos0 = find_token_positions(first_frame[0], self.radius, self.detect_threshold)
        pos1 = find_token_positions(second_frame[0], self.radius, self.detect_threshold)
        dtype = first_frame.dtype
        if pos1.shape[0] == 0:
            return pos1, torch.zeros((0, 2), dtype=dtype), torch.zeros((0, self.dynamics.hidden_dim), dtype=dtype)
        if pos0.shape[0] == 0:
            velocities = torch.zeros_like(pos1)
        else:
            dists = torch.cdist(pos1, pos0)
            nearest = dists.argmin(dim=1)
            velocities = (pos1 - pos0[nearest]) / self.dt
        hidden = torch.zeros((pos1.shape[0], self.dynamics.hidden_dim), dtype=dtype)
        return pos1, velocities, hidden

    def step(self, positions, velocities, hidden, observed_frame):
        """Advances one dt. `observed_frame` is whatever grid the
        observation branch should read from -- ground truth during
        teacher-forced training, the model's own previous rasterized
        output during self-feed rollout; the caller decides which."""
        delta_pos, delta_vel, new_hidden = self.dynamics(positions, velocities, hidden)
        predicted_pos = positions + velocities * self.dt + delta_pos
        predicted_vel = velocities + delta_vel

        occluding = occluding_mask(predicted_pos, self.radius)
        final_pos = predicted_pos.clone()
        final_vel = predicted_vel.clone()
        w = self.observation_weight
        for i in range(predicted_pos.shape[0]):
            if occluding[i] or w == 0.0:
                continue
            obs_pos = centroid_near(observed_frame[0], predicted_pos[i], self.radius)
            obs_vel = (obs_pos - positions[i]) / self.dt
            final_pos[i] = (1 - w) * predicted_pos[i] + w * obs_pos
            final_vel[i] = (1 - w) * predicted_vel[i] + w * obs_vel

        next_grid = rasterize_tokens(final_pos, final_vel, self.n, self.radius)
        return final_pos, final_vel, new_hidden, next_grid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_token_model.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_model.py tests/test_token_model.py
git commit -m "feat: assemble TokenModel (init/step/occlusion-gated observation blend)"
```

---

## Task 8: Grid-space loss

**Files:**
- Create: `model/token_losses.py`
- Test: `tests/test_token_losses.py`

**Interfaces:**
- Consumes: `model.losses.weighted_channel_mse(pred: Tensor[B,C,H,W], target: Tensor[B,C,H,W], weights: Tensor[C]) -> Tensor[scalar]` (existing, unchanged).
- Produces: `token_grid_loss(pred_grid: Tensor[3,n,n], target_grid: Tensor[3,n,n], weights: Tensor[3]) -> Tensor[scalar]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_token_losses.py
import torch

from model.token_losses import token_grid_loss


def test_token_grid_loss_matches_manual_weighted_mse():
    pred = torch.zeros(3, 4, 4)
    target = torch.zeros(3, 4, 4)
    target[0] = 1.0  # PROB channel differs by 1.0 everywhere
    weights = torch.tensor([2.0, 0.5, 0.5])

    loss = token_grid_loss(pred, target, weights)
    expected = 2.0 * 1.0  # channel-0 MSE=1.0 * weight 2.0, other channels 0
    assert torch.allclose(loss, torch.tensor(expected))


def test_token_grid_loss_is_zero_for_identical_grids():
    grid = torch.rand(3, 5, 5)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss = token_grid_loss(grid, grid, weights)
    assert torch.allclose(loss, torch.tensor(0.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_token_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.token_losses'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/token_losses.py
from model.losses import weighted_channel_mse


def token_grid_loss(pred_grid, target_grid, weights):
    """Grid-space loss for the token model's rasterized output, reusing
    the existing per-channel MSE weighting (model.losses) so numbers stay
    directly comparable to the flow-warp/windowed-attention baselines.
    Operates on a single (3, n, n) frame -- the token pipeline has no
    batch dimension (see model.token_dataset)."""
    return weighted_channel_mse(pred_grid.unsqueeze(0), target_grid.unsqueeze(0), weights)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_token_losses.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_losses.py tests/test_token_losses.py
git commit -m "feat: add token model grid-space loss"
```

---

## Task 9: Training script (per-sample rollout, teacher-forcing ramp)

**Files:**
- Create: `model/token_train.py`
- Test: `tests/test_token_train.py`

**Interfaces:**
- Consumes: `BounceTokenSequenceDataset` (Task 3), `TokenModel` (Task 7), `token_grid_loss` (Task 8).
- Produces: `sampling_probability(epoch, ramp_epochs) -> float`; `token_rollout_loss(model, grid_seq, horizon, sampling_p, weights) -> Tensor[scalar]`; `train(args)`; CLI `main`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_token_train.py
from unittest.mock import patch

import torch

import model.token_train as token_train_module
from model.token_dataset import BounceTokenSequenceDataset
from model.token_model import TokenModel
from model.token_train import sampling_probability, token_rollout_loss


def test_sampling_probability_ramps_linearly_and_clamps():
    assert sampling_probability(epoch=0, ramp_epochs=10) == 0.0
    assert sampling_probability(epoch=5, ramp_epochs=10) == 0.5
    assert sampling_probability(epoch=10, ramp_epochs=10) == 1.0
    assert sampling_probability(epoch=20, ramp_epochs=10) == 1.0


def test_token_rollout_loss_overfits_a_single_sequence():
    torch.manual_seed(4738)
    dataset = BounceTokenSequenceDataset(num_samples=1, n=20, ball_range=(2, 3), seed=4738, horizon=3)
    grid_seq, _ = dataset[0]
    model = TokenModel(n=20, radius=0.75, dt=0.15, hidden_dim=8, neighbor_radius=3.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    weights = torch.tensor([1.0, 0.1, 0.1])

    first_loss = None
    last_loss = None
    for step in range(150):
        loss = token_rollout_loss(model, grid_seq, horizon=3, sampling_p=0.0, weights=weights)
        if step == 0:
            first_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()

    assert last_loss < first_loss * 0.5


def test_token_rollout_loss_self_feeds_when_sampling_p_is_one():
    torch.manual_seed(4738)
    dataset = BounceTokenSequenceDataset(num_samples=1, n=20, ball_range=(2, 3), seed=4738, horizon=3)
    grid_seq, _ = dataset[0]
    model = TokenModel(n=20, radius=0.75, dt=0.15, hidden_dim=8, neighbor_radius=3.0)
    weights = torch.tensor([1.0, 0.1, 0.1])

    loss = token_rollout_loss(model, grid_seq, horizon=3, sampling_p=1.0, weights=weights)
    assert torch.isfinite(loss)


def test_token_rollout_loss_self_feed_decided_per_step():
    torch.manual_seed(4738)
    dataset = BounceTokenSequenceDataset(num_samples=1, n=20, ball_range=(2, 3), seed=4738, horizon=4)
    grid_seq, _ = dataset[0]
    model = TokenModel(n=20, radius=0.75, dt=0.15, hidden_dim=8, neighbor_radius=3.0)
    weights = torch.tensor([1.0, 0.1, 0.1])

    # horizon=4 -> 3 prediction steps (frames 2, 3, 4); self_feed per
    # step: False, True, False
    with patch.object(token_train_module.random, "random", side_effect=[0.9, 0.1, 0.9]):
        with patch.object(model, "step", wraps=model.step) as mock_step:
            token_rollout_loss(model, grid_seq, horizon=4, sampling_p=0.5, weights=weights)

    assert mock_step.call_count == 3
    # step 2's call used ground-truth frame 2 as observed_frame (step 1 not self-fed)
    _, _, _, observed_frame_2 = mock_step.call_args_list[1].args
    assert torch.equal(observed_frame_2, grid_seq[2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_token_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.token_train'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/token_train.py
import argparse
import random

import torch
from torch.utils.data import DataLoader

from model.token_dataset import BounceTokenSequenceDataset
from model.token_model import TokenModel
from model.token_losses import token_grid_loss


def sampling_probability(epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, epoch / ramp_epochs))


def token_rollout_loss(model, grid_seq, horizon, sampling_p, weights):
    """Teacher-forced/self-feed rollout loss for one sequence sample.
    Mirrors model.train.rollout_loss's per-step self-feed coin flip
    (re-drawn every step, not once per rollout, per
    tests/test_train.py's regression test for the same reason). Token
    state (positions/velocities/hidden) always carries forward through
    the recurrence regardless of self-feed; only the *observed grid* fed
    into the observation branch is swapped for the model's own (detached)
    prediction on a self-fed step."""
    positions, velocities, hidden = model.init_tokens(grid_seq[0], grid_seq[1])
    observed_frame = grid_seq[1]
    total_loss = grid_seq.new_zeros(())
    num_steps = max(horizon - 1, 1)
    for step in range(num_steps):
        positions, velocities, hidden, pred_grid = model.step(
            positions, velocities, hidden, observed_frame
        )
        target_grid = grid_seq[step + 2]
        total_loss = total_loss + token_grid_loss(pred_grid, target_grid, weights)
        self_feed = random.random() < sampling_p
        observed_frame = pred_grid.detach() if self_feed else target_grid
    return total_loss / num_steps


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dataset = BounceTokenSequenceDataset(
        num_samples=args.num_samples, n=args.n, ball_range=(args.min_balls, args.max_balls),
        seed=args.seed, horizon=args.horizon,
    )
    loader = DataLoader(dataset, batch_size=None, shuffle=True)
    model = TokenModel(n=args.n, radius=0.75, dt=0.15, hidden_dim=args.hidden_dim,
                        neighbor_radius=args.neighbor_radius)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.tensor([1.0, 0.1, 0.1])

    for epoch in range(args.epochs):
        sampling_p = sampling_probability(epoch, args.ramp_epochs)
        epoch_loss = 0.0
        for grid_seq, _ in loader:
            loss = token_rollout_loss(model, grid_seq, args.horizon, sampling_p, weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch} sampling_p={sampling_p:.2f} loss={epoch_loss / len(dataset):.4f}", flush=True)
        torch.save(model.state_dict(), args.checkpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-balls", type=int, default=2)
    ap.add_argument("--max-balls", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ramp-epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--neighbor-radius", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--checkpoint", type=str, default="checkpoint_token.pt")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_token_train.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_train.py tests/test_token_train.py
git commit -m "feat: add token model training script"
```

---

## Task 10: Rollout video script (manual verification, no unit test)

**Files:**
- Create: `scripts/render_token_rollout_video.py`

No test file — this repo's other `render_*_rollout_video.py` scripts (`scripts/render_flownet_rollout_video.py`) are likewise not unit-tested; verification is the manual run in Step 2 below plus the standing project requirement (`[[feedback_auto_run_video_analysis_on_sim_finish]]`) to run the diagnostic-grid + unbiased-subagent review once a real checkpoint exists.

**Interfaces:**
- Consumes: `TokenModel` (Task 7), `model.dataset.make_scenario_uniform`, `bounce.make_grid/splat_all/step`.

- [ ] **Step 1: Write the script**

```python
# scripts/render_token_rollout_video.py
"""Renders a ground-truth vs. token-model-rollout comparison video for a
TokenModel checkpoint.

Usage:
    PYTHONPATH=. python3 scripts/render_token_rollout_video.py \
        --checkpoint checkpoint_token.pt \
        --out videos/token_model_rollout_comparison.mp4 \
        --num-steps 30
"""
import argparse
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import bounce
from model.dataset import make_scenario_uniform
from model.token_model import TokenModel


def load_model(checkpoint_path, n, hidden_dim, neighbor_radius):
    model = TokenModel(n=n, radius=0.75, dt=0.15, hidden_dim=hidden_dim, neighbor_radius=neighbor_radius)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def simulate_ground_truth(n, num_balls, seed, num_steps, dt=0.15, gravity=9.0,
                           radius=0.75, stiffness=400.0, substeps=8, vy=2.3):
    rng = random.Random(seed)
    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    frames = [np.array(G, dtype=np.float32)]
    for _ in range(num_steps):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
        frames.append(np.array(G, dtype=np.float32))
    return frames


def rollout(model, frame0, frame1, num_steps):
    g0 = torch.from_numpy(frame0.transpose(2, 0, 1))
    g1 = torch.from_numpy(frame1.transpose(2, 0, 1))
    frames = [frame0, frame1]
    with torch.no_grad():
        positions, velocities, hidden = model.init_tokens(g0, g1)
        observed = g1
        for _ in range(num_steps - 1):
            positions, velocities, hidden, pred_grid = model.step(positions, velocities, hidden, observed)
            frames.append(pred_grid.numpy().transpose(1, 2, 0))
            observed = pred_grid
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-balls", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--neighbor-radius", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=4738)
    args = ap.parse_args()

    model = load_model(args.checkpoint, args.n, args.hidden_dim, args.neighbor_radius)
    gt_frames = simulate_ground_truth(args.n, args.num_balls, args.seed, args.num_steps)
    pred_frames = rollout(model, gt_frames[0], gt_frames[1], args.num_steps)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].set_title("ground truth")
    axes[1].set_title("token model")
    im0 = axes[0].imshow(gt_frames[0][:, :, 0], vmin=0, vmax=1, cmap="viridis")
    im1 = axes[1].imshow(pred_frames[0][:, :, 0], vmin=0, vmax=1, cmap="viridis")

    def update(i):
        im0.set_data(gt_frames[i][:, :, 0])
        im1.set_data(pred_frames[i][:, :, 0])
        return im0, im1

    ani = animation.FuncAnimation(fig, update, frames=len(gt_frames), interval=1000 / 12)
    ani.save(args.out, writer="ffmpeg", fps=12)
    print(f"saved {args.out}", flush=True)
```

- [ ] **Step 2: Manual smoke run** (after Task 9's `train()` has produced at least a short-run `checkpoint_token.pt`)

Run:
```bash
PYTHONPATH=. python3 model/token_train.py --num-samples 50 --epochs 3 --checkpoint /tmp/smoke_token.pt
PYTHONPATH=. python3 scripts/render_token_rollout_video.py --checkpoint /tmp/smoke_token.pt --out videos/token_model_smoke.mp4 --num-steps 20
```
Expected: both commands exit 0 and `videos/token_model_smoke.mp4` is produced. This is a wiring smoke test only, not a quality claim — real validation follows the design spec's staged plan (Polaris training run, honest-horizon comparison, unbiased video review) once this plan's tasks are all merged.

- [ ] **Step 3: Commit**

```bash
git add scripts/render_token_rollout_video.py
git commit -m "feat: add token model rollout video rendering script"
```

---

## Post-implementation (not part of this plan's tasks)

Per the design spec's Validation plan, once Tasks 1-10 are merged: run a real Polaris training job at the spec's training scale, compute the honest-horizon step-count-until-breakdown metric against v6/v7 for direct comparison, and run the standing unbiased-subagent video review before drawing any conclusion about whether this architecture beats the current default. A synthetic-only check of this plan's components is not sufficient validation by itself (see `docs/debugging/findings-mass-conservation-loss.md`'s documented miss).
