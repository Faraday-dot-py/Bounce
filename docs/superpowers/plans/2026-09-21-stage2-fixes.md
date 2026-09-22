# Stage-2 Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three issues parked at the end of Stage-1: unmasked padding tokens in windowed attention, the target-only occupancy mask in the training loss, and the autoregressive checkerboard/hallucination artifact (via ICNR init + multi-step scheduled-sampling training).

**Architecture:** Five small, independently-testable changes to the existing Stage-1 codebase: a validity mask in the Swin block, a union-occupancy mask in the loss, ICNR-initialized Unpatchify weights, a sequence-producing dataset, and a scheduled-sampling multi-step training loop. No new files beyond tests; all changes land in existing `model/*.py` modules.

**Tech Stack:** PyTorch, pytest, existing `bounce.py` simulator.

**Spec:** `docs/superpowers/specs/2026-09-21-stage2-fixes-design.md`

## Global Constraints

- Seed `4738` for all new tests/scripts requiring a seed (matches existing convention, see `~/.claude/CLAUDE.md`).
- No architecture/parameter-count changes beyond what's specified (ICNR only reinitializes existing weights).
- Old pair-format dataset caches are incompatible with the new sequence format — this is expected; caches/checkpoints are gitignored and regenerated on Polaris.
- Follow existing code style: no comments unless non-obvious, no docstrings beyond what's already in the file, small focused functions matching the current `model/*.py` style.

---

### Task 1: Padding validity mask

**Files:**
- Modify: `model/windows.py` (add `compute_validity_mask`)
- Modify: `model/block.py:1-46` (`SwinBlock.forward`)
- Test: `tests/test_windows.py`, `tests/test_block.py`

**Interfaces:**
- Produces: `compute_validity_mask(Hp, Wp, orig_H, orig_W, window_size, device, roll_shift=0)` → tensor of shape `(num_windows, window_size*window_size, window_size*window_size)`, additive bias (`0.0` valid-valid, `-100.0` otherwise).

- [ ] **Step 1: Write failing tests for `compute_validity_mask`**

```python
# tests/test_windows.py (append)
from model.windows import compute_validity_mask


def test_validity_mask_all_valid_is_zero_bias():
    mask = compute_validity_mask(Hp=16, Wp=16, orig_H=16, orig_W=16, window_size=8, device="cpu")
    assert mask.shape == (4, 64, 64)
    assert torch.all(mask == 0.0)


def test_validity_mask_marks_padded_region():
    mask = compute_validity_mask(Hp=16, Wp=16, orig_H=10, orig_W=10, window_size=8, device="cpu")
    assert (mask != 0.0).any()
    # the window covering rows/cols 0-7 is fully inside the valid 10x10 region
    assert torch.all(mask[0] == 0.0)


def test_validity_mask_respects_roll_shift():
    mask_noroll = compute_validity_mask(16, 16, 10, 10, 8, "cpu", roll_shift=0)
    mask_roll = compute_validity_mask(16, 16, 10, 10, 8, "cpu", roll_shift=4)
    assert not torch.equal(mask_noroll, mask_roll)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_windows.py -v -k validity_mask`
Expected: FAIL with `ImportError: cannot import name 'compute_validity_mask'`

- [ ] **Step 3: Implement `compute_validity_mask` in `model/windows.py`**

```python
def compute_validity_mask(Hp, Wp, orig_H, orig_W, window_size, device, roll_shift=0):
    valid = torch.zeros((1, Hp, Wp, 1), device=device)
    valid[:, :orig_H, :orig_W, :] = 1.0
    if roll_shift:
        valid = torch.roll(valid, shifts=(-roll_shift, -roll_shift), dims=(1, 2))
    mask_windows = window_partition(valid, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    pair_valid = mask_windows.unsqueeze(1) * mask_windows.unsqueeze(2)
    return torch.where(pair_valid > 0, torch.zeros_like(pair_valid), torch.full_like(pair_valid, -100.0))
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_windows.py -v -k validity_mask`
Expected: PASS (3 tests)

- [ ] **Step 5: Write failing test for `SwinBlock` no-shift padding case**

```python
# tests/test_block.py (append)
def test_swin_block_preserves_shape_non_multiple_of_window_no_shift():
    block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=False)
    x = torch.randn(2, 25, 25, 16)
    out = block(x)
    assert out.shape == x.shape
```

This test alone won't catch the masking bug (shape-only), but confirms the no-shift+padding code path runs end to end once wired.

- [ ] **Step 6: Run test, verify it currently passes (shape-only, no regression yet)**

Run: `pytest tests/test_block.py -v -k no_shift`
Expected: PASS (padding already worked shape-wise before this change; this locks in the behavior while Step 7 adds real masking underneath)

- [ ] **Step 7: Wire validity mask into `SwinBlock.forward` in `model/block.py`**

```python
from model.windows import pad_to_multiple, window_partition, window_reverse, compute_shift_mask, compute_validity_mask
from model.attention import WindowAttention


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift, mlp_ratio=4.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        x, (orig_H, orig_W) = pad_to_multiple(x, self.window_size)
        Hp, Wp = x.shape[1], x.shape[2]
        has_padding = (Hp != orig_H) or (Wp != orig_W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = compute_shift_mask(Hp, Wp, self.window_size, self.shift_size, x.device)
            if has_padding:
                validity_mask = compute_validity_mask(
                    Hp, Wp, orig_H, orig_W, self.window_size, x.device, roll_shift=self.shift_size,
                )
                mask = torch.maximum(mask, validity_mask)
        else:
            mask = None
            if has_padding:
                mask = compute_validity_mask(Hp, Wp, orig_H, orig_W, self.window_size, x.device)

        windows = window_partition(x, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, C)
        attn_out = self.attn(windows, mask=mask)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_out, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x[:, :orig_H, :orig_W, :]
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        return x
```

- [ ] **Step 8: Run full block + windows test suite**

Run: `pytest tests/test_windows.py tests/test_block.py -v`
Expected: PASS (all tests)

- [ ] **Step 9: Commit**

```bash
git add model/windows.py model/block.py tests/test_windows.py tests/test_block.py
git commit -m "fix: mask padded tokens in windowed attention"
```

---

### Task 2: Union-occupancy loss mask

**Files:**
- Modify: `model/losses.py`
- Modify: `model/train.py:34` (call site, minimal — full rollout rewrite is Task 5)
- Test: `tests/test_losses.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `occupancy_weighted_mse(pred, target, source, weights, bg_weight=0.05)` — signature gains a required `source` (frame the prediction was made from) inserted after `target`. Task 5 relies on this exact signature.

- [ ] **Step 1: Update existing tests to pass `source` and add the vacate-case test**

```python
# tests/test_losses.py (full file)
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
    loss_bg = occupancy_weighted_mse(pred_bg_err, target, source, weights, bg_weight=0.0)
    assert torch.isclose(loss_bg, torch.tensor(0.0), atol=1e-6)

    pred_occ_err = target.clone()
    pred_occ_err[:, 0, 5, 5] = 5.0  # error only at the occupied pixel
    loss_occ = occupancy_weighted_mse(pred_occ_err, target, source, weights, bg_weight=0.0)
    assert loss_occ > 0.0


def test_bg_weight_only_discounts_prob_channel():
    target = torch.zeros(1, 3, 10, 10)
    target[:, 0, 5, 5] = 1.0  # occupied pixel; rest of grid is background
    source = target.clone()

    pred_vel_err = target.clone()
    pred_vel_err[:, 1, 0, 0] = 5.0  # VX error at a background pixel
    weights = torch.tensor([0.0, 1.0, 0.0])
    loss_vel_bg = occupancy_weighted_mse(pred_vel_err, target, source, weights, bg_weight=0.05)
    loss_vel_uniform = weighted_channel_mse(pred_vel_err, target, weights)
    assert torch.isclose(loss_vel_bg, loss_vel_uniform)


def test_bg_weight_one_matches_weighted_channel_mse():
    pred = torch.randn(2, 3, 10, 10)
    target = torch.randn(2, 3, 10, 10)
    source = torch.randn(2, 3, 10, 10)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss_occ = occupancy_weighted_mse(pred, target, source, weights, bg_weight=1.0)
    loss_uniform = weighted_channel_mse(pred, target, weights)
    assert torch.isclose(loss_occ, loss_uniform)


def test_vacated_cell_keeps_full_weight_via_source_occupancy():
    source = torch.zeros(1, 3, 10, 10)
    source[:, 0, 5, 5] = 1.0  # occupied at t
    target = torch.zeros(1, 3, 10, 10)  # empty at t+1 (vacated)
    weights = torch.tensor([1.0, 0.0, 0.0])

    pred_correct = target.clone()  # correctly predicts the cell clears
    pred_wrong = target.clone()
    pred_wrong[:, 0, 5, 5] = 5.0  # incorrectly predicts mass stayed

    loss_correct = occupancy_weighted_mse(pred_correct, target, source, weights, bg_weight=0.0)
    loss_wrong = occupancy_weighted_mse(pred_wrong, target, source, weights, bg_weight=0.0)
    assert torch.isclose(loss_correct, torch.tensor(0.0), atol=1e-6)
    assert loss_wrong > 0.0
```

- [ ] **Step 2: Run tests, verify failures**

Run: `pytest tests/test_losses.py -v`
Expected: FAIL (`TypeError: occupancy_weighted_mse() missing 1 required positional argument`)

- [ ] **Step 3: Implement the union-occupancy mask in `model/losses.py`**

```python
import torch


def weighted_channel_mse(pred, target, weights):
    per_channel = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    return (per_channel * weights).sum()


def occupancy_weighted_mse(pred, target, source, weights, bg_weight=0.05):
    target_occ = target[:, 0:1, :, :] > 1e-6
    source_occ = source[:, 0:1, :, :] > 1e-6
    occ = target_occ | source_occ
    prob_spatial_weight = torch.where(occ, 1.0, bg_weight)
    spatial_weight = torch.ones_like(target)
    spatial_weight[:, 0:1, :, :] = prob_spatial_weight
    weighted_sq_err = (pred - target) ** 2 * spatial_weight
    per_channel = weighted_sq_err.mean(dim=(0, 2, 3))
    return (per_channel * weights).sum()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_losses.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Fix the `model/train.py` call site (minimal — pass `g_t` as source)**

```python
# model/train.py:34, change:
loss = occupancy_weighted_mse(pred, g_t1, weights, bg_weight=args.bg_weight)
# to:
loss = occupancy_weighted_mse(pred, g_t1, g_t, weights, bg_weight=args.bg_weight)
```

- [ ] **Step 6: Run full test suite to confirm nothing else broke**

Run: `pytest -v`
Expected: PASS (Task 5 will restructure `train.py` further; this step just keeps the tree green in between tasks)

- [ ] **Step 7: Commit**

```bash
git add model/losses.py model/train.py tests/test_losses.py
git commit -m "fix: weight occupancy loss mask by source-or-target occupancy"
```

---

### Task 3: ICNR init for Unpatchify

**Files:**
- Modify: `model/patchify.py`
- Test: `tests/test_patchify.py`

**Interfaces:**
- Produces: `icnr_init(weight, patch_size, out_channels)` — in-place reinit of a `(patch_size*patch_size*out_channels, in_features)` weight tensor.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_patchify.py (append)
def test_icnr_init_sub_pixel_positions_start_equal():
    torch.manual_seed(4738)
    unpatch = Unpatchify(embed_dim=16, patch_size=2, out_channels=3)
    w = unpatch.proj.weight.data  # (12, 16)
    group_size = 4  # patch_size * patch_size
    out_channels = 3
    w_grouped = w.view(group_size, out_channels, -1)
    for c in range(out_channels):
        for g in range(1, group_size):
            assert torch.allclose(w_grouped[0, c], w_grouped[g, c])


def test_icnr_init_diverges_after_optimizer_step():
    torch.manual_seed(4738)
    unpatch = Unpatchify(embed_dim=16, patch_size=2, out_channels=3)
    opt = torch.optim.Adam(unpatch.parameters(), lr=0.1)
    x = torch.randn(2, 5, 5, 16)
    target = torch.randn(2, 3, 10, 10)
    out = unpatch(x)
    loss = ((out - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    w_grouped = unpatch.proj.weight.data.view(4, 3, -1)
    assert not torch.allclose(w_grouped[0, 0], w_grouped[1, 0])
```

- [ ] **Step 2: Run tests, verify first fails (weights currently independent, not equal)**

Run: `pytest tests/test_patchify.py -v -k icnr`
Expected: FAIL on `test_icnr_init_sub_pixel_positions_start_equal`

- [ ] **Step 3: Implement `icnr_init` and wire into `Unpatchify.__init__` in `model/patchify.py`**

```python
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


def icnr_init(weight, patch_size, out_channels):
    group_size = patch_size * patch_size
    out_features, in_features = weight.shape
    sub_weight = torch.empty(out_channels, in_features, device=weight.device, dtype=weight.dtype)
    nn.init.kaiming_uniform_(sub_weight, a=5 ** 0.5)
    weight.data.copy_(sub_weight.repeat(group_size, 1))


class Unpatchify(nn.Module):
    def __init__(self, embed_dim, patch_size, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        icnr_init(self.proj.weight, patch_size, out_channels)

    def forward(self, x):
        B, H, W, _ = x.shape
        x = self.proj(x)
        x = x.view(B, H, W, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, H * self.patch_size, W * self.patch_size)
        return x
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_patchify.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add model/patchify.py tests/test_patchify.py
git commit -m "fix: ICNR-init Unpatchify to avoid sub-pixel checkerboard artifact"
```

---

### Task 4: Sequence dataset (`generate_sequence`, `BounceSequenceDataset`)

**Files:**
- Modify: `model/dataset.py` (rename `generate_pair`→`generate_sequence`, `BouncePairDataset`→`BounceSequenceDataset`)
- Modify: `model/evaluate.py:1-30` (update `generate_pair` call site)
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces: `generate_sequence(balls, n, dt, gravity, radius, stiffness, substeps, horizon)` → `list` of `horizon + 1` `np.float32` arrays shape `(n, n, 3)`, index `0` is the pre-step frame.
- Produces: `BounceSequenceDataset(num_samples, n, ball_range, seed, horizon=3, ..., cache_path=None)`. `__getitem__` returns one `torch.Tensor` of shape `(horizon+1, 3, n, n)`.
- Consumes (Task 5): `sequence[:, k]` indexing along the horizon dimension after `DataLoader` batching (batch dim is dim 0, horizon is dim 1).

- [ ] **Step 1: Write failing tests for `generate_sequence`**

```python
# tests/test_dataset.py — replace the generate_pair tests near the top with:
import os
import random
import numpy as np
import bounce
from model.dataset import generate_sequence, make_scenario_uniform


def test_generate_sequence_shapes_and_prob_bounds():
    rng = random.Random(4738)
    balls = make_scenario_uniform(10, 50, 2.3, rng)
    frames = generate_sequence(balls, 50, 0.15, 9.0, 0.75, 400.0, 8, horizon=3)
    assert len(frames) == 4
    for frame in frames:
        assert frame.shape == (50, 50, bounce.NUM_CHANNELS)
        assert frame.dtype == np.float32
        assert (frame[:, :, bounce.PROB] >= 0.0).all()
        assert (frame[:, :, bounce.PROB] < 1.0).all()


def test_generate_sequence_frames_differ_across_steps():
    rng = random.Random(4738)
    balls = make_scenario_uniform(10, 50, 2.3, rng)
    frames = generate_sequence(balls, 50, 0.15, 9.0, 0.75, 400.0, 8, horizon=3)
    assert not np.allclose(frames[0], frames[1])
    assert not np.allclose(frames[1], frames[2])
    assert not np.allclose(frames[2], frames[3])
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_dataset.py -v -k generate_sequence`
Expected: FAIL with `ImportError: cannot import name 'generate_sequence'`

- [ ] **Step 3: Rename `generate_pair`→`generate_sequence` in `model/dataset.py`**

```python
def generate_sequence(balls, n, dt, gravity, radius, stiffness, substeps, horizon):
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    frames = [np.array(G, dtype=np.float32)]
    for _ in range(horizon):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
        frames.append(np.array(G, dtype=np.float32))
    return frames
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_dataset.py -v -k generate_sequence`
Expected: PASS (2 tests)

- [ ] **Step 5: Write failing tests for `BounceSequenceDataset`**

```python
# tests/test_dataset.py (replace the BouncePairDataset tests)
import torch
from model.dataset import BounceSequenceDataset


def test_bounce_sequence_dataset_shapes_and_determinism():
    ds1 = BounceSequenceDataset(num_samples=6, n=50, ball_range=(5, 15), seed=4738, horizon=3)
    ds2 = BounceSequenceDataset(num_samples=6, n=50, ball_range=(5, 15), seed=4738, horizon=3)
    assert len(ds1) == 6
    seq = ds1[0]
    assert seq.shape == (4, 3, 50, 50)
    assert isinstance(seq, torch.Tensor)
    seq_again = ds2[0]
    assert torch.equal(seq, seq_again)


def test_bounce_sequence_dataset_cache_roundtrip(tmp_path):
    cache_path = str(tmp_path / "cache.npz")
    ds1 = BounceSequenceDataset(num_samples=4, n=50, ball_range=(5, 15), seed=4738, horizon=3, cache_path=cache_path)
    assert os.path.exists(cache_path)
    ds2 = BounceSequenceDataset(num_samples=4, n=50, ball_range=(5, 15), seed=4738, horizon=3, cache_path=cache_path)
    assert len(ds2) == len(ds1)
    for i in range(len(ds1)):
        assert torch.equal(ds1[i], ds2[i])


def test_bounce_sequence_dataset_cache_rejects_mismatched_config(tmp_path):
    cache_path = str(tmp_path / "cache.npz")
    BounceSequenceDataset(num_samples=4, n=50, ball_range=(5, 15), seed=4738, horizon=3, cache_path=cache_path)
    try:
        BounceSequenceDataset(num_samples=4, n=50, ball_range=(5, 15), seed=4738, horizon=2, cache_path=cache_path)
        assert False, "expected ValueError for mismatched cache config"
    except ValueError:
        pass
```

Keep the existing `test_clustered_scenario_balls_stay_within_cluster_radius` and `test_settled_scenario_balls_move_toward_high_x_under_gravity` tests unchanged — `make_scenario_clustered`/`make_scenario_settled` aren't touched by this task.

- [ ] **Step 6: Run tests, verify failure**

Run: `pytest tests/test_dataset.py -v -k bounce_sequence`
Expected: FAIL with `ImportError: cannot import name 'BounceSequenceDataset'`

- [ ] **Step 7: Rename and rework `BouncePairDataset`→`BounceSequenceDataset` in `model/dataset.py`**

```python
class BounceSequenceDataset(Dataset):
    SCENARIOS = ("uniform", "clustered", "settled")

    def __init__(self, num_samples, n, ball_range, seed, horizon=3, dt=0.15, gravity=9.0,
                 radius=0.75, stiffness=400.0, substeps=8, vy=2.3,
                 cluster_radius=3.0, settle_steps=200, cache_path=None):
        config = {
            "num_samples": num_samples, "n": n, "ball_range": tuple(ball_range),
            "seed": seed, "horizon": horizon,
        }
        if cache_path is not None and os.path.exists(cache_path):
            self.samples = self._load_cache(cache_path, config)
            print(f"[dataset] loaded {len(self.samples)} samples from {cache_path}", flush=True)
            return

        self.samples = []
        rng = random.Random(seed)
        t_start = time.time()
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
            frames = generate_sequence(balls, n, dt, gravity, radius, stiffness, substeps, horizon)
            self.samples.append(np.stack(frames))
            if (i + 1) % 100 == 0 or (i + 1) == num_samples:
                elapsed = time.time() - t_start
                print(f"[dataset] generated {i + 1}/{num_samples} samples ({elapsed:.1f}s elapsed)", flush=True)

        if cache_path is not None:
            self._save_cache(cache_path, config)
            print(f"[dataset] saved {len(self.samples)} samples to {cache_path}", flush=True)

    def _save_cache(self, cache_path, config):
        seq_arr = np.stack(self.samples)
        np.savez(
            cache_path, sequences=seq_arr,
            num_samples=config["num_samples"], n=config["n"],
            ball_range=np.array(config["ball_range"]), seed=config["seed"],
            horizon=config["horizon"],
        )

    @staticmethod
    def _load_cache(cache_path, config):
        data = np.load(cache_path)
        cached_config = {
            "num_samples": int(data["num_samples"]), "n": int(data["n"]),
            "ball_range": tuple(int(x) for x in data["ball_range"]), "seed": int(data["seed"]),
            "horizon": int(data["horizon"]),
        }
        if cached_config != config:
            raise ValueError(
                f"dataset cache at {cache_path} was generated with config {cached_config}, "
                f"but this run requested {config}. Delete the cache or use a different --cache-path."
            )
        seq_arr = data["sequences"]
        return [seq_arr[i] for i in range(seq_arr.shape[0])]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        return torch.from_numpy(seq.transpose(0, 3, 1, 2)).clone()
```

- [ ] **Step 8: Run tests, verify pass**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (all tests)

- [ ] **Step 9: Update `model/evaluate.py` call sites**

```python
# model/evaluate.py: change import and both call sites
from model.dataset import make_scenario_uniform, generate_sequence

# held_out_loss, inside the loop:
frames = generate_sequence(balls, n, dt, gravity, radius, stiffness, substeps, horizon=1)
g_t, g_t1 = frames[0], frames[1]
g_t, g_t1 = _pair_to_batch(g_t, g_t1, device)
```

- [ ] **Step 10: Run evaluate tests to confirm no regression**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 11: Commit**

```bash
git add model/dataset.py model/evaluate.py tests/test_dataset.py
git commit -m "feat: generate multi-frame sequences for rollout training"
```

---

### Task 5: Multi-step scheduled-sampling training loop

**Files:**
- Modify: `model/train.py`
- Modify: `scripts/polaris_train.sh`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `BounceSequenceDataset` (Task 4), `occupancy_weighted_mse(pred, target, source, weights, bg_weight)` (Task 2).
- Produces: `train(args)` with new `args.horizon` (default `3`) and `args.sampling_ramp_epochs` (default `None`, resolved to `args.epochs` when unset).

- [ ] **Step 1: Write failing test for the scheduled-sampling probability schedule**

```python
# tests/test_train.py (add alongside the existing overfit test)
from model.train import sampling_probability


def test_sampling_probability_ramps_linearly_and_clamps():
    assert sampling_probability(epoch=0, ramp_epochs=10) == 0.0
    assert sampling_probability(epoch=5, ramp_epochs=10) == 0.5
    assert sampling_probability(epoch=10, ramp_epochs=10) == 1.0
    assert sampling_probability(epoch=20, ramp_epochs=10) == 1.0
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest tests/test_train.py -v -k sampling_probability`
Expected: FAIL with `ImportError: cannot import name 'sampling_probability'`

- [ ] **Step 3: Write failing test for the multi-step rollout loss**

```python
# tests/test_train.py (add)
import torch
from torch.utils.data import DataLoader
from model.dataset import BounceSequenceDataset
from model.net import BounceNextFrameModel
from model.losses import occupancy_weighted_mse
from model.train import rollout_loss


def test_rollout_loss_overfits_a_single_batch():
    torch.manual_seed(4738)
    dataset = BounceSequenceDataset(num_samples=4, n=20, ball_range=(3, 6), seed=4738, horizon=3)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    sequence = next(iter(loader))

    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = torch.tensor([1.0, 0.1, 0.1])

    first_loss = None
    last_loss = None
    for step in range(50):
        loss = rollout_loss(model, sequence, horizon=3, sampling_p=0.0, weights=weights, bg_weight=0.05)
        if step == 0:
            first_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()

    assert last_loss < first_loss * 0.5


def test_rollout_loss_self_feeds_when_sampling_p_is_one():
    torch.manual_seed(4738)
    dataset = BounceSequenceDataset(num_samples=2, n=20, ball_range=(3, 6), seed=4738, horizon=3)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    sequence = next(iter(loader))
    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    weights = torch.tensor([1.0, 0.1, 0.1])

    # sampling_p=1.0 must run without error even though the model's own
    # (detached) predictions feed forward instead of ground truth
    loss = rollout_loss(model, sequence, horizon=3, sampling_p=1.0, weights=weights, bg_weight=0.05)
    assert torch.isfinite(loss)
```

- [ ] **Step 4: Run tests, verify failure**

Run: `pytest tests/test_train.py -v -k rollout_loss`
Expected: FAIL with `ImportError: cannot import name 'rollout_loss'`

- [ ] **Step 5: Implement `sampling_probability`, `rollout_loss`, and rewire `train()` in `model/train.py`**

```python
import argparse
import random

import torch
from torch.utils.data import DataLoader

from model.dataset import BounceSequenceDataset
from model.net import BounceNextFrameModel
from model.losses import occupancy_weighted_mse


def sampling_probability(epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, epoch / ramp_epochs)


def rollout_loss(model, sequence, horizon, sampling_p, weights, bg_weight):
    device = next(model.parameters()).device
    sequence = sequence.to(device)
    weights = weights.to(device)
    frame = sequence[:, 0]
    total = 0.0
    self_feed = random.random() < sampling_p
    for k in range(1, horizon + 1):
        target = sequence[:, k]
        pred = model(frame)
        total = total + occupancy_weighted_mse(pred, target, frame, weights, bg_weight=bg_weight)
        frame = pred.detach() if self_feed else target
    return total / horizon


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BounceSequenceDataset(
        num_samples=args.num_samples,
        n=args.n,
        ball_range=(args.min_balls, args.max_balls),
        seed=args.seed,
        horizon=args.horizon,
        cache_path=args.cache_path,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = BounceNextFrameModel(
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        window_size=args.window_size,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.tensor(args.channel_weights, device=device)
    ramp_epochs = args.sampling_ramp_epochs if args.sampling_ramp_epochs is not None else args.epochs

    for epoch in range(args.epochs):
        p = sampling_probability(epoch, ramp_epochs)
        total_loss = 0.0
        for sequence in loader:
            loss = rollout_loss(model, sequence, args.horizon, p, weights, args.bg_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"epoch {epoch} loss {avg:.6f} sampling_p {p:.3f}")
        torch.save(model.state_dict(), args.checkpoint)


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-balls", type=int, default=50)
    ap.add_argument("--max-balls", type=int, default=250)
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--channel-weights", type=float, nargs=3, default=[1.0, 0.1, 0.1])
    ap.add_argument("--bg-weight", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--sampling-ramp-epochs", type=int, default=None)
    ap.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    ap.add_argument("--cache-path", type=str, default=None)
    return ap


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
```

Note: `test_training_loop_overfits_a_single_batch` (the pre-existing test using `BouncePairDataset`/`weighted_channel_mse` directly) must be removed from `tests/test_train.py` — it exercises a training pattern (`BouncePairDataset`) that no longer exists; `test_rollout_loss_overfits_a_single_batch` (Step 3) is its replacement.

- [ ] **Step 6: Run the full train test suite**

Run: `pytest tests/test_train.py -v`
Expected: PASS (3 tests: `sampling_probability`, `rollout_loss` overfit, `rollout_loss` self-feed)

- [ ] **Step 7: Update `scripts/polaris_train.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=bounce-stage2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=bounce-stage2-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"

echo "[$(date -Iseconds)] starting pip install"
pip install -r requirements.txt
echo "[$(date -Iseconds)] pip install done"

mkdir -p checkpoints

echo "[$(date -Iseconds)] starting training"
python -m model.train \
  --n 50 --min-balls 50 --max-balls 250 \
  --num-samples 3000 --batch-size 16 --epochs 20 \
  --embed-dim 128 --depth 6 --num-heads 4 --window-size 8 \
  --horizon 3 \
  --seed 4738 \
  --checkpoint checkpoints/stage2.pt \
  --cache-path checkpoints/dataset_cache_seq.npz
echo "[$(date -Iseconds)] training done"
```

(`--job-name`/`--output` renamed to `bounce-stage2`, checkpoint to `stage2.pt`, cache path changed to `dataset_cache_seq.npz` — the old `dataset_cache.npz` is pair-format and incompatible with `BounceSequenceDataset`; using a new filename avoids a stale-cache `ValueError` on Polaris.)

- [ ] **Step 8: Run the full local test suite**

Run: `pytest -v`
Expected: PASS, all tests across the repo

- [ ] **Step 9: Commit**

```bash
git add model/train.py scripts/polaris_train.sh tests/test_train.py
git commit -m "feat: multi-step scheduled-sampling rollout training"
```

---

### Task 6: Polaris validation run

**Files:**
- None (execution task — uses `scripts/polaris_train.sh` from Task 5)

**Interfaces:**
- Consumes: `scripts/polaris_train.sh`, `model/evaluate.py::rollout_divergence`.

- [ ] **Step 1: Upload updated repo to Polaris**

Use the `polaris_upload` tool (see [[infra_polaris_for_training]]) to sync the working tree, including `scripts/polaris_train.sh`, `model/`, `bounce.py`, `requirements.txt`.

- [ ] **Step 2: Submit the training job**

Use `polaris_submit_job` with `scripts/polaris_train.sh`. Monitor via `polaris_job_status`/`polaris_job_logs` until it completes or fails; if it fails, diagnose from the log output before resubmitting (don't blind-retry).

- [ ] **Step 3: Download the resulting checkpoint**

Use `polaris_download` to pull `checkpoints/stage2.pt` locally.

- [ ] **Step 4: Compare rollout divergence against `stage1.pt`**

Run `model.evaluate.rollout_divergence` (or the existing pilot/rollout-video tooling from Stage-1) against both `stage1.pt` and `stage2.pt` at matched settings, and check whether occupied-cell ballooning within the first 10-20 self-fed steps is materially reduced, per the spec's validation gate.

- [ ] **Step 5: Record the result**

Commit any new eval script output/artifacts produced (not the checkpoint itself — stays gitignored), with the key rollout-divergence metric in the commit message, per [[user's standing rule on committing experiment results]] (`~/.claude/CLAUDE.md`: "After each experiment completes: commit scripts + results with the key metric in the commit message").
