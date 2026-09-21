# Windowed-Attention Next-Frame Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, and validate a windowed-attention model that predicts `bounce.py`'s next grid frame, trained at `n=50`/50-250 balls, checked for generalization to 500 balls without retraining.

**Architecture:** Patchify the `n×n×3` grid into `2×2` tokens, run through a stack of Swin-style shifted-window attention blocks with learned relative positional bias, unpatchify back to a residual `ΔG`, output `G_t + ΔG`. See spec for full rationale.

**Tech Stack:** Python, PyTorch, NumPy, pytest. No other framework/packaging infra — matches the existing repo's zero-dependency, single-purpose style as closely as an ML subsystem reasonably can.

**Spec:** `docs/superpowers/specs/2026-09-21-windowed-attention-model-design.md`

## Global Constraints

- Grid size `n` must be even (patch size is fixed at 2 grid cells; `n` must divide evenly).
- Patch size: 2×2 grid cells, fixed, never scaled with `n`.
- Window size: 8×8 patches (16×16 grid cells), fixed, never scaled with `n`.
- Positional encoding: learned relative bias only, bounded by window size — no absolute positional embeddings anywhere.
- Output head: local unpatchify + residual (`G_t + ΔG`) — never a global-bottleneck encoder-decoder.
- Physics defaults (match `bounce.py`'s CLI defaults unless a task says otherwise): `dt=0.15`, `gravity=9.0`, `radius=0.75`, `stiffness=400.0`, `substeps=8`, `vy=2.3`.
- Seed: `4738` for all stochastic generation (per project standing rule), unless a task needs a distinct seed for a specific comparison.
- Training compute: Polaris, not TIDE.
- Stage 1 training scale: `n=50`, 50–250 balls. First generalization check: 500 balls, same `n=50` (per explicit instruction — do not scale `n` for this check; `n`-scaling for larger stages is an open question deferred past this plan).

---

## File Structure

```
bounce.py                          # existing, untouched
conftest.py                        # new: puts repo root on sys.path for pytest
requirements.txt                   # new: torch, numpy, pytest
model/
  __init__.py
  dataset.py                       # scenario generation + (G_t, G_t+1) pair dataset
  windows.py                       # window partition/reverse, padding, shift mask
  attention.py                     # WindowAttention + RelativePositionBias
  block.py                         # SwinBlock (windowed attn + MLP + shift)
  patchify.py                      # PatchEmbed, Unpatchify
  net.py                           # BounceNextFrameModel (full assembly)
  losses.py                        # weighted_channel_mse
  train.py                         # training CLI entrypoint
  evaluate.py                      # held-out loss, rollout stability, generalization check
tests/
  test_dataset.py
  test_windows.py
  test_attention.py
  test_patchify.py
  test_net.py
  test_losses.py
scripts/
  polaris_train.sh                 # SBATCH-style job script for Polaris
```

Each `model/*.py` file gets its own test file. Files split by responsibility (windowing mechanics vs. attention math vs. patch embedding are independently testable and independently reviewable).

---

### Task 1: Project scaffolding

**Files:**
- Create: `conftest.py`
- Create: `requirements.txt`
- Create: `model/__init__.py`

**Interfaces:**
- Produces: repo root importable as `bounce` from anywhere under `model/` or `tests/`, via `conftest.py` inserting repo root onto `sys.path` at pytest collection time.

- [ ] **Step 1: Create `conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

- [ ] **Step 2: Create `requirements.txt`**

```
torch>=2.1
numpy>=1.24
pytest>=7.4
```

- [ ] **Step 3: Create `model/__init__.py`** (empty file, makes `model` a package)

- [ ] **Step 4: Install dependencies and verify pytest runs cleanly**

Run: `pip install -r requirements.txt && pytest --collect-only`
Expected: exits 0, "no tests ran" (no test files exist yet).

- [ ] **Step 5: Commit**

```bash
git add conftest.py requirements.txt model/__init__.py
git commit -m "chore: scaffold model package and test infra"
```

---

### Task 2: Dataset — pair generation core

**Files:**
- Create: `model/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `bounce.init_balls`, `bounce.make_grid`, `bounce.splat_all`, `bounce.step`, `bounce.PROB`, `bounce.NUM_CHANNELS`.
- Produces:
  - `generate_pair(balls, n, dt, gravity, radius, stiffness, substeps) -> (np.ndarray[n,n,3], np.ndarray[n,n,3])` — channel-last, `float32`.
  - `make_scenario_uniform(num_balls, n, vy, rng) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import random
import numpy as np
import bounce
from model.dataset import generate_pair, make_scenario_uniform


def test_generate_pair_shapes_and_prob_bounds():
    rng = random.Random(4738)
    balls = make_scenario_uniform(10, 50, 2.3, rng)
    g_t, g_t1 = generate_pair(balls, 50, 0.15, 9.0, 0.75, 400.0, 8)
    assert g_t.shape == (50, 50, bounce.NUM_CHANNELS)
    assert g_t1.shape == (50, 50, bounce.NUM_CHANNELS)
    assert g_t.dtype == np.float32
    assert (g_t[:, :, bounce.PROB] >= 0.0).all()
    assert (g_t[:, :, bounce.PROB] < 1.0).all()


def test_generate_pair_differs_after_one_step():
    rng = random.Random(4738)
    balls = make_scenario_uniform(10, 50, 2.3, rng)
    g_t, g_t1 = generate_pair(balls, 50, 0.15, 9.0, 0.75, 400.0, 8)
    assert not np.allclose(g_t, g_t1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.dataset'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/dataset.py
import numpy as np
import bounce


def make_scenario_uniform(num_balls, n, vy, rng):
    return bounce.init_balls(num_balls, n, vy, rng)


def generate_pair(balls, n, dt, gravity, radius, stiffness, substeps):
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    g_t = np.array(G, dtype=np.float32)
    bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
    g_t1 = np.array(G, dtype=np.float32)
    return g_t, g_t1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add model/dataset.py tests/test_dataset.py
git commit -m "feat: add core (G_t, G_t+1) pair generation from bounce.py"
```

---

### Task 3: Dataset — clustered and settled curriculum scenarios

**Files:**
- Modify: `model/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `bounce.make_grid`, `bounce.step` (for settling rollout).
- Produces:
  - `make_scenario_clustered(num_balls, n, vy, rng, cluster_radius) -> list[dict]`
  - `make_scenario_settled(num_balls, n, vy, rng, radius, gravity, stiffness, dt, substeps, settle_steps) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_dataset.py
from model.dataset import make_scenario_clustered, make_scenario_settled


def test_clustered_scenario_balls_stay_within_cluster_radius():
    rng = random.Random(4738)
    balls = make_scenario_clustered(20, 50, 2.3, rng, cluster_radius=3.0)
    xs = [b["x"] for b in balls]
    ys = [b["y"] for b in balls]
    assert max(xs) - min(xs) <= 2 * 3.0 + 1e-6
    assert max(ys) - min(ys) <= 2 * 3.0 + 1e-6


def test_settled_scenario_balls_move_toward_high_x_under_gravity():
    rng = random.Random(4738)
    start = make_scenario_uniform(20, 50, 0.0, rng)
    start_mean_x = sum(b["x"] for b in start) / len(start)
    rng2 = random.Random(4738)
    settled = make_scenario_settled(
        20, 50, 0.0, rng2, radius=0.75, gravity=9.0, stiffness=400.0,
        dt=0.15, substeps=8, settle_steps=200,
    )
    settled_mean_x = sum(b["x"] for b in settled) / len(settled)
    assert settled_mean_x > start_mean_x
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset.py -v -k "clustered or settled"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# append to model/dataset.py

def make_scenario_clustered(num_balls, n, vy, rng, cluster_radius):
    lo, hi = cluster_radius, (n - 1.0) - cluster_radius
    cx = rng.uniform(lo, hi)
    cy = rng.uniform(lo, hi)
    balls = []
    for _ in range(num_balls):
        x = min(max(cx + rng.uniform(-cluster_radius, cluster_radius), 0.0), n - 1.0)
        y = min(max(cy + rng.uniform(-cluster_radius, cluster_radius), 0.0), n - 1.0)
        vx = rng.uniform(-vy, vy)
        vy_ = rng.uniform(-vy, vy)
        balls.append({"x": x, "y": y, "vx": vx, "vy": vy_})
    return balls


def make_scenario_settled(num_balls, n, vy, rng, radius, gravity, stiffness, dt, substeps, settle_steps):
    balls = bounce.init_balls(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    for _ in range(settle_steps):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
    return balls
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/dataset.py tests/test_dataset.py
git commit -m "feat: add clustered and gravity-settled curriculum scenarios"
```

---

### Task 4: Dataset — PyTorch Dataset wrapper

**Files:**
- Modify: `model/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `make_scenario_uniform`, `make_scenario_clustered`, `make_scenario_settled`, `generate_pair`.
- Produces: `BouncePairDataset(torch.utils.data.Dataset)` — `__getitem__(idx) -> (torch.Tensor[3,n,n], torch.Tensor[3,n,n])`, `__len__() -> int`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dataset.py
import torch
from model.dataset import BouncePairDataset


def test_bounce_pair_dataset_shapes_and_determinism():
    ds1 = BouncePairDataset(num_samples=6, n=50, ball_range=(5, 15), seed=4738)
    ds2 = BouncePairDataset(num_samples=6, n=50, ball_range=(5, 15), seed=4738)
    assert len(ds1) == 6
    g_t, g_t1 = ds1[0]
    assert g_t.shape == (3, 50, 50)
    assert g_t1.shape == (3, 50, 50)
    assert isinstance(g_t, torch.Tensor)
    g_t_again, _ = ds2[0]
    assert torch.equal(g_t, g_t_again)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v -k dataset_shapes`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# add to top of model/dataset.py
import random
import torch
from torch.utils.data import Dataset

# append to model/dataset.py

class BouncePairDataset(Dataset):
    SCENARIOS = ("uniform", "clustered", "settled")

    def __init__(self, num_samples, n, ball_range, seed, dt=0.15, gravity=9.0,
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
            g_t, g_t1 = generate_pair(balls, n, dt, gravity, radius, stiffness, substeps)
            self.samples.append((g_t, g_t1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        g_t, g_t1 = self.samples[idx]
        g_t = torch.from_numpy(g_t.transpose(2, 0, 1)).clone()
        g_t1 = torch.from_numpy(g_t1.transpose(2, 0, 1)).clone()
        return g_t, g_t1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add model/dataset.py tests/test_dataset.py
git commit -m "feat: add BouncePairDataset combining all three curriculum scenarios"
```

---

### Task 5: Window partition, reverse, and padding

**Files:**
- Create: `model/windows.py`
- Test: `tests/test_windows.py`

**Interfaces:**
- Produces:
  - `pad_to_multiple(x: Tensor[B,H,W,C], window_size: int) -> (Tensor, (orig_H, orig_W))`
  - `window_partition(x: Tensor[B,H,W,C], window_size: int) -> Tensor[num_windows*B, window_size, window_size, C]`
  - `window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor[B,H,W,C]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_windows.py
import torch
from model.windows import pad_to_multiple, window_partition, window_reverse


def test_pad_to_multiple_pads_up_and_reports_original_size():
    x = torch.randn(1, 10, 10, 4)
    padded, (orig_h, orig_w) = pad_to_multiple(x, window_size=8)
    assert padded.shape == (1, 16, 16, 4)
    assert (orig_h, orig_w) == (10, 10)


def test_pad_to_multiple_no_op_when_already_multiple():
    x = torch.randn(1, 16, 16, 4)
    padded, (orig_h, orig_w) = pad_to_multiple(x, window_size=8)
    assert padded.shape == (1, 16, 16, 4)


def test_window_partition_reverse_round_trip():
    x = torch.randn(2, 16, 16, 4)
    windows = window_partition(x, window_size=8)
    assert windows.shape == (2 * 4, 8, 8, 4)  # 2 batch * (16/8)*(16/8)=4 windows
    reconstructed = window_reverse(windows, window_size=8, H=16, W=16)
    assert torch.allclose(reconstructed, x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_windows.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.windows'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/windows.py
import torch
import torch.nn.functional as F


def pad_to_multiple(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    return x, (H, W)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    B = windows.shape[0] // (num_windows_h * num_windows_w)
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_windows.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add model/windows.py tests/test_windows.py
git commit -m "feat: add window partition/reverse with padding for non-multiple grids"
```

---

### Task 6: Shift mask

**Files:**
- Modify: `model/windows.py`
- Test: `tests/test_windows.py`

**Interfaces:**
- Consumes: `window_partition`.
- Produces: `compute_shift_mask(H: int, W: int, window_size: int, shift_size: int, device) -> Tensor[num_windows, window_size*window_size, window_size*window_size]`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_windows.py
from model.windows import compute_shift_mask


def test_shift_mask_shape_and_self_attend_is_always_allowed():
    mask = compute_shift_mask(H=16, W=16, window_size=8, shift_size=4, device="cpu")
    num_windows = (16 // 8) * (16 // 8)
    assert mask.shape == (num_windows, 64, 64)
    # a token always "attends" to itself (zero relative region id difference)
    diag = torch.diagonal(mask, dim1=-2, dim2=-1)
    assert torch.all(diag == 0.0)


def test_shift_mask_has_some_masked_entries():
    mask = compute_shift_mask(H=16, W=16, window_size=8, shift_size=4, device="cpu")
    assert (mask != 0.0).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_windows.py -v -k shift_mask`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# append to model/windows.py

def compute_shift_mask(H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_windows.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add model/windows.py tests/test_windows.py
git commit -m "feat: add shifted-window attention mask"
```

---

### Task 7: Relative position bias

**Files:**
- Create: `model/attention.py`
- Test: `tests/test_attention.py`

**Interfaces:**
- Produces: `RelativePositionBias(nn.Module)` — `__init__(window_size, num_heads)`, `forward() -> Tensor[num_heads, window_size*window_size, window_size*window_size]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_attention.py
import torch
from model.attention import RelativePositionBias


def test_relative_position_bias_shape():
    bias_module = RelativePositionBias(window_size=8, num_heads=4)
    bias = bias_module()
    assert bias.shape == (4, 64, 64)


def test_relative_position_bias_table_size_independent_of_grid():
    # table size depends only on window_size, never on any grid dimension
    bias_module = RelativePositionBias(window_size=8, num_heads=4)
    expected_table_size = (2 * 8 - 1) * (2 * 8 - 1)
    assert bias_module.bias_table.shape == (expected_table_size, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_attention.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.attention'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/attention.py
import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        N = self.window_size * self.window_size
        bias = self.bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        return bias
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_attention.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add model/attention.py tests/test_attention.py
git commit -m "feat: add learned relative position bias"
```

---

### Task 8: Window attention module

**Files:**
- Modify: `model/attention.py`
- Test: `tests/test_attention.py`

**Interfaces:**
- Consumes: `RelativePositionBias`.
- Produces: `WindowAttention(nn.Module)` — `__init__(dim, window_size, num_heads)`, `forward(x: Tensor[B_, N, C], mask=None) -> Tensor[B_, N, C]` where `N = window_size**2`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_attention.py
from model.attention import WindowAttention


def test_window_attention_output_shape_no_mask():
    attn = WindowAttention(dim=16, window_size=8, num_heads=4)
    x = torch.randn(3, 64, 16)  # 3 windows, 64 tokens each, dim 16
    out = attn(x)
    assert out.shape == (3, 64, 16)


def test_window_attention_output_shape_with_mask():
    attn = WindowAttention(dim=16, window_size=8, num_heads=4)
    num_windows = 4
    x = torch.randn(num_windows * 2, 64, 16)  # batch 2, 4 windows each
    mask = torch.zeros(num_windows, 64, 64)
    out = attn(x, mask=mask)
    assert out.shape == (num_windows * 2, 64, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_attention.py -v -k WindowAttention`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# append to model/attention.py

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_bias = RelativePositionBias(window_size, num_heads)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos_bias().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_attention.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/attention.py tests/test_attention.py
git commit -m "feat: add windowed multi-head attention with relative bias and mask"
```

---

### Task 9: Swin block (windowed attention + shift + MLP)

**Files:**
- Create: `model/block.py`
- Test: `tests/test_block.py`

**Interfaces:**
- Consumes: `model.windows.{pad_to_multiple, window_partition, window_reverse, compute_shift_mask}`, `model.attention.WindowAttention`.
- Produces: `SwinBlock(nn.Module)` — `__init__(dim, num_heads, window_size, shift, mlp_ratio=4.0)`, `forward(x: Tensor[B,H,W,C]) -> Tensor[B,H,W,C]` (same shape in and out, including non-multiple-of-window-size `H`/`W`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_block.py
import torch
from model.block import SwinBlock


def test_swin_block_preserves_shape_multiple_of_window():
    block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=False)
    x = torch.randn(2, 16, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_swin_block_preserves_shape_non_multiple_of_window():
    block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=True)
    x = torch.randn(2, 25, 25, 16)  # not divisible by 8
    out = block(x)
    assert out.shape == x.shape


def test_swin_block_shift_and_noshift_both_run():
    x = torch.randn(1, 16, 16, 16)
    for shift in (False, True):
        block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=shift)
        out = block(x)
        assert out.shape == x.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_block.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.block'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/block.py
import torch
import torch.nn as nn

from model.windows import pad_to_multiple, window_partition, window_reverse, compute_shift_mask
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

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = compute_shift_mask(Hp, Wp, self.window_size, self.shift_size, x.device)
        else:
            mask = None

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_block.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/block.py tests/test_block.py
git commit -m "feat: add SwinBlock combining shifted window attention and MLP"
```

---

### Task 10: Patch embed and unpatchify

**Files:**
- Create: `model/patchify.py`
- Test: `tests/test_patchify.py`

**Interfaces:**
- Produces:
  - `PatchEmbed(nn.Module)` — `__init__(in_channels, patch_size, embed_dim)`, `forward(x: Tensor[B,C,n,n]) -> Tensor[B, n/patch_size, n/patch_size, embed_dim]`.
  - `Unpatchify(nn.Module)` — `__init__(embed_dim, patch_size, out_channels)`, `forward(x: Tensor[B,H,W,embed_dim]) -> Tensor[B, out_channels, H*patch_size, W*patch_size]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_patchify.py
import torch
from model.patchify import PatchEmbed, Unpatchify


def test_patch_embed_shape():
    embed = PatchEmbed(in_channels=3, patch_size=2, embed_dim=16)
    x = torch.randn(2, 3, 50, 50)
    out = embed(x)
    assert out.shape == (2, 25, 25, 16)


def test_unpatchify_shape():
    unpatch = Unpatchify(embed_dim=16, patch_size=2, out_channels=3)
    x = torch.randn(2, 25, 25, 16)
    out = unpatch(x)
    assert out.shape == (2, 3, 50, 50)


def test_patch_embed_resolution_independent():
    embed = PatchEmbed(in_channels=3, patch_size=2, embed_dim=16)
    for n in (50, 300):
        x = torch.randn(1, 3, n, n)
        out = embed(x)
        assert out.shape == (1, n // 2, n // 2, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_patchify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.patchify'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/patchify.py
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


class Unpatchify(nn.Module):
    def __init__(self, embed_dim, patch_size, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, x):
        B, H, W, _ = x.shape
        x = self.proj(x)
        x = x.view(B, H, W, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, H * self.patch_size, W * self.patch_size)
        return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_patchify.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add model/patchify.py tests/test_patchify.py
git commit -m "feat: add PatchEmbed and Unpatchify (conv-based patch projection)"
```

---

### Task 11: Full model assembly

**Files:**
- Create: `model/net.py`
- Test: `tests/test_net.py`

**Interfaces:**
- Consumes: `model.patchify.{PatchEmbed, Unpatchify}`, `model.block.SwinBlock`.
- Produces: `BounceNextFrameModel(nn.Module)` — `__init__(in_channels=3, patch_size=2, embed_dim=128, depth=6, num_heads=4, window_size=8, mlp_ratio=4.0)`, `forward(g_t: Tensor[B,3,n,n]) -> Tensor[B,3,n,n]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_net.py
import torch
from model.net import BounceNextFrameModel


def test_model_output_shape_matches_input():
    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    g_t = torch.randn(2, 3, 50, 50)
    out = model(g_t)
    assert out.shape == g_t.shape


def test_model_runs_at_a_different_resolution_without_retraining():
    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    for n in (50, 300):
        g_t = torch.randn(1, 3, n, n)
        out = model(g_t)
        assert out.shape == (1, 3, n, n)


def test_model_output_is_residual_identity_plus_delta():
    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    g_t = torch.randn(1, 3, 50, 50)
    out = model(g_t)
    assert not torch.equal(out, g_t)  # delta is non-trivial (random init weights)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_net.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.net'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/net.py
import torch.nn as nn

from model.patchify import PatchEmbed, Unpatchify
from model.block import SwinBlock


class BounceNextFrameModel(nn.Module):
    def __init__(self, in_channels=3, patch_size=2, embed_dim=128, depth=6,
                 num_heads=4, window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)
        self.blocks = nn.ModuleList([
            SwinBlock(embed_dim, num_heads, window_size, shift=(i % 2 == 1), mlp_ratio=mlp_ratio)
            for i in range(depth)
        ])
        self.unpatchify = Unpatchify(embed_dim, patch_size, in_channels)

    def forward(self, g_t):
        x = self.patch_embed(g_t)
        for block in self.blocks:
            x = block(x)
        delta = self.unpatchify(x)
        return g_t + delta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_net.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add model/net.py tests/test_net.py
git commit -m "feat: assemble BounceNextFrameModel end to end"
```

---

### Task 12: Weighted per-channel loss

**Files:**
- Create: `model/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Produces: `weighted_channel_mse(pred: Tensor[B,3,n,n], target: Tensor[B,3,n,n], weights: Tensor[3]) -> Tensor[scalar]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_losses.py
import torch
from model.losses import weighted_channel_mse


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.losses'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/losses.py
def weighted_channel_mse(pred, target, weights):
    per_channel = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    return (per_channel * weights).sum()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_losses.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add model/losses.py tests/test_losses.py
git commit -m "feat: add per-channel weighted MSE loss"
```

---

### Task 13: Training script with local smoke test

**Files:**
- Create: `model/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `model.dataset.BouncePairDataset`, `model.net.BounceNextFrameModel`, `model.losses.weighted_channel_mse`.
- Produces: `train(args) -> None` (writes checkpoint to `args.checkpoint`); CLI entrypoint via `python -m model.train`.

- [ ] **Step 1: Write the failing test (overfit sanity check)**

This is the standard "does the training loop actually learn anything" test:
train on a single tiny batch for many steps and confirm loss drops sharply.

```python
# tests/test_train.py
import torch
from torch.utils.data import DataLoader

from model.dataset import BouncePairDataset
from model.net import BounceNextFrameModel
from model.losses import weighted_channel_mse


def test_training_loop_overfits_a_single_batch():
    torch.manual_seed(4738)
    dataset = BouncePairDataset(num_samples=4, n=20, ball_range=(3, 6), seed=4738)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    g_t, g_t1 = next(iter(loader))

    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = torch.tensor([1.0, 0.1, 0.1])

    first_loss = None
    last_loss = None
    for step in range(50):
        pred = model(g_t)
        loss = weighted_channel_mse(pred, g_t1, weights)
        if step == 0:
            first_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()

    assert last_loss < first_loss * 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — either import error (`model.train` doesn't exist yet, though this
test only needs `model.net`/`model.dataset`/`model.losses` which already exist,
so it may instead fail on the assertion if something upstream is broken; if it
already passes at this point, that confirms Tasks 2-12 compose correctly, note
it and proceed).

- [ ] **Step 3: Write minimal implementation**

```python
# model/train.py
import argparse

import torch
from torch.utils.data import DataLoader

from model.dataset import BouncePairDataset
from model.net import BounceNextFrameModel
from model.losses import weighted_channel_mse


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BouncePairDataset(
        num_samples=args.num_samples,
        n=args.n,
        ball_range=(args.min_balls, args.max_balls),
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = BounceNextFrameModel(
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        window_size=args.window_size,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.tensor(args.channel_weights, device=device)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for g_t, g_t1 in loader:
            g_t, g_t1 = g_t.to(device), g_t1.to(device)
            pred = model(g_t)
            loss = weighted_channel_mse(pred, g_t1, weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"epoch {epoch} loss {avg:.6f}")
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
    ap.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    return ap


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS (loss on the single overfit batch drops by more than half over 50 steps)

- [ ] **Step 5: Run a short end-to-end smoke test from the CLI**

Run:
```bash
python -m model.train --n 20 --min-balls 3 --max-balls 6 --num-samples 20 \
  --batch-size 4 --epochs 2 --embed-dim 16 --depth 2 --checkpoint /tmp/smoke.pt
```
Expected: prints two epoch loss lines, exits 0, `/tmp/smoke.pt` exists.

- [ ] **Step 6: Commit**

```bash
git add model/train.py tests/test_train.py
git commit -m "feat: add training loop with overfit sanity test"
```

---

### Task 14: Evaluation — held-out loss, rollout stability, generalization check

**Files:**
- Create: `model/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `model.net.BounceNextFrameModel`, `model.dataset.{make_scenario_uniform, generate_pair}`, `model.losses.weighted_channel_mse`, `bounce.step`.
- Produces:
  - `held_out_loss(model, n, ball_range, num_samples, seed, weights) -> float`
  - `rollout_divergence(model, n, num_balls, seed, num_steps) -> list[float]` (per-step MSE between model rollout and ground-truth `bounce.step` rollout)
  - `generalization_check(model, n, ball_count, num_samples, seed, weights) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import torch

from model.net import BounceNextFrameModel
from model.evaluate import held_out_loss, rollout_divergence, generalization_check


def test_held_out_loss_is_finite_nonnegative():
    torch.manual_seed(4738)
    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss = held_out_loss(model, n=20, ball_range=(3, 6), num_samples=3, seed=4738, weights=weights)
    assert loss >= 0.0
    assert loss == loss  # not NaN


def test_rollout_divergence_returns_one_value_per_step():
    torch.manual_seed(4738)
    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    divergence = rollout_divergence(model, n=20, num_balls=5, seed=4738, num_steps=4)
    assert len(divergence) == 4
    assert all(d >= 0.0 for d in divergence)


def test_generalization_check_runs_at_higher_ball_count():
    torch.manual_seed(4738)
    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss = generalization_check(model, n=20, ball_count=40, num_samples=2, seed=4738, weights=weights)
    assert loss >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.evaluate'`

- [ ] **Step 3: Write minimal implementation**

```python
# model/evaluate.py
import random

import torch

import bounce
from model.dataset import make_scenario_uniform, generate_pair
from model.losses import weighted_channel_mse


def _pair_to_batch(g_t, g_t1, device):
    g_t = torch.from_numpy(g_t.transpose(2, 0, 1)).unsqueeze(0).to(device)
    g_t1 = torch.from_numpy(g_t1.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return g_t, g_t1


def held_out_loss(model, n, ball_range, num_samples, seed, weights,
                   dt=0.15, gravity=9.0, radius=0.75, stiffness=400.0, substeps=8, vy=2.3):
    device = next(model.parameters()).device
    rng = random.Random(seed)
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(num_samples):
            num_balls = rng.randint(*ball_range)
            balls = make_scenario_uniform(num_balls, n, vy, rng)
            g_t, g_t1 = generate_pair(balls, n, dt, gravity, radius, stiffness, substeps)
            g_t, g_t1 = _pair_to_batch(g_t, g_t1, device)
            pred = model(g_t)
            total += weighted_channel_mse(pred, g_t1, weights.to(device)).item()
    return total / num_samples


def rollout_divergence(model, n, num_balls, seed, num_steps,
                        dt=0.15, gravity=9.0, radius=0.75, stiffness=400.0, substeps=8, vy=2.3):
    device = next(model.parameters()).device
    rng = random.Random(seed)
    model.eval()

    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    import numpy as np
    g_true = np.array(G, dtype=np.float32)
    g_pred = torch.from_numpy(g_true.transpose(2, 0, 1)).unsqueeze(0).to(device)

    divergences = []
    with torch.no_grad():
        for _ in range(num_steps):
            bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
            g_true = np.array(G, dtype=np.float32)
            g_true_t = torch.from_numpy(g_true.transpose(2, 0, 1)).unsqueeze(0).to(device)

            g_pred = model(g_pred)
            divergences.append(((g_pred - g_true_t) ** 2).mean().item())
    return divergences


def generalization_check(model, n, ball_count, num_samples, seed, weights,
                          dt=0.15, gravity=9.0, radius=0.75, stiffness=400.0, substeps=8, vy=2.3):
    return held_out_loss(
        model, n, ball_range=(ball_count, ball_count), num_samples=num_samples,
        seed=seed, weights=weights, dt=dt, gravity=gravity, radius=radius,
        stiffness=stiffness, substeps=substeps, vy=vy,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add model/evaluate.py tests/test_evaluate.py
git commit -m "feat: add held-out loss, rollout divergence, and generalization check"
```

---

### Task 15: Polaris training job script

**Files:**
- Create: `scripts/polaris_train.sh`

**Interfaces:**
- Consumes: `model/train.py`'s CLI.
- Produces: a shell script suitable for `mcp__polaris__polaris_submit_job`'s `script_or_code` argument, or for direct upload + sbatch.

- [ ] **Step 1: Write the job script**

```bash
#!/bin/bash
#SBATCH --job-name=bounce-stage1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=bounce-stage1-%j.log

set -euo pipefail
cd "$(dirname "$0")/.."

pip install -r requirements.txt

python -m model.train \
  --n 50 --min-balls 50 --max-balls 250 \
  --num-samples 3000 --batch-size 16 --epochs 20 \
  --embed-dim 128 --depth 6 --num-heads 4 --window-size 8 \
  --seed 4738 \
  --checkpoint checkpoints/stage1.pt
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/polaris_train.sh`

- [ ] **Step 3: Commit**

```bash
git add scripts/polaris_train.sh
git commit -m "chore: add Polaris stage-1 training job script"
```

---

### Task 16: Run stage-1 training on Polaris and check 500-ball generalization

**Files:** none created — this task runs the pipeline built in Tasks 1-15.

- [ ] **Step 1: Confirm Polaris connection**

Use `mcp__polaris__polaris_status`; if not connected, use `mcp__polaris__polaris_start`.

- [ ] **Step 2: Upload the repo**

Use `mcp__polaris__polaris_upload` for `bounce.py`, `conftest.py`, `requirements.txt`,
`model/`, `scripts/polaris_train.sh` to a remote directory (e.g. `bounce/`).

- [ ] **Step 3: Submit the training job**

Use `mcp__polaris__polaris_submit_job` with `script_or_code` set to the contents of
`scripts/polaris_train.sh` (or reference the uploaded script path per that tool's
conventions), `gres_gpu=1`, `time="02:00:00"`.

- [ ] **Step 4: Monitor**

Poll with `mcp__polaris__polaris_job_status` and `mcp__polaris__polaris_job_logs`
until the job completes. Confirm the log shows 20 decreasing epoch-loss lines and
no traceback.

- [ ] **Step 5: Download the checkpoint**

Use `mcp__polaris__polaris_download` to fetch `checkpoints/stage1.pt` locally to
`checkpoints/stage1.pt`.

- [ ] **Step 6: Run the 500-ball generalization check locally**

```python
import torch
from model.net import BounceNextFrameModel
from model.evaluate import held_out_loss, generalization_check

model = BounceNextFrameModel(embed_dim=128, depth=6, num_heads=4, window_size=8)
model.load_state_dict(torch.load("checkpoints/stage1.pt", map_location="cpu"))

weights = torch.tensor([1.0, 0.1, 0.1])
train_scale_loss = held_out_loss(model, n=50, ball_range=(50, 250), num_samples=50, seed=1, weights=weights)
gen_500_loss = generalization_check(model, n=50, ball_count=500, num_samples=50, seed=1, weights=weights)
print(f"train-scale held-out loss: {train_scale_loss:.6f}")
print(f"500-ball generalization loss: {gen_500_loss:.6f}")
```

Run this as a one-off script (e.g. `python -c "..."` written to a scratch file, or
inline) and report both numbers.

- [ ] **Step 7: Record the result**

Report the two loss numbers to the user, and whether `gen_500_loss` stays within
the same order of magnitude as `train_scale_loss` (the working definition of
"generalizes" for this first checkpoint — a precise threshold wasn't set in the
spec, so flag this explicitly rather than silently picking one).

No commit for this task — it's a run-and-report step, not a code change. If the
generalization check reveals a problem (e.g. loss blows up), that's a new finding
to bring back to the user before deciding next steps, not something to silently
patch around.

---

## Self-Review Notes

- **Spec coverage:** I/O contract → Tasks 2, 4, 11. Tokenization → Task 10. Windowed attention + shift + relative bias → Tasks 5-9. Output head (unpatchify + residual) → Tasks 10-11. Loss → Task 12. Data pipeline/curriculum → Tasks 2-4. Validation plan stage 1 (train n=50/50-250, check 500-ball generalization) → Tasks 13-16. Later validation stages (n=300 mid-scale, 1M target) are explicitly out of scope for this plan per the user's "try n=50 first" instruction — a follow-on plan once stage 1 results are in.
- **Placeholder scan:** no TBD/TODO markers; all code blocks are complete, runnable implementations.
- **Type consistency:** `generate_pair` returns `(np.ndarray, np.ndarray)` used consistently in Tasks 2-4, 14. `BounceNextFrameModel.forward` signature (`g_t -> g_t1`) consistent across Tasks 11, 13, 14. `weighted_channel_mse(pred, target, weights)` signature consistent across Tasks 12, 13, 14.
- **Scope:** single cohesive subsystem (one model, one training run, one generalization check) — not decomposed further, since each task already produces independently testable, working software and the whole plan targets one deliverable (an answer to "does stage-1 training generalize to 500 balls").
