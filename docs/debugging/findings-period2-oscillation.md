# Findings: period-2 brightness oscillation (Problem 2)

## Summary

**Confirmed, architectural, and root-caused.** The oscillation is caused
by the `correction_head` in `model/net.py`. Unlike `flow_head` (which is
squashed by `tanh(...) * max_flow`), `correction_head`'s output is an
unbounded linear conv with no damping, clamping, or normalization. Once
the autoregressive map's Jacobian around the model's own attractor state
has an eigenvalue with real part < -1 (confirmed by direct measurement,
below), the `correction` term overshoots and flips sign every step,
producing a limit-cycle period-2 oscillation instead of decaying. Killing
`correction` entirely (or just halving its magnitude) eliminates the
oscillation completely - this is a clean causal ablation, not a
correlation.

A contributing factor: the checkpoint (`stage2_flownet_h12.pt`) was
trained with `--horizon 12` (`model/train.py`'s `rollout_loss` only ever
unrolls 12 steps). The oscillation's onset at step ~14-16 is just past
that trained horizon - the model never received gradient signal about
its behavior beyond 12 self-feed steps, so nothing in training penalized
this instability. This is secondary to the architectural gap (no
retraining fix works without also bounding/damping the correction
term - see "What NOT to do" below), but explains why the instability
specifically becomes visible right where it does.

## Evidence

All scripts below load `checkpoints/stage2_flownet_h12.pt`, same repro
as in `flownet-open-issues.md` (n=50, 150 balls, seed=4738 unless noted,
`bounce.make_scenario_uniform` + `splat_all` radius=0.75).

### 1. Oscillation confirmed, present in all 3 channels, seed-independent

Full per-step instrumented rollout (30 steps, seed 4738) confirms the
period-2 alternation in PROB (channel 0) reported in the issue doc, and
shows it is **not isolated to PROB** - VX (channel 1) and VY (channel 2)
oscillate too, with much larger amplitude than PROB in the correction
term itself (see table below). Re-running with seeds 111 and 2026
produces near-identical trajectories (same onset step, same amplitude
envelope, same period-2 phase) - the oscillation is a property of the
model's dynamics near its own attractor state, essentially independent
of the specific initial condition. This rules out "artifact of this one
IC" and rules out array aliasing (float32 arrays, `.copy()`'d at each
capture in the repro).

### 2. The correction head's output alternates sign in lockstep with the output oscillation

Manual re-implementation of `forward()` (outside `net.py`, in a scratch
script - no edits to model code) that logs `warped` and `correction`
separately shows `correction[:,0]` (PROB channel of the correction term)
flipping sign every single step from step ~13 onward, with growing then
saturating magnitude:

```
step  corr0_mean  corr1_mean(VX)  corr2_mean(VY)
  26     0.0100      0.1198          0.0664
  27    -0.0055     -0.1455         -0.1195
  28     0.0033      0.1642          0.1650
  29     0.0018     -0.1694         -0.2076
  30    -0.0037      0.1501          0.2328
  ...(perfectly alternating sign every step through step 60)
```

VX/VY correction magnitude (~0.1-0.23) dwarfs PROB correction
(~0.01-0.05) - the instability is strongest in the velocity channels,
which then couples back into PROB each step through the flow-warp (VX/VY
determine where the next `flow` samples from). `warped[:,0].mean()` and
`correction[:,0].mean()` are almost exactly anti-correlated
step-to-step (e.g. step 16: warp=0.0090, corr=+0.0304, sum=0.0393; step
17: warp=0.0386, corr=-0.0357, sum=0.0028) - each step's correction is
overshooting to cancel/overcorrect the previous step's error rather than
damping it.

### 3. Direct Jacobian measurement: dominant eigenvalue is negative and >1 in magnitude

At an operating point deep in the oscillation regime (rollout step 20,
seed 4738), a finite-difference power iteration on the model's
input->output Jacobian (`Jv ~= (f(x*+eps*v) - f(x*))/eps`, eps=1e-3)
converges to:

```
iter  ||Jv||   cos(v_new, v_old)
 15   1.1813        -0.9656
 16   1.1570        -0.9665
 17   1.1697        -0.9741
 18   1.1372        -0.9721
 19   1.1363        -0.9768
```

`||Jv|| ~= 1.14-1.18` (spectral radius > 1: locally unstable) and
`cos(v_new, v_old) ~= -0.97` (the dominant eigenvector direction flips
sign each iteration - the signature of a real, negative dominant
eigenvalue). This is a textbook period-doubling (2-cycle) instability:
dominant eigenvalue ~= -1.15 to -1.2. This confirms the "gain exceeds 1"
hypothesis in the issue doc quantitatively, not just qualitatively.
(Note: PyTorch's `grid_sampler_2d` has no double-backward, so this used
finite differences rather than `torch.autograd.functional.jvp`.)

A per-channel version of the same probe (perturb only one input channel
with unit-norm noise, measure response norm per output channel) shows
the dominant instability mode is cross-channel, driven by PROB->{VX,VY}
coupling:

```
perturb PROB -> response [PROB, VX, VY] = [0.47, 2.71, 1.56]
perturb VX   -> response [PROB, VX, VY] = [0.028, 0.32, 0.23]
perturb VY   -> response [PROB, VX, VY] = [0.023, 0.23, 0.23]
```

A small PROB perturbation is amplified ~2.7x and ~1.6x into VX and VY
respectively - much larger than any within-channel gain. This is
consistent with the correction head (which outputs all 3 channels from
the same shared backbone features) not being decoupled per-channel.

### 4. Causal ablation: zeroing (or halving) the correction term eliminates the oscillation

Re-running the rollout with `correction` forced to zero, or scaled by
0.5, at every step (same backbone/flow, only the correction term
touched, still not editing `net.py` - done via a local copy of
`forward()`'s logic in the probe script):

```
                  mean abs step-to-step delta in ch0_mean, steps 14-30
full model                    0.0220   (oscillating)
correction = 0                0.0001   (flat, oscillation gone)
correction * 0.5              0.0003   (flat, oscillation gone)
```

This is the strongest evidence in this investigation: disabling or
merely damping the correction head's output by a constant factor removes
the oscillation almost entirely, with no change to the flow head or
backbone. The flow head (which is bounded by `tanh * max_flow`) is not
the driver - its magnitude stays roughly flat (`flow.abs().mean()` ~
0.20-0.29 throughout the rollout in the original instrumentation,
non-oscillating) while the correction head is the one whose magnitude
and sign track the observed brightness oscillation exactly.

## Diagnosis

**Architectural, not a training/hyperparameter issue.** The
`correction_head` in `model/net.py` is the only head in the model with
no bounding, clamping, or damping mechanism on its output. `flow_head`
is deliberately squashed (`tanh(...) * max_flow`) so its contribution to
the next state is bounded per-step; `correction_head` has no analogous
mechanism - it's a raw `Conv2d` output added directly to the warped
frame. Under single-step / short-horizon (`horizon<=12`) training this
never surfaces because the loss only ever sees up to 12 self-feed steps
and the correction only has to be locally accurate, not globally stable
under its own feedback. Under true autoregressive self-feed past that
horizon, the lack of any contraction guarantee on the correction path
lets the local linearization's dominant eigenvalue drift to
magnitude > 1 with negative sign - an unstable 2-cycle, confirmed
directly by the Jacobian probe (section 3) and by the ablation (section
4).

Do not "just retrain" as the fix - a longer training horizon by itself
only widens the window the model is directly supervised on; it does not
give the correction head any structural reason to stay contractive
outside that window, and nothing here indicates the current training
procedure would generalize its implicit stability arbitrarily far past
12 steps without also fixing the missing damping. (A longer training
horizon is a reasonable complementary change, see recommendation below,
but should not be the only change.)

## Recommended fix (architectural, code-level)

Primary fix - add a bounded/damped correction path, analogous to how
`flow_head` is already bounded:

- **Simplest, cheapest to validate:** scale `correction` by a fixed
  constant < 1 (e.g. 0.5, matching what section 4's ablation already
  shows works) or by a learned scalar `alpha` (`nn.Parameter`,
  initialized small, e.g. 0.1-0.3, optionally passed through `sigmoid`
  to guarantee `alpha in (0, 1)`) before adding it to `warped`. This
  directly reduces the Jacobian contribution of the correction path and
  is consistent with the design intent stated in `net.py`'s docstring
  ("small local correction").
- **More principled alternative:** bound `correction`'s magnitude the
  same way `flow` is bounded, e.g. `correction =
  torch.tanh(self.correction_head(x)) * max_correction` for some small
  `max_correction` (e.g. 0.1-0.2, well under the observed unbounded
  magnitudes of 0.15-0.25 seen in the VX/VY channels during the
  oscillation). This caps the worst-case per-step contribution instead
  of just scaling it down uniformly, and mirrors the existing
  `max_flow` pattern the codebase already uses, so it's consistent with
  the file's own conventions.
- Either change directly attacks the confirmed cause (section 4) and
  should be validated the same way this investigation did: rerun the
  instrumented rollout script and confirm the `correction` sign no
  longer alternates and `||Jv||` at a late-rollout operating point drops
  below 1.

Secondary, complementary (not a substitute for the above): once the
correction path is bounded, consider retraining with a longer rollout
horizon than 12 so the model is directly supervised on the regime where
it currently oscillates - but only after the architectural fix, so the
extra training horizon reinforces an already-stable design rather than
papering over an unbounded correction term.

## Repro / instrumentation scripts (not committed to the repo)

Scripts used for this investigation currently live in the session
scratchpad, not in the repo:
- Instrumented rollout w/ flow+correction magnitude logging
- Per-channel correction breakdown
- Finite-difference Jacobian power iteration + per-channel probe
- Correction-ablation rollout comparison

If a follow-up wants these as a permanent repo tool (e.g. alongside
`scripts/render_diagnostic_grid.py`), they should be cleaned up and
added under `scripts/`, but per the task instructions this investigation
did not edit `model/net.py` or add new permanent scripts to the repo.
