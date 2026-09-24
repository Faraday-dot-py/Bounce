"""Spatially tiled sim step for ball counts whose per-token state and
attention intermediates do not fit on one GPU. Token state (x, y, vx, vy,
hidden) lives in host memory, one buffer per horizontal strip of the grid;
each tick streams one strip at a time through the GPU together with a halo of
the neighbouring strips' tokens within `neighbor_radius` of the boundary, so
every token sees exactly the neighbours it would in one global step. Tokens
that cross a strip boundary are handed to the neighbouring strip.

Usage: see scripts/bench_scaling.py.
"""
import torch


class TiledSim:
    def __init__(self, model, strips, device, slack=1.1, pin=True, track_ids=False):
        self.model = model
        self.model.dynamics.cell_graph = True
        self.model.dynamics.local_softmax = True
        self.n = model.n
        self.strips = strips
        self.rows = self.n / strips
        self.device = device
        self.slack = slack
        self.pin = pin and device.type == "cuda"
        self.hidden_dim = model.dynamics.hidden_dim
        self.track_ids = track_ids
        self.cols = 4 + self.hidden_dim + (1 if track_ids else 0)
        self.bufs = [None] * strips
        self.counts = [0] * strips
        self.band_low = [None] * strips
        self.band_high = [None] * strips
        self.max_speed = 30.0

    def lo(self, s):
        return s * self.rows

    def strip_of(self, x):
        return (x / self.rows).floor().long().clamp(0, self.strips - 1)

    def load(self, strip_states):
        """strip_states: one (m, cols) device tensor per strip, holding the
        tokens inside that strip."""
        for s, state in enumerate(strip_states):
            cap = max(int(state.shape[0] * self.slack), state.shape[0] + 1024)
            self.bufs[s] = torch.empty((cap, self.cols), pin_memory=self.pin)
            self.bufs[s][:state.shape[0]].copy_(state)
            self.counts[s] = state.shape[0]
            self.band_low[s], self.band_high[s] = self.bands(state, s)

    def bands(self, state, s):
        x = state[:, 0]
        return (state[x < self.lo(s) + self.model.dynamics.neighbor_radius],
                state[x >= self.lo(s + 1) - self.model.dynamics.neighbor_radius])

    def load_flat(self, positions, velocities, hidden=None):
        """Splits a flat token set into strips (test / small-N convenience)."""
        cols = [positions, velocities,
                torch.zeros(positions.shape[0], self.hidden_dim, device=positions.device) if hidden is None else hidden]
        if self.track_ids:
            cols.append(torch.arange(positions.shape[0], device=positions.device, dtype=torch.float32)[:, None])
        state = torch.cat(cols, dim=1)
        dest = self.strip_of(state[:, 0])
        self.load([state[dest == s] for s in range(self.strips)])

    def init_random(self, count, seed):
        """Uniform random positions and velocities (the benchmark start),
        generated strip by strip on the device."""
        n = self.n
        states = []
        for s in range(self.strips):
            m = count // self.strips + (count % self.strips if s == self.strips - 1 else 0)
            gen = torch.Generator(device=self.device).manual_seed(seed + s)
            r = torch.rand(m, 4, generator=gen, device=self.device)
            x = (self.lo(s) + r[:, 0] * self.rows).clamp(1.5, n - 1.5)
            y = r[:, 1] * (n - 3) + 1.5
            cols = [x[:, None], y[:, None], (r[:, 2:4] - 0.5) * 4.6, torch.zeros(m, self.hidden_dim, device=self.device)]
            if self.track_ids:
                cols.append(torch.zeros(m, 1, device=self.device))
            states.append(torch.cat(cols, dim=1))
        self.load(states)

    def alive(self):
        return sum(self.counts)

    def step(self):
        S = self.strips
        hd = self.hidden_dim
        incoming = [[] for _ in range(S)]
        next_low, next_high = [None] * S, [None] * S
        with torch.no_grad():
            for s in range(S):
                own = self.bufs[s][:self.counts[s]].to(self.device, non_blocking=True)
                m = own.shape[0]
                parts = [own]
                if s > 0:
                    parts.append(self.band_high[s - 1])
                if s < S - 1:
                    parts.append(self.band_low[s + 1])
                nodes = torch.cat(parts)
                p, v, h, _ = self.model.step_free(nodes[:, 0:2].contiguous(), nodes[:, 2:4].contiguous(),
                                                  nodes[:, 4:4 + hd].contiguous(), render=False)
                cols = [p[:m], v[:m], h[:m]]
                if self.track_ids:
                    cols.append(own[:, 4 + hd:])
                new = torch.cat(cols, dim=1)
                del nodes, p, v, h, own
                finite = torch.isfinite(new[:, :4]).all(dim=1)
                if not bool(finite.all()):
                    new = new[finite]
                top = self.n - 1.0
                pos, vel = new[:, 0:2], new[:, 2:4]
                outward = ((pos < 0) & (vel < 0)) | ((pos > top) & (vel > 0))
                new[:, 2:4] = torch.where(outward, torch.zeros_like(vel), vel)
                new[:, 0:2] = pos.clamp(0.0, top)
                speed = new[:, 2:4].norm(dim=1, keepdim=True).clamp(min=1e-6)
                new[:, 2:4] = new[:, 2:4] * (speed.clamp(max=self.max_speed) / speed)
                dest = self.strip_of(new[:, 0])
                stay = dest == s
                kept = new[stay]
                self.bufs[s][:kept.shape[0]].copy_(kept, non_blocking=True)
                self.counts[s] = kept.shape[0]
                moved = 0
                for d in (s - 1, s + 1):
                    if 0 <= d < S:
                        mig = new[dest == d]
                        moved += mig.shape[0]
                        if mig.shape[0]:
                            incoming[d].append(mig)
                assert kept.shape[0] + moved == new.shape[0], "token moved more than one strip in a tick"
                next_low[s], next_high[s] = self.bands(kept, s)
        for d in range(S):
            if incoming[d]:
                mig = torch.cat(incoming[d])
                c = self.counts[d]
                assert c + mig.shape[0] <= self.bufs[d].shape[0], "strip buffer overflow; raise slack"
                self.bufs[d][c:c + mig.shape[0]].copy_(mig, non_blocking=True)
                self.counts[d] = c + mig.shape[0]
                lo, hi = self.bands(mig, d)
                next_low[d] = torch.cat([next_low[d], lo])
                next_high[d] = torch.cat([next_high[d], hi])
        self.band_low, self.band_high = next_low, next_high
