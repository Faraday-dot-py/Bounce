import torch

from model.token_model import TokenModel
from scripts.realtime_sim import Sim
from scripts.tiled_sim import TiledSim


def make_model(n):
    torch.manual_seed(4738)
    model = TokenModel(n=n, radius=0.75, dt=0.15, hidden_dim=32, neighbor_radius=4.0, free_rollout=True,
                       velocity_readout=True, wall_lookahead=True, wall_head=True, pair_impulse=True)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.003)
    model.eval()
    return model


def test_tiled_step_matches_global_step():
    n, count, ticks = 90, 1500, 12
    model = make_model(n)
    gen = torch.Generator().manual_seed(4738)
    pos = torch.rand(count, 2, generator=gen) * (n - 3) + 1.5
    vel = (torch.rand(count, 2, generator=gen) - 0.5) * 4.6

    model.dynamics.local_softmax = True
    sim = Sim(model, count, 1, 2.3, 4738)
    sim.positions, sim.velocities, sim.hidden = pos.clone(), vel.clone(), torch.zeros(count, 32)
    tiled = TiledSim(model, strips=3, device=torch.device("cpu"), track_ids=True, slack=8.0)
    tiled.load_flat(pos.clone(), vel.clone())

    for _ in range(ticks):
        sim.step(None)
        tiled.step()

    assert tiled.alive() == sim.positions.shape[0] == count
    state = torch.cat([b[:c] for b, c in zip(tiled.bufs, tiled.counts)])
    order = state[:, -1].long().argsort()
    state = state[order]
    assert torch.allclose(state[:, 0:2], sim.positions, atol=1e-3)
    assert torch.allclose(state[:, 2:4], sim.velocities, atol=1e-3)
