import torch

from model.net import BounceNextFrameModel
from model.evaluate import rollout_divergence


def load_model(checkpoint_path):
    model = BounceNextFrameModel(channels=64, depth=7)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def occupied_cell_count(model, n, num_balls, seed, num_steps, radius=0.75, **kwargs):
    import random
    import numpy as np
    import bounce
    from model.dataset import make_scenario_uniform

    rng = random.Random(seed)
    balls = make_scenario_uniform(num_balls, n, kwargs.get("vy", 2.3), rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    g_true = np.array(G, dtype=np.float32)
    g_pred = torch.from_numpy(g_true.transpose(2, 0, 1)).unsqueeze(0)

    counts = []
    with torch.no_grad():
        for _ in range(num_steps):
            g_pred = model(g_pred)
            occupied = (g_pred[0, 0] > 1e-6).sum().item()
            counts.append(occupied)
    return counts


if __name__ == "__main__":
    n = 50
    num_balls = 150
    seed = 4738
    num_steps = 20

    stage1 = load_model("checkpoints/stage1.pt")
    stage2 = load_model("checkpoints/stage2.pt")

    div1 = rollout_divergence(stage1, n=n, num_balls=num_balls, seed=seed, num_steps=num_steps)
    div2 = rollout_divergence(stage2, n=n, num_balls=num_balls, seed=seed, num_steps=num_steps)

    occ1 = occupied_cell_count(stage1, n=n, num_balls=num_balls, seed=seed, num_steps=num_steps)
    occ2 = occupied_cell_count(stage2, n=n, num_balls=num_balls, seed=seed, num_steps=num_steps)

    print(f"true occupied cells (approx, ball count): {num_balls}")
    print(f"{'step':>4} {'stage1_mse':>12} {'stage2_mse':>12} {'stage1_occ':>11} {'stage2_occ':>11}")
    for i in range(num_steps):
        print(f"{i+1:>4} {div1[i]:>12.4f} {div2[i]:>12.4f} {occ1[i]:>11d} {occ2[i]:>11d}")
