import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import bounce
from model.dataset import make_scenario_uniform
from model.net import BounceNextFrameModel


def load_model(checkpoint_path):
    model = BounceNextFrameModel(channels=64, depth=7)
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


def rollout(model, g0, num_steps):
    g_pred = torch.from_numpy(g0.transpose(2, 0, 1)).unsqueeze(0)
    frames = [g0]
    with torch.no_grad():
        for _ in range(num_steps):
            g_pred = model(g_pred)
            frames.append(g_pred[0].permute(1, 2, 0).numpy())
    return frames


if __name__ == "__main__":
    n = 50
    num_balls = 150
    seed = 4738
    num_steps = 20

    stage1 = load_model("checkpoints/stage1.pt")
    stage2 = load_model("checkpoints/stage2.pt")

    truth = simulate_ground_truth(n, num_balls, seed, num_steps)
    pred1 = rollout(stage1, truth[0], num_steps)
    pred2 = rollout(stage2, truth[0], num_steps)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    titles = ["ground truth", "stage1 (rollout)", "stage2 (rollout)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    ims = [
        axes[0].imshow(truth[0][:, :, bounce.PROB], vmin=0, vmax=1, cmap="inferno"),
        axes[1].imshow(pred1[0][:, :, bounce.PROB], vmin=0, vmax=1, cmap="inferno"),
        axes[2].imshow(pred2[0][:, :, bounce.PROB], vmin=0, vmax=1, cmap="inferno"),
    ]
    step_text = fig.suptitle("step 0")

    def update(i):
        ims[0].set_data(truth[i][:, :, bounce.PROB])
        ims[1].set_data(pred1[i][:, :, bounce.PROB])
        ims[2].set_data(pred2[i][:, :, bounce.PROB])
        step_text.set_text(f"step {i}")
        return ims + [step_text]

    anim = animation.FuncAnimation(fig, update, frames=num_steps + 1, interval=300, blit=False)
    anim.save("videos/stage2_rollout_comparison.mp4", writer="ffmpeg", fps=3, dpi=120)
    print("wrote videos/stage2_rollout_comparison.mp4")
