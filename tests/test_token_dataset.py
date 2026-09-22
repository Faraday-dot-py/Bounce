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


def test_token_dataset_frames_are_independent_snapshots_not_aliased():
    # Regression test: bounce.step() mutates ball dicts in place, so
    # generate_token_sequence must copy each frame's state independently
    # (see model.token_dataset._copy_states) -- appending references to
    # the live `balls` list instead would make every frame in state_seq
    # collapse to the final post-rollout state.
    dataset = BounceTokenSequenceDataset(
        num_samples=1, n=20, ball_range=(2, 2), seed=4738, horizon=5
    )
    _, state_seq = dataset[0]
    first_frame_x = state_seq[0]["x"].clone()
    last_frame_x = state_seq[-1]["x"].clone()
    assert not torch.allclose(first_frame_x, last_frame_x)
    # each frame's tensors must be distinct objects, not views into a
    # shared mutable structure
    assert state_seq[0]["x"] is not state_seq[-1]["x"]
