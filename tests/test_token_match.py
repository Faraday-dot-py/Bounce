import torch

from model.token_match import match_tokens_to_state


def test_match_tokens_to_state_matches_nearest_positions():
    # Detection order (positions) is scrambled relative to state order --
    # match must recover state index by nearest position, not by index.
    positions = torch.tensor([[9.0, 9.0], [1.0, 1.0]])
    state = {
        "x": torch.tensor([1.0, 9.0]),
        "y": torch.tensor([1.0, 9.0]),
    }
    match_idx = match_tokens_to_state(positions, state)
    assert match_idx.tolist() == [1, 0]


def test_match_tokens_to_state_identity_when_already_aligned():
    positions = torch.tensor([[2.0, 3.0], [7.0, 5.0]])
    state = {
        "x": torch.tensor([2.0, 7.0]),
        "y": torch.tensor([3.0, 5.0]),
    }
    match_idx = match_tokens_to_state(positions, state)
    assert match_idx.tolist() == [0, 1]


def test_match_tokens_to_state_empty_positions_returns_empty():
    positions = torch.zeros((0, 2))
    state = {"x": torch.tensor([1.0]), "y": torch.tensor([1.0])}
    match_idx = match_tokens_to_state(positions, state)
    assert match_idx.shape == (0,)
