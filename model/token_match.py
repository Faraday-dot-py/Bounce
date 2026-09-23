import torch


def match_tokens_to_state(positions, state_frame):
    """One-time nearest-position match between detected token order
    (model.token_model.TokenModel.init_tokens' output) and the dataset's
    ground-truth ball order (model.token_dataset's state_seq entries) at
    the same frame. Token identity is stable for the rest of the rollout
    (TokenDynamics never reorders tokens, the simulator's ball list is
    never reshuffled), so this match only needs to run once, unlike
    init_tokens' own frame0/frame1 velocity-estimation match which reruns
    every call."""
    if positions.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.long)
    state_pos = torch.stack([state_frame["x"], state_frame["y"]], dim=1)
    dists = torch.cdist(positions, state_pos)
    return dists.argmin(dim=1)
