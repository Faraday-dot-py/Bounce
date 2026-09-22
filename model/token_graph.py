import torch


def build_radius_graph(positions, neighbor_radius):
    """Undirected edges (as directed pairs, both directions) between
    tokens within `neighbor_radius`, in absolute position space. Has no
    notion of tiles -- passing positions from two adjacent tiles together
    produces edges across the tile boundary exactly as if the tokens had
    come from one untiled scene. Building the graph from one tile's
    tokens in isolation loses any cross-boundary edge, so callers must
    always pass the full combined position set (design spec's tiling
    requirement)."""
    n = positions.shape[0]
    if n < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=positions.device)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)
    within = dist <= neighbor_radius
    within.fill_diagonal_(False)
    src, dst = torch.nonzero(within, as_tuple=True)
    return torch.stack([src, dst], dim=0)
