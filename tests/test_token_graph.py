import torch

from model.token_graph import build_radius_graph


def test_build_radius_graph_connects_only_within_threshold():
    positions = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],   # within 2.0 of token 0
        [10.0, 0.0],  # far from everything
    ])
    edges = build_radius_graph(positions, neighbor_radius=2.0)
    pairs = set(zip(edges[0].tolist(), edges[1].tolist()))
    assert pairs == {(0, 1), (1, 0)}


def test_build_radius_graph_no_self_loops():
    positions = torch.tensor([[0.0, 0.0]])
    edges = build_radius_graph(positions, neighbor_radius=5.0)
    assert edges.shape == (2, 0)


def test_build_radius_graph_crosses_tile_boundary_when_positions_are_absolute():
    # "Tile A" token near x=9.5, "tile B" token near x=10.5: adjacent
    # tiles' tokens must connect when passed together in absolute
    # coordinates (design spec's tiling requirement).
    tile_a = torch.tensor([[9.5, 5.0]])
    tile_b = torch.tensor([[10.5, 5.0]])
    combined = torch.cat([tile_a, tile_b], dim=0)

    edges_combined = build_radius_graph(combined, neighbor_radius=2.0)
    assert edges_combined.shape[1] == 2  # (0,1) and (1,0)

    edges_tile_a_alone = build_radius_graph(tile_a, neighbor_radius=2.0)
    assert edges_tile_a_alone.shape[1] == 0  # the cross-boundary neighbor is invisible in isolation


def test_cell_graph_matches_dense_graph():
    from model.token_graph import build_radius_graph_cells
    torch.manual_seed(4738)
    for count in (2, 5, 60, 300):
        positions = torch.rand(count, 2) * 30.0 - 5.0
        dense = build_radius_graph(positions, 4.0)
        cells = build_radius_graph_cells(positions, 4.0)
        assert set(map(tuple, dense.t().tolist())) == set(map(tuple, cells.t().tolist()))
        assert dense.shape == cells.shape
