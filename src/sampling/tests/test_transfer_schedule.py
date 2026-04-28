import torch

from sampling.transfer_schedule import (
    compute_move_grid,
    get_num_transfer_tokens_move,
    get_num_transfer_tokens_uniform,
)


def test_transfer_tokens_sum_matches_mask_num():
    mask_index = torch.tensor(
        [
            [True, False, True, True, False],
            [False, False, False, True, True],
        ],
        dtype=torch.bool,
    )
    steps = 4
    _, move_grid, _ = compute_move_grid(steps)
    counts = get_num_transfer_tokens_move(mask_index, steps, move_grid)
    assert torch.all(counts.sum(dim=1) == mask_index.sum(dim=1))


def test_transfer_tokens_nonnegative():
    mask_index = torch.zeros((3, 7), dtype=torch.bool)
    mask_index[0, :3] = True
    mask_index[1, :5] = True
    steps = 5
    move_grid = torch.tensor([1.0, 0.8, 0.5, 0.3, 0.1, 0.0])
    counts = get_num_transfer_tokens_move(mask_index, steps, move_grid)
    assert torch.all(counts >= 0)


def test_uniform_equivalence_for_constant_delta():
    mask_index = torch.tensor(
        [
            [True, True, True, False],
            [True, False, False, False],
        ],
        dtype=torch.bool,
    )
    steps = 3
    _, move_grid, _ = compute_move_grid(steps)
    move_counts = get_num_transfer_tokens_move(mask_index, steps, move_grid)
    uniform_counts = get_num_transfer_tokens_uniform(mask_index, steps)
    assert torch.equal(move_counts, uniform_counts)


def test_mask_num_zero_edge_case():
    mask_index = torch.zeros((2, 6), dtype=torch.bool)
    steps = 4
    _, move_grid, _ = compute_move_grid(steps)
    counts = get_num_transfer_tokens_move(mask_index, steps, move_grid)
    assert torch.all(counts == 0)
