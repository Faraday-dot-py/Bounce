import random

from model.token_train import curriculum_advance, curriculum_steps


def test_advance_needs_two_chunks_unless_max():
    assert not curriculum_advance([1.0], tol=0.03, max_chunks=4)
    assert curriculum_advance([1.0], tol=0.03, max_chunks=1)


def test_advance_on_plateau_not_on_progress():
    assert not curriculum_advance([1.0, 0.8], tol=0.03, max_chunks=4)
    assert curriculum_advance([1.0, 0.99], tol=0.03, max_chunks=4)
    assert curriculum_advance([1.0, 1.1], tol=0.03, max_chunks=4)


def test_advance_forced_at_max_chunks():
    assert curriculum_advance([1.0, 0.8, 0.6, 0.4], tol=0.03, max_chunks=4)


def test_steps_stage_one_never_replays():
    rng = random.Random(4738)
    assert all(curriculum_steps(1, 1.0, rng) == 1 for _ in range(20))


def test_steps_replay_within_range_and_mixes():
    rng = random.Random(4738)
    draws = [curriculum_steps(10, 0.5, rng) for _ in range(500)]
    assert all(1 <= d <= 10 for d in draws)
    assert 200 < sum(d == 10 for d in draws) < 400 and min(draws) < 10
