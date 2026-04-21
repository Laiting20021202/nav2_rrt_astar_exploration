from mapless_nav2.exploration_memory import ExplorationMemory
from mapless_nav2.exploration_types import Pose2D


def test_frontier_blacklist_after_repeated_failures():
    memory = ExplorationMemory(
        {
            "frontier_fail_blacklist_threshold": 2,
            "frontier_blacklist_base_sec": 10.0,
            "frontier_blacklist_max_sec": 60.0,
        }
    )

    now = 0.0
    frontier = "f_1"
    assert memory.frontier_available(frontier, now)

    memory.register_frontier_failure(frontier, now + 1.0)
    assert not memory.frontier_available(frontier, now + 1.1)

    memory.register_frontier_failure(frontier, now + 2.0)
    assert memory.frontier_blacklist_until.get(frontier, 0.0) > now + 2.0


def test_stuck_detector():
    memory = ExplorationMemory({"stuck_window_sec": 5.0, "stuck_radius_m": 0.2})

    t = 0.0
    for _ in range(20):
        memory.update_pose(Pose2D(1.0, 1.0, 0.0), t, known_cell_count=100)
        t += 0.2

    assert memory.is_stuck(t)
