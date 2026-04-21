from mapless_nav2.exploration_memory import ExplorationMemory
from mapless_nav2.exploration_types import FrontierCandidate, Pose2D
from mapless_nav2.frontier_scoring import FrontierScorer


def test_score_prefers_high_info_low_cost():
    scorer = FrontierScorer(
        {
            "w_info": 1.4,
            "w_cost": 1.2,
            "w_visit": 0.7,
            "w_fail": 1.0,
            "w_turn": 0.3,
            "w_goal": 0.4,
            "w_commit": 0.4,
        }
    )
    memory = ExplorationMemory({})
    robot = Pose2D(0.0, 0.0, 0.0)

    a = FrontierCandidate(
        candidate_id="a",
        frontier_id="f1",
        world=(2.0, 0.0),
        cell=(20, 20),
        information_gain=120.0,
        path_cost=12.0,
    )
    b = FrontierCandidate(
        candidate_id="b",
        frontier_id="f2",
        world=(1.0, 0.0),
        cell=(10, 10),
        information_gain=40.0,
        path_cost=30.0,
    )

    ranked = scorer.score_candidates([a, b], robot, None, memory, now_sec=1.0)
    assert ranked[0].candidate_id == "a"
