#!/usr/bin/env python3
from typing import Dict, List

from .exploration_types import FrontierCandidate


class OptionalLearnedRanker:
    """Optional learning-based reranker.

    This baseline keeps the system fully classical and deterministic by default.
    If enabled, it only applies a tiny deterministic bias and never overrides
    safety / reachability filtering.
    """

    def __init__(self, params: Dict[str, float]) -> None:
        self.enabled = bool(params.get("enabled", False))
        self.bias_gain = float(params.get("bias_gain", 0.05))

    def rerank(self, candidates: List[FrontierCandidate]) -> List[FrontierCandidate]:
        if not self.enabled or len(candidates) <= 1:
            return candidates

        # Tiny optional bias toward information gain only; classical score remains dominant.
        max_info = max(1e-3, max(c.information_gain for c in candidates))
        for c in candidates:
            c.score += self.bias_gain * (c.information_gain / max_info)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
