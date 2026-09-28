"""Coverage-oriented quota union and Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from .base import RetrievalResult


def quota_union(
    rankings: Mapping[str, Sequence[RetrievalResult]], quotas: Mapping[str, int], top_k: int = 50
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for name, ranking in rankings.items():
        limit = int(quotas.get(name, top_k))
        if limit <= 0:
            continue
        taken = 0
        for result in ranking:
            if result.item_id not in seen:
                seen.add(result.item_id)
                output.append(result.item_id)
                taken += 1
                if len(output) == top_k or taken == limit:
                    break
        if len(output) == top_k:
            return output
    # Остаток квоты заполняем по очереди из всех списков.
    depth = 0
    while len(output) < top_k:
        added = False
        for ranking in rankings.values():
            if depth < len(ranking) and ranking[depth].item_id not in seen:
                seen.add(ranking[depth].item_id)
                output.append(ranking[depth].item_id)
                added = True
                if len(output) == top_k:
                    break
        if not added and all(depth >= len(ranking) for ranking in rankings.values()):
            break
        depth += 1
    return output


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievalResult]], top_k: int = 50, rrf_k: int = 60
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for ranking in rankings.values():
        seen_in_ranking: set[str] = set()
        for rank, result in enumerate(ranking, start=1):
            if result.item_id in seen_in_ranking:
                continue
            seen_in_ranking.add(result.item_id)
            scores[result.item_id] += 1.0 / (rrf_k + rank)
            best_rank[result.item_id] = min(best_rank.get(result.item_id, rank), rank)
    ordered = sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))
    return ordered[:top_k]
