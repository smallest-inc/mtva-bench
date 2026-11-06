"""Percentiles and clustered bootstrap intervals.

A benchmark's calls come in groups that share an agent: its prompt, tools and policies. Calls of
one agent rise and fall together, so treating them as independent would give intervals far too
narrow. The bootstrap resamples agents, then calls within each chosen agent.
"""

import random
from collections.abc import Mapping, Sequence

RESAMPLES = 2000
SEED = 0
LEVEL = 0.95


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0 to 100), interpolating between the closest ranks."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def clustered_bootstrap(
    clusters: Sequence[str],
    columns: Mapping[str, Sequence[float]],
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict[str, tuple[float, float]]:
    """A 95% interval for the mean of each column, where row i belongs to `clusters[i]`.

    Every column is resampled with the same rows, so intervals of quantities computed from
    several columns, and differences between them, stay consistent."""
    members: dict[str, list[int]] = {}
    for row, cluster in enumerate(clusters):
        members.setdefault(cluster, []).append(row)
    groups = list(members.values())
    rng = random.Random(seed)
    means: dict[str, list[float]] = {name: [] for name in columns}
    for _ in range(resamples):
        rows = [
            row
            for group in rng.choices(groups, k=len(groups))
            for row in rng.choices(group, k=len(group))
        ]
        for name, column in columns.items():
            means[name].append(sum(column[row] for row in rows) / len(rows))
    tail = (1 - LEVEL) / 2 * 100
    return {
        name: (percentile(values, tail), percentile(values, 100 - tail))
        for name, values in means.items()
    }
