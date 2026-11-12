"""The MTVA Index: call quality, speed and cost in one number from 0 to 100.

A voice agent needs all three at once: a cheap model that handles calls badly is no use, and
neither is a good one that keeps the caller waiting. So the index is a weighted geometric mean,
in which a poor part cannot be made up by a strong one, the way an average would allow.

Speed and cost are scored on log scales between fixed ends, because a delay or a price is felt
as a ratio: 200 ms to 400 ms matters more than 2.0 s to 2.2 s. The ends are fixed numbers, not
the best and worst model, so a new model never changes another's index.
"""

import math

WEIGHTS = {"quality": 0.5, "speed": 0.3, "cost": 0.2}
# (value worth 100 points, value worth 0 points)
FIRST_TOKEN_MS = (200.0, 10_000.0)
USD_PER_CALL = (0.001, 1.0)


def scaled(value: float, best: float, worst: float) -> float:
    """0 to 100 on a log scale: 100 at `best` or better, 0 at `worst` or worse."""
    share = math.log(worst / value) / math.log(worst / best)
    return 100 * min(max(share, 0.0), 1.0)


def mtva_index(mtva_score: float, first_token_ms: float, usd_per_call: float) -> float:
    parts = {
        "quality": mtva_score,
        "speed": scaled(first_token_ms, *FIRST_TOKEN_MS),
        "cost": scaled(usd_per_call, *USD_PER_CALL),
    }
    return math.prod(parts[name] ** weight for name, weight in WEIGHTS.items())
