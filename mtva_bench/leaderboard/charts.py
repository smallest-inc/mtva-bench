"""The README's charts as SVG cards.

The SVG is written by hand, with no plotting library, so every chart shares one typography,
spacing and palette, stays crisp at any size and diffs cleanly in git. Each chart is a light card
with its own background, so it reads the same on GitHub's light and dark pages. One
organization's models are drawn in the accent and the rest in warm gray, so the reader sees them
against the field. Headlines that make a claim are computed from the data, and fall back to a
plain title when the data no longer supports the claim. Text is never drawn in a data color.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html import escape
from itertools import pairwise

from mtva_bench.leaderboard.build import Entry
from mtva_bench.run.arena import ArenaSummary

HIGHLIGHT = "Smallest AI"
WORDMARK = "smallest.ai"
WIDTH = 1200
MARGIN = 64
FONT = (
    "-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', 'Noto Sans', Helvetica, Arial, "
    "sans-serif"
)

CARD = "#fbfbf9"
BORDER = "#ebe9e4"
TILE = "#ffffff"
INK = "#111111"
SECONDARY = "#55534e"
MUTED = "#8a877f"
GRID = "#ebe9e4"
AXIS = "#d3d0c9"
ACCENT = "#2a78d6"
ACCENT_SOFT = "#e3edf9"
FIELD = "#a29e96"
BAR = "#d9d6cf"
RAMP = ("#f4f2ee", "#dce7f5", "#b5cdef", "#86b0e6", "#5592de", "#2a78d6")

CAPABILITIES = {
    "task_completion": (
        "Task completion",
        "Did the caller get what they called for?",
        "The judge reads the whole call and rates its outcome: the caller's goal reached, an "
        "accepted alternative, partly reached, not reached, or a wrong result. A goal reached "
        "without the caller hearing everything they needed scores one level lower.",
    ),
    "tool_use": (
        "Tool calling",
        "Did the agent call the right tools, with the right arguments, in the right order?",
        "Each scenario's key actions are checked against the agent's tool calls, in code and "
        "then by the judge: the right tool, the right arguments, the right order. A missed or "
        "wrong action costs more the more it matters to the business.",
    ),
    "instruction_following": (
        "Instruction following",
        "Did the agent keep to the rules of its system prompt?",
        "The judge checks each scenario's focus rules from the agent's system prompt, and any "
        "other rule the agent broke, and rates each problem minor, major or critical.",
    ),
    "faithfulness": (
        "Faithfulness",
        "Was everything the agent said true?",
        "The judge flags anything the agent said that its tools, its prompt or the caller did "
        "not support, such as an invented fact or an action claimed but never run, and rates "
        "each one minor, major or critical.",
    ),
}
SCORING = (
    "Each capability is scored 1 to 5 per call from the judge's verdicts and mapped to 0 to "
    "100. Intervals are 95%, bootstrapped by agent. Every model runs at its lowest reasoning "
    "setting."
)

# Advance widths in em of a Helvetica-like sans, to lay out text without a font engine.
NARROW = set("fijlrtI.,:;'|() ")
WIDE = set("mwMW%@")


def text_width(text: str, size: float, bold: bool = False) -> float:
    def advance(char: str) -> float:
        if char in NARROW:
            return 0.29
        if char in WIDE:
            return 0.86
        if char.isupper() or char.isdigit():
            return 0.64 if char.isupper() else 0.57
        return 0.54

    return sum(advance(char) for char in text) * size * (1.07 if bold else 1.0)


def wrap(text: str, size: float, width: float) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and text_width(candidate, size) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return [*lines, line] if line else lines


def highlighted(entry: Entry) -> bool:
    return entry.organization == HIGHLIGHT


def lead(entries: Sequence[Entry]) -> Entry | None:
    return next((entry for entry in entries if highlighted(entry)), None)


class Card:
    """One chart: a kicker, a headline, a subtitle, the body the caller draws, then a method
    note and the wordmark. `y` is where the body may start."""

    def __init__(self, kicker: str, title: str, subtitle: str, description: str):
        self.parts: list[str] = []
        self.title = title
        self.description = description
        self.add(f'<circle cx="{MARGIN + 4}" cy="67" r="4.5" fill="{ACCENT}"/>')
        self.text(MARGIN + 16, 72, kicker.upper(), MUTED, 13, weight=600, spacing=1.6)
        lines = wrap(title, 36 * 1.04, WIDTH - 2 * MARGIN)
        for index, line in enumerate(lines):
            self.text(MARGIN, 124 + 44 * index, line, INK, 36, weight=700, spacing=-0.9)
        below = 124 + 44 * (len(lines) - 1)
        self.text(MARGIN, below + 36, subtitle, SECONDARY, 17)
        self.y = below + 80.0

    def add(self, element: str) -> None:
        self.parts.append(element)

    def text(
        self,
        x: float,
        y: float,
        content: str,
        fill: str,
        size: float = 14,
        anchor: str = "start",
        weight: int = 400,
        spacing: float = 0,
        numeric: bool = False,
    ) -> None:
        extra = f' letter-spacing="{spacing}"' if spacing else ""
        if numeric:
            extra += ' font-variant-numeric="tabular-nums"'
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{extra}>{escape(content)}</text>'
        )

    def runs(
        self,
        x: float,
        y: float,
        parts: Sequence[tuple[str, str, float, int, float]],
        anchor: str = "start",
    ) -> None:
        """Pieces of text laid out one after another by the renderer, each as (content, fill,
        size, weight, gap before it), so no width has to be guessed."""
        spans = "".join(
            f'<tspan fill="{fill}" font-size="{size}" font-weight="{weight}"'
            + (f' dx="{gap}"' if gap else "")
            + f">{escape(content)}</tspan>"
            for content, fill, size, weight, gap in parts
        )
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{spans}</text>')

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str,
        width: float = 1,
        opacity: float = 1,
    ) -> None:
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}" stroke-linecap="round"/>'
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: str,
        radius: float = 0,
        stroke: str | None = None,
    ) -> None:
        border = f' stroke="{stroke}"' if stroke else ""
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0):.1f}" '
            f'height="{height:.1f}" rx="{radius}" fill="{fill}"{border}/>'
        )

    def bar(self, x0: float, x1: float, center: float, thickness: float, fill: str) -> None:
        """A bar from x0 to x1, rounded at its far end and square at the baseline."""
        left, right = sorted((x0, x1))
        radius = min(5.0, (right - left) / 2, thickness / 2)
        y, bottom = center - thickness / 2, center + thickness / 2
        if x1 >= x0:
            path = (
                f"M{left:.1f},{y:.1f} H{right - radius:.1f} Q{right:.1f},{y:.1f} {right:.1f},"
                f"{y + radius:.1f} V{bottom - radius:.1f} Q{right:.1f},{bottom:.1f} "
                f"{right - radius:.1f},{bottom:.1f} H{left:.1f} Z"
            )
        else:
            path = (
                f"M{right:.1f},{y:.1f} H{left + radius:.1f} Q{left:.1f},{y:.1f} {left:.1f},"
                f"{y + radius:.1f} V{bottom - radius:.1f} Q{left:.1f},{bottom:.1f} "
                f"{left + radius:.1f},{bottom:.1f} H{right:.1f} Z"
            )
        self.add(f'<path d="{path}" fill="{fill}"/>')

    def render(self, method: str) -> str:
        lines = wrap(f"Method. {method}", 13.5, WIDTH - 2 * MARGIN - 160)
        lines[0] = lines[0].removeprefix("Method. ")
        y = self.y + 36
        for index, line in enumerate(lines):
            if index == 0:
                self.runs(
                    MARGIN, y, [("Method.", SECONDARY, 13.5, 700, 0), (line, MUTED, 13.5, 400, 5)]
                )
            else:
                self.text(MARGIN, y, line, MUTED, 13.5)
            y += 21
        self.text(WIDTH - MARGIN, y - 21, WORDMARK, INK, 16, anchor="end", weight=700)
        height = y + 36
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height:.0f}" '
            f'viewBox="0 0 {WIDTH} {height:.0f}" font-family="{FONT}" role="img">\n'
            f"<title>{escape(self.title)}</title>\n<desc>{escape(self.description)}</desc>\n"
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1:.0f}" rx="18" '
            f'fill="{CARD}" stroke="{BORDER}"/>\n{body}\n</svg>\n'
        )


def judge_label(model: str, entries: Sequence[Entry]) -> str:
    return next((e.label for e in entries if e.summary.run == model), model)


# --- arena wins -----------------------------------------------------------------------------


def preferred(pairs: Sequence[ArenaSummary]) -> list[ArenaSummary]:
    """The pairs the first run won beyond doubt: its whole interval above half."""
    return [p for p in pairs if p.scores is not None and p.scores.wins.ci[0] > 50]


# --- score against latency -------------------------------------------------------------------

LATENCY_TICKS = (100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000)


def latency_label(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:g} s"


def frontier(entries: Sequence[Entry]) -> list[Entry]:
    """The runs no other run beats on both score and latency, fastest first."""
    best = -math.inf
    found = []
    for entry in sorted(
        entries, key=lambda e: (e.latency.ttft_warm_p50, -e.scores.mtva_score.value)
    ):
        if entry.scores.mtva_score.value > best:
            found.append(entry)
            best = entry.scores.mtva_score.value
    return found


Box = tuple[float, float, float, float]


def overlap(a: Box, b: Box) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(width, 0) * max(height, 0)


@dataclass
class Label:
    x: float
    y: float
    anchor: str
    box: Box
    leader: bool


@dataclass
class Obstacles:
    """What a label should not cover, from worst to least bad to cover."""

    points: list[Box]
    intervals: list[Box]
    lines: list[Box]


RINGS = ((12.0, 0.0), (26.0, 30.0), (44.0, 70.0))


def place_label(
    px: float,
    py: float,
    width: float,
    size: float,
    taken: list[Box],
    avoid: Obstacles,
    bounds: Box,
    rings: Sequence[tuple[float, float]] = RINGS,
) -> Label:
    """The spot around a point that covers the least of what is already drawn, preferring spots
    close to it; a spot away from the point gets a leader line."""
    height = size + 2
    best: tuple[float, Label] | None = None
    for ring, (gap, penalty) in enumerate(rings):
        for step in range(12):
            angle = math.radians(step * 30)
            dx, dy = math.cos(angle) * gap, -math.sin(angle) * gap * 0.8
            anchor = "start" if dx > 3 else "end" if dx < -3 else "middle"
            x, y = px + dx, py + dy
            left = {"start": x, "end": x - width, "middle": x - width / 2}[anchor]
            box = (left - 2, y - height / 2, left + width + 2, y + height / 2)
            outside = (
                max(bounds[0] - box[0], 0)
                + max(box[2] - bounds[2], 0)
                + max(bounds[1] - box[1], 0)
                + max(box[3] - bounds[3], 0)
            )
            cost = (
                sum(overlap(box, other) for other in taken) * 25
                + sum(overlap(box, other) for other in avoid.points) * 30
                + sum(overlap(box, other) for other in avoid.intervals) * 6
                + sum(overlap(box, other) for other in avoid.lines) * 3
                + outside * 400
                + penalty
                + (0 if step in (0, 6) else 4)
            )
            if best is None or cost < best[0]:
                best = (cost, Label(x, y + size * 0.35, anchor, box, ring > 0))
    assert best is not None
    return best[1]


def leader_end(label: Label, px: float, py: float) -> tuple[float, float]:
    """The point of the label's box nearest the point it names."""
    x0, y0, x1, y1 = label.box
    return min(max(px, x0), x1), min(max(py, y0), y1)


def crowding(point: tuple[float, float], others: list[tuple[float, float]]) -> int:
    return sum(math.dist(point, other) < 60 for other in others)


def score_vs_latency(entries: Sequence[Entry]) -> str:
    card = Card(
        f"MTVA-Bench · {len(entries)} models",
        "Quality against speed",
        "MTVA Score against median time to first token. Up and to the left is better.",
        "Each model's MTVA Score against its warm time to first token, with the Pareto frontier.",
    )
    left, right, top = MARGIN + 44, WIDTH - MARGIN, card.y + 8
    bottom = top + 470
    latencies = [e.latency.ttft_warm_p50 for e in entries]
    low_x = max(t for t in LATENCY_TICKS if t <= min(latencies) * 0.8)
    high_x = min(t for t in LATENCY_TICKS if t >= max(latencies) * 1.25)
    low_y = math.floor((min(e.scores.mtva_score.ci[0] for e in entries) - 2) / 10) * 10
    high_y = min(100, math.ceil((max(e.scores.mtva_score.ci[1] for e in entries) + 2) / 5) * 5)

    def x_at(ms: float) -> float:
        span = math.log(high_x) - math.log(low_x)
        return left + (math.log(ms) - math.log(low_x)) / span * (right - left)

    def y_at(score: float) -> float:
        return bottom - (score - low_y) / (high_y - low_y) * (bottom - top)

    for tick in range(low_y, high_y + 1, 10):
        y = y_at(tick)
        card.line(left, y, right, y, GRID)
        card.text(left - 12, y + 5, str(tick), MUTED, 13, anchor="end", numeric=True)
    for tick in (t for t in LATENCY_TICKS if low_x <= t <= high_x):
        x = x_at(tick)
        card.line(x, top, x, bottom, GRID)
        card.text(x, bottom + 24, latency_label(tick), MUTED, 13, anchor="middle")
    card.line(left, bottom, right, bottom, AXIS)
    card.text(
        (left + right) / 2,
        bottom + 52,
        "Time to first token, median of warm turns (log scale)",
        SECONDARY,
        14,
        anchor="middle",
    )
    card.add(
        f'<text transform="translate({MARGIN - 2} {(top + bottom) / 2:.1f}) rotate(-90)" '
        f'fill="{SECONDARY}" font-size="14" text-anchor="middle">MTVA Score</text>'
    )

    points = [(e, x_at(e.latency.ttft_warm_p50), y_at(e.scores.mtva_score.value)) for e in entries]
    edge = [
        (x_at(e.latency.ttft_warm_p50), y_at(e.scores.mtva_score.value)) for e in frontier(entries)
    ]
    if len(edge) > 1:
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(edge))
        card.add(
            f'<path d="{path}" fill="none" stroke="{FIELD}" stroke-width="1.5" '
            f'stroke-opacity="0.6" stroke-linejoin="round"/>'
        )
    avoid = Obstacles(points=[], intervals=[], lines=[])
    for entry, x, y in points:
        color = ACCENT if highlighted(entry) else FIELD
        low, high = entry.scores.mtva_score.ci
        card.line(x, y_at(low), x, y_at(high), color, width=2, opacity=0.45)
        reach = 17 if highlighted(entry) else 8
        avoid.points.append((x - reach, y - reach, x + reach, y + reach))
        avoid.intervals.append((x - 1.5, y_at(high), x + 1.5, y_at(low)))
    for (ax, ay), (bx, by) in pairwise(edge):
        steps = max(int(math.dist((ax, ay), (bx, by)) / 6), 1)
        for i in range(steps + 1):
            x, y = ax + (bx - ax) * i / steps, ay + (by - ay) * i / steps
            avoid.lines.append((x - 1.5, y - 1.5, x + 1.5, y + 1.5))
    for entry, x, y in points:
        if highlighted(entry):
            card.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="{ACCENT}" opacity="0.14"/>')
            card.add(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7.5" fill="{ACCENT}" stroke="{CARD}" '
                f'stroke-width="2.5"/>'
            )
        else:
            card.add(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{FIELD}" stroke="{CARD}" '
                f'stroke-width="2"/>'
            )

    taken: list[Box] = []
    if len(edge) > 1:
        (ax, ay), (bx, by) = max(pairwise(edge), key=lambda pair: math.dist(pair[0], pair[1]))
        angle = math.degrees(math.atan2(by - ay, bx - ax))
        mx, my = (ax + bx) / 2, (ay + by) / 2 - 8
        card.add(
            f'<text transform="translate({mx:.1f} {my:.1f}) rotate({angle:.1f})" '
            f'fill="{MUTED}" font-size="13" text-anchor="middle">Pareto frontier</text>'
        )
        width = text_width("Pareto frontier", 13)
        taken.append((mx - width / 2, my - 13, mx + width / 2, my + 2))
    centers = [(x, y) for _, x, y in points]
    order = sorted(
        points,
        key=lambda p: (not highlighted(p[0]), -crowding((p[1], p[2]), centers), p[2]),
    )
    labels = []
    for entry, x, y in order:
        bold = highlighted(entry)
        size = 16.0 if bold else 14.0
        # The highlighted point has a halo, so its label starts further out.
        rings = tuple((gap + 8, penalty) for gap, penalty in RINGS) if bold else RINGS
        width = text_width(entry.label, size, bold)
        label = place_label(x, y, width, size, taken, avoid, (left, top, right, bottom), rings)
        taken.append(label.box)
        labels.append((entry, x, y, label))
    for _, x, y, label in labels:
        if label.leader:
            end_x, end_y = leader_end(label, x, y)
            distance = math.dist((x, y), (end_x, end_y))
            if distance > 10:
                start_x = x + (end_x - x) / distance * 9
                start_y = y + (end_y - y) / distance * 9
                card.line(start_x, start_y, end_x, end_y, MUTED, width=1, opacity=0.6)
    for entry, _, _, label in labels:
        bold = highlighted(entry)
        card.text(
            label.x,
            label.y,
            entry.label,
            INK if bold else SECONDARY,
            16.0 if bold else 14.0,
            anchor=label.anchor,
            weight=700 if bold else 400,
        )
    card.y = bottom + 60
    return card.render(
        "Every model ran the same calls at its lowest reasoning setting. Latency is the median "
        "time to first token on warm turns, after the opening one, measured from India over each "
        "provider's public API, so it includes network distance. Vertical lines are 95% "
        "intervals of the MTVA Score. The frontier joins the models no other model beats on both "
        "score and speed."
    )


# --- one score, ranked ------------------------------------------------------------------------


def ranked_bars(
    entries: Sequence[Entry], key: str, kicker: str, title: str, subtitle: str, method: str
) -> str:
    """Models ranked on one score, each with its 95% interval; `key` is a capability or
    "mtva_score"."""

    def interval(entry: Entry):
        scores = entry.scores
        return scores.mtva_score if key == "mtva_score" else scores.capabilities[key]

    ranked = sorted(entries, key=lambda e: (-interval(e).value, e.label))
    card = Card(kicker, title, subtitle, f"{title} {subtitle}")
    row, thickness = 36, 20
    top = card.y + 8
    label_width = max(text_width(e.label, 15, True) for e in ranked)
    x0 = MARGIN + label_width + 24
    x1 = WIDTH - MARGIN - 70
    bottom = top + row * len(ranked)

    def x_at(value: float) -> float:
        return x0 + value / 100 * (x1 - x0)

    for tick in (0, 25, 50, 75, 100):
        x = x_at(tick)
        card.line(x, top - 6, x, bottom + 4, GRID)
        card.text(x, bottom + 26, str(tick), MUTED, 13, anchor="middle", numeric=True)
    for index, entry in enumerate(ranked):
        center = top + row * index + row / 2
        bold = highlighted(entry)
        if bold:
            card.rect(
                MARGIN - 12,
                center - row / 2 + 2,
                WIDTH - 2 * MARGIN + 24,
                row - 4,
                ACCENT_SOFT,
                radius=8,
            )
        score = interval(entry)
        card.bar(x0, x_at(score.value), center, thickness, ACCENT if bold else BAR)
        low, high = score.ci
        card.line(x_at(low), center, x_at(high), center, INK, width=1.5, opacity=0.45)
        card.text(
            MARGIN + label_width,
            center + 5,
            entry.label,
            INK if bold else SECONDARY,
            15,
            anchor="end",
            weight=700 if bold else 400,
        )
        card.text(
            x_at(max(score.value, high)) + 10,
            center + 5,
            f"{score.value:.1f}",
            INK if bold else SECONDARY,
            15,
            weight=700 if bold else 400,
            numeric=True,
        )
    card.line(x0, top - 6, x0, bottom + 4, AXIS)
    card.y = bottom + 36
    return card.render(method)


# --- score by language ------------------------------------------------------------------------


def mix(low: str, high: str, share: float) -> str:
    a = [int(low[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(high[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * share):02x}" for x, y in zip(a, b, strict=True))


def ramp_color(share: float) -> str:
    position = min(max(share, 0.0), 1.0) * (len(RAMP) - 1)
    index = min(int(position), len(RAMP) - 2)
    return mix(RAMP[index], RAMP[index + 1], position - index)


def luminance(color: str) -> float:
    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(int(color[i : i + 2], 16)) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    light, dark = sorted((luminance(a), luminance(b)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def language_order(names: Iterable[str]) -> list[str]:
    """English first, as the language most calls are in, then the rest alphabetically."""
    return sorted(set(names), key=lambda language: (language != "English", language))


def by_language(entries: Sequence[Entry]) -> str:
    cuts = [e.scores.slices["language"] for e in entries]
    names = language_order(language for cut in cuts for language in cut)
    calls = {name: max(cut[name].calls for cut in cuts if name in cut) for name in names}
    values = [cut[name].mtva_score for cut in cuts for name in cut]
    values += [e.scores.mtva_score.value for e in entries]
    low = math.floor(min(values) / 5) * 5
    high = math.ceil(max(values) / 5) * 5
    card = Card(
        "MTVA-Bench · by caller language",
        f"Every model, in all {len(names)} languages",
        "MTVA Score by the language of the call. Models ranked by overall score.",
        "Each model's MTVA Score in each language.",
    )
    label_width = max(text_width(e.label, 15, True) for e in entries)
    x0 = MARGIN + label_width + 20
    gap, split = 3, 14
    cell_w = (WIDTH - MARGIN - x0 - split) / (len(names) + 1)
    cell_h = 36
    top = card.y + 44
    columns = [("All", None, x0)] + [
        (name, calls[name], x0 + split + cell_w * (i + 1)) for i, name in enumerate(names)
    ]
    for name, count, x in columns:
        card.text(x + cell_w / 2, top - 24, name, INK, 14, anchor="middle", weight=700)
        if count is not None:
            card.text(x + cell_w / 2, top - 8, f"{count} calls", MUTED, 12, anchor="middle")
    for row, entry in enumerate(entries):
        y = top + cell_h * row
        bold = highlighted(entry)
        if bold:
            card.rect(MARGIN - 12, y + 1, label_width + 26, cell_h - 2, ACCENT_SOFT, radius=8)
        card.text(
            MARGIN + label_width,
            y + cell_h / 2 + 5,
            entry.label,
            INK if bold else SECONDARY,
            15,
            anchor="end",
            weight=700 if bold else 400,
        )
        cut = entry.scores.slices["language"]
        cells = [(columns[0][2], entry.scores.mtva_score.value)] + [
            (x, cut[name].mtva_score if name in cut else None) for name, _, x in columns[1:]
        ]
        for x, value in cells:
            if value is None:
                continue
            fill = ramp_color((value - low) / (high - low))
            ink = max((TILE, INK), key=lambda text: contrast(text, fill))
            card.rect(x + gap / 2, y + gap / 2, cell_w - gap, cell_h - gap, fill, radius=5)
            card.text(
                x + cell_w / 2,
                y + cell_h / 2 + 5,
                f"{value:.0f}",
                ink,
                14,
                anchor="middle",
                weight=700 if bold else 400,
                numeric=True,
            )
    legend_y = top + cell_h * len(entries) + 26
    legend_x = WIDTH - MARGIN - 220
    for i in range(44):
        card.rect(legend_x + i * 5, legend_y, 5.4, 10, ramp_color(i / 43))
    card.text(legend_x - 8, legend_y + 10, str(low), MUTED, 13, anchor="end", numeric=True)
    card.text(legend_x + 228, legend_y + 10, str(high), MUTED, 13, numeric=True)
    card.y = legend_y + 14
    return card.render(
        "Each cell is the mean MTVA Score over that language's calls; the number of calls is "
        "under each name. The agent speaks the caller's language, and the judge holds it to the "
        "language and register its system prompt sets."
    )


# --- sounds human: the arena ------------------------------------------------------------------


def sounds_human(entries: Sequence[Entry], pairs: Sequence[ArenaSummary]) -> str | None:
    """The focus model against each opponent: the share of calls the blind judge gave each."""
    focus = lead(entries)
    if focus is None:
        return None
    mine = sorted(
        (p for p in pairs if p.a.run == focus.summary.run and p.scores is not None),
        key=lambda p: -p.scores.wins.value,  # type: ignore[union-attr]
    )
    if not mine:
        return None
    won = preferred(mine)
    title = (
        f"A blind judge found {focus.label} more human than all {len(mine)} models it faced"
        if len(won) == len(mine)
        else "Which agent sounds more human"
    )
    calls = max(p.coverage.judged for p in mine)
    card = Card(
        "Blind head-to-head · model names hidden",
        title,
        f"Same scenario, two calls, labelled only A and B. Share of {calls} calls each side won.",
        f"{focus.label} against each model on how human it sounds.",
    )
    label_width = max(text_width(f"vs {p.b.label}", 15, True) for p in mine)
    center = MARGIN + label_width + 40 + 300
    half = 300
    top = card.y + 52
    row, thickness = 46, 24
    card.text(center - 10, top - 20, "← other model preferred", MUTED, 13, anchor="end", weight=600)
    card.text(center + 10, top - 20, f"{focus.label} preferred →", INK, 13, weight=700)
    for index, pair in enumerate(mine):
        scores = pair.scores
        assert scores is not None
        y = top + row * index + row / 2
        card.text(MARGIN, y + 5, f"vs {pair.b.label}", INK, 15, weight=600)
        loss = scores.losses.value / 100 * half
        win = scores.wins.value / 100 * half
        card.bar(center - 1.5, center - 1.5 - loss, y, thickness, BAR)
        card.bar(center + 1.5, center + 1.5 + win, y, thickness, ACCENT)
        card.text(
            center - loss - 10,
            y + 5,
            f"{scores.losses.value:.0f}%",
            SECONDARY,
            14,
            anchor="end",
            numeric=True,
        )
        low, high = scores.wins.ci
        extra = f"{low:.0f}–{high:.0f}"  # noqa: RUF001
        if scores.ties.value:
            extra += f" · ties {scores.ties.value:.0f}%"
        card.runs(
            center + win + 12,
            y + 5,
            [(f"{scores.wins.value:.0f}%", INK, 16, 700, 0), (extra, MUTED, 13, 400, 8)],
        )
    bottom = top + row * len(mine)
    card.line(center, top - 6, center, bottom + 4, AXIS, width=1.5)
    card.y = bottom + 8
    judge_model = mine[0].provenance.judge_model
    judge = judge_label(judge_model, entries)
    method = (
        f"Judge: {judge}, reading only what each agent said aloud, with no system prompt, "
        f"tools or model names. {focus.label}'s call was shown first in half of the scenarios "
        "against each model. Lines after each share are its 95% interval, bootstrapped by agent."
    )
    if any(p.b.run == judge_model for p in won):
        method += f" The judge preferred {focus.label} even over itself."
    return card.render(method)


def charts(entries: Sequence[Entry], pairs: Sequence[ArenaSummary]) -> dict[str, str]:
    """Every chart, by file name."""
    drawn: dict[str, str | None] = {
        "score-vs-latency.svg": score_vs_latency(entries),
    }
    for index, (key, (name, question, method)) in enumerate(CAPABILITIES.items(), start=1):
        drawn[f"{key.replace('_', '-')}.svg"] = ranked_bars(
            entries,
            key,
            f"MTVA-Bench · capability {index} of {len(CAPABILITIES)}",
            name,
            f"{question} Scored 0 to 100; lines are 95% intervals.",
            f"{method} {SCORING}",
        )
    drawn["sounds-human.svg"] = sounds_human(entries, pairs)
    drawn["languages.svg"] = by_language(entries)
    drawn["mtva-score.svg"] = ranked_bars(
        entries,
        "mtva_score",
        "MTVA-Bench · overall",
        "The MTVA Score",
        "The four capabilities combined. Scored 0 to 100; lines are 95% intervals.",
        "The MTVA Score is the mean of task completion, tool use, instruction following and "
        f"faithfulness. {SCORING}",
    )
    return {name: svg for name, svg in drawn.items() if svg is not None}
