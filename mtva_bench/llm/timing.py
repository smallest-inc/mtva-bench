import re
from time import perf_counter

from mtva_bench.llm.types import Delta, Timing

LATIN_SENTENCE_END = frozenset(".!?;…")
SENTENCE_END = LATIN_SENTENCE_END | frozenset("。？！；．｡।॥؟؛۔")  # noqa: RUF001
ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "smt", "shri",
        "rs", "pvt", "ltd", "co", "inc", "dept", "approx", "vs", "e.g", "i.e",
        "sra", "srta", "dra", "ud", "uds", "lic", "ing", "av", "avda", "núm",
    }
)  # fmt: skip
WORD_BEFORE = re.compile(r"[\w.]+$")


def has_sentence_end(text: str) -> bool:
    for index, char in enumerate(text):
        if char not in SENTENCE_END:
            continue
        if char not in LATIN_SENTENCE_END:
            return True
        if index + 1 < len(text) and text[index + 1].isspace() and not abbreviated(text, index):
            return True
    return False


def abbreviated(text: str, index: int) -> bool:
    if text[index] != ".":
        return False
    word = WORD_BEFORE.search(text, 0, index)
    if word is None:
        return False
    token = word.group().lower()
    return token in ABBREVIATIONS or (len(token) == 1 and token.isalpha())


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


class StreamClock:
    def __init__(self) -> None:
        self.start = perf_counter()
        self.first_token_ms: float | None = None
        self.first_speech_token_ms: float | None = None
        self.first_sentence_ms: float | None = None
        self.speech = ""

    def observe(self, delta: Delta | None) -> None:
        if delta is None:
            return
        now = elapsed_ms(self.start)
        if self.first_token_ms is None:
            self.first_token_ms = now
        if not delta.text or self.first_sentence_ms is not None:
            return
        if self.first_speech_token_ms is None:
            self.first_speech_token_ms = now
        self.speech += delta.text
        if has_sentence_end(self.speech):
            self.first_sentence_ms = now

    def timing(self) -> Timing:
        total_ms = elapsed_ms(self.start)
        first_sentence_ms = self.first_sentence_ms
        if first_sentence_ms is None and self.speech:
            first_sentence_ms = total_ms
        return Timing(
            first_token_ms=self.first_token_ms,
            first_speech_token_ms=self.first_speech_token_ms,
            first_sentence_ms=first_sentence_ms,
            total_ms=total_ms,
        )
