import pytest

from mtva_bench.llm.timing import StreamClock, has_sentence_end
from mtva_bench.llm.types import Delta


@pytest.mark.parametrize(
    ("text", "ends"),
    [
        ("Sure", False),
        ("Sure.", False),
        ("Sure. ", True),
        ("Your total is 3.5", False),
        ("Can I help? Yes", True),
        ("जी हाँ।", True),
        ("आपका ऑर्डर तैयार है। धन्यवाद", True),
        ("ありがとうございます。", True),
        ("That is Rs. 500 for the plan", False),
        ("Dr. Mehta will call", False),
        ("Hola Sra. García, le llamo", False),
        ("Signed by A. P. J. Kalam", False),
        ("Use the app, e.g. Paytm", False),
        ("That is Rs. 500. Shall I book it?", True),
    ],
)
def test_sentence_end(text: str, ends: bool) -> None:
    assert has_sentence_end(text) is ends


def test_clock_stamps_in_order() -> None:
    clock = StreamClock()
    for delta in [None, Delta(tool_call=True), Delta("Okay"), Delta(". Let"), Delta(" me")]:
        clock.observe(delta)
    timing = clock.timing()
    assert timing.first_token_ms <= timing.first_speech_token_ms <= timing.first_sentence_ms
    assert clock.speech == "Okay. Let"


def test_unpunctuated_speech_counts_as_a_sentence_at_the_end() -> None:
    clock = StreamClock()
    clock.observe(Delta("Okay one moment"))
    timing = clock.timing()
    assert timing.first_sentence_ms == timing.total_ms


def test_tool_only_turn_has_no_speech() -> None:
    clock = StreamClock()
    clock.observe(Delta(tool_call=True))
    timing = clock.timing()
    assert timing.first_speech_token_ms is None
    assert timing.first_sentence_ms is None
