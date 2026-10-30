import pytest

from gemma_wav_transcriber.phonetic import to_letters


@pytest.mark.parametrize(
    "heard, expected",
    [
        ("Call sign Alpha Bravo Charlie.", "Call sign ABC."),
        ("alfa, bravo, x-ray now", "ABX now"),
        ("X ray delta and Xray echo", "XD and XE"),
        ("golf-hotel-india", "GHI"),
        ("the kilo eight sierra cluster", "the K8S cluster"),
        ("Juliett whisky niner zulu", "JW9Z"),
        # left alone: single code words, plain numbers, digits at the edges
        ("Delta flight to Lima", "Delta flight to Lima"),
        ("one two three", "one two three"),
        ("echo one two", "echo one two"),
        ("Romeo and Juliet", "Romeo and Juliet"),
        ("alphabet soup bravo", "alphabet soup bravo"),
    ],
)
def test_to_letters(heard, expected):
    assert to_letters(heard) == expected
