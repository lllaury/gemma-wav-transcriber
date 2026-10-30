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
        ("alpha beta charlie", "ABC"),
        ("the cluster is kilo 8 sierra", "the cluster is K8S"),
        # left alone: single code words, plain numbers, digits at the edges
        ("Delta flight to Lima", "Delta flight to Lima"),
        ("one two three", "one two three"),
        ("echo one two", "echo one two"),
        ("echo 1 2", "echo 1 2"),
        ("Romeo and Juliet", "Romeo and Juliet"),
        ("alphabet soup bravo", "alphabet soup bravo"),
    ],
)
def test_to_letters(heard, expected):
    assert to_letters(heard) == expected


def test_known_acronym_splits_off_next_word():
    heard = "covered by hotel india papa alpha alpha charlie flew delta to lima"
    assert to_letters(heard) == "covered by HIPAAC flew delta to lima"
    assert to_letters(heard, ["HIPAA"]) == "covered by HIPAA charlie flew delta to lima"
    # a known acronym that is spelled exactly is not split
    assert to_letters("alpha bravo charlie", ["AB", "ABC"]) == "ABC"
