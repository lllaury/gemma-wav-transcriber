import pytest

from gemma_wav_transcriber.prompt import build_prompt, language_name
from gemma_wav_transcriber.vocab import Vocabulary, VocabError, load, parse

SAMPLE = """
# comment
Kubernetes
C#
cooper netties = Kubernetes
sequel, see quel -> SQL   # trailing comment

[acronyms]
NASA, FBI
hip ah = HIPAA
IT
COVID-19
"""


def test_parse_sections_and_pronunciations():
    v = parse(SAMPLE)
    assert v.terms == ["Kubernetes", "C#"]
    assert v.acronyms == ["NASA", "FBI", "HIPAA", "IT", "COVID-19"]
    assert ("cooper netties", "Kubernetes") in v.sounds_like
    assert ("see quel", "SQL") in v.sounds_like
    assert ("hip ah", "HIPAA") in v.sounds_like


@pytest.mark.parametrize(
    "heard, expected",
    [
        ("We deploy on cooper-netties and Sequel.", "We deploy on Kubernetes and SQL."),
        ("He worked at Nasa and the f.b.i. for years.", "He worked at NASA and the FBI for years."),
        ("The F B I and n a s a.", "The FBI and NASA."),
        ("I joined N.A.S.A.", "I joined NASA."),
        ("covid-19 and hip ah rules", "COVID-19 and HIPAA rules"),
        # "it" is an everyday word: only spelled-out forms become IT
        ("Call it support, I T or i.t.", "Call it support, IT or IT."),
        # full stops after acronyms are kept
        ("Ask nasa. Next, the F.B.I. We go.", "Ask NASA. Next, the FBI. We go."),
        ("the f.b.i. agents", "the FBI agents"),
        # plain terms get their spelling back
        ("we run kubernetes and c#", "we run Kubernetes and C#"),
        # letters split across a sentence break are left alone
        ("Plan F. B I", "Plan F. B I"),
    ],
)
def test_correct(heard, expected):
    assert parse(SAMPLE).correct(heard) == expected


def test_longest_phrase_wins():
    v = parse("new york = New York\nnew york times = The New York Times")
    assert v.correct("I read new york times in new york") == "I read The New York Times in New York"


def test_bad_lines_are_reported(tmp_path):
    with pytest.raises(VocabError, match=":2: unknown section"):
        parse("a\n[nope]\n")
    with pytest.raises(VocabError, match=":1: expected"):
        parse("= SQL")
    with pytest.raises(VocabError, match="cannot read"):
        load(tmp_path / "missing.txt")


def test_merge_and_load(tmp_path):
    f = tmp_path / "v.txt"
    f.write_text("﻿Alpha\n[acronyms]\nGPU\n", encoding="utf-8")
    v = Vocabulary()
    v.merge(load(f))
    v.merge(load(f))
    assert v.terms == ["Alpha"] and v.acronyms == ["GPU"]


def test_language_names():
    assert language_name("es") == "Spanish"
    assert language_name("FR") == "French"
    assert language_name("klingon") == "Klingon"
    assert language_name("auto") is None and language_name(None) is None


def test_build_prompt():
    v = parse(SAMPLE)
    p = build_prompt(language="de", vocab=v)
    assert "The speech is in German" in p
    assert "capital letters" in p
    assert "Kubernetes, C#, SQL" in p and '"hip ah" is written HIPAA' in p

    t = build_prompt(language="es", translate_to="en")
    assert "Translate it faithfully into English" in t
    assert "translations" not in t  # the default "no translations" rule is dropped

    plain = build_prompt(acronyms=False)
    assert "capital letters" not in plain and "speech is in" not in plain


def test_example_file_parses():
    from pathlib import Path

    v = load(Path(__file__).parent.parent / "examples" / "vocabulary.txt")
    assert "HIPAA" in v.acronyms and v.correct("we run engine x") == "we run nginx"
