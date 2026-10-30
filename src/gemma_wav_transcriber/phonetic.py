"""Turns NATO phonetic spelling ("alpha bravo charlie") into letters ("ABC")."""

from __future__ import annotations

import re

LETTERS = {
    "alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E",
    "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I", "juliet": "J", "juliett": "J",
    "kilo": "K", "lima": "L", "mike": "M", "november": "N", "oscar": "O", "papa": "P",
    "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T", "uniform": "U", "victor": "V",
    "whiskey": "W", "whisky": "W", "x-ray": "X", "xray": "X", "x ray": "X", "yankee": "Y",
    "zulu": "Z",
}
DIGITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "niner": "9",
}


def _alternation(words) -> str:
    # Longest first, and spaces/hyphens inside a word ("x-ray") match either.
    return "|".join(
        re.escape(w).replace(r"\-", r"[\s\-]?").replace(r"\ ", r"[\s\-]?")
        for w in sorted(words, key=len, reverse=True)
    )


_LETTER = rf"(?:{_alternation(LETTERS)})"
_DIGIT = rf"(?:{_alternation(DIGITS)})"
_SEP = r"(?:\s*[,\-]\s*|\s+)"
# At least two code words in a row; digits may only appear between them,
# so "kilo eight sierra" is K8S but "one two three" is left alone.
_RUN = re.compile(
    rf"(?<![\w\-]){_LETTER}(?:{_SEP}(?:{_DIGIT}{_SEP})*{_LETTER})+(?![\w\-])", re.IGNORECASE
)
_TOKEN = re.compile(rf"{_LETTER}|{_DIGIT}", re.IGNORECASE)


def _key(word: str) -> str:
    return re.sub(r"[\s\-]+", " ", word.lower()).replace("x ray", "x-ray")


def to_letters(text: str) -> str:
    """Replace each run of two or more phonetic code words with its letters."""

    def spell(match: re.Match[str]) -> str:
        return "".join(
            LETTERS.get(k) or DIGITS[k] for k in map(_key, _TOKEN.findall(match.group(0)))
        )

    return _RUN.sub(spell, text)
