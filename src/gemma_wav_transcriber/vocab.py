"""Custom vocabulary: domain terms, pronunciations and acronyms.

A vocabulary file is plain UTF-8 text, one entry per line::

    # comment
    Kubernetes                    a term to spell exactly like this
    cooper netties = Kubernetes   "how it sounds" = how to write it
    sequel, see quel -> SQL       several sound-alikes, "->" also works

    [acronyms]
    NASA, FBI                     acronyms, always written in capitals
    hip ah = HIPAA                acronym with how it is pronounced

The entries are given to the model as context, and the transcript is then
post-corrected: sound-alikes are replaced by their spelling, and acronyms
are fixed to capitals, including spelled-out forms like "h i p a a".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SECTIONS = ("terms", "acronyms")

# Acronyms that are also everyday words are only corrected when spelled out
# letter by letter or already in capitals, so "it" never becomes "IT".
COMMON_WORDS = frozenset(
    "a am an and any as at be by can do go he hi i if in is it me my no of oh ok on or "
    "our out so to up us we who why yes you all are art bat bed big bus car cat cod cot "
    "dot ear eat end era far few fit fun gap gas god hat her him his hot ice its job key "
    "lab law led let lot low mad man map may men mid mix mom net new nor not now oil old "
    "one own pan pay pet pin pop pro put ram ran raw red rid rip rob rot row run sad saw "
    "say sea see set she sit six sky son spa sun tab tan tax tea ten the tie tip top toy "
    "try two use van via war was way web wet who win yet zip".split()
)


class VocabError(Exception):
    """Raised when a vocabulary file cannot be read or parsed."""


@dataclass
class Vocabulary:
    terms: list[str] = field(default_factory=list)
    acronyms: list[str] = field(default_factory=list)
    # (sounds like, written) pairs, for terms and acronyms alike
    sounds_like: list[tuple[str, str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.terms or self.acronyms or self.sounds_like)

    def merge(self, other: Vocabulary) -> None:
        for mine, theirs in (
            (self.terms, other.terms),
            (self.acronyms, other.acronyms),
            (self.sounds_like, other.sounds_like),
        ):
            mine.extend(x for x in theirs if x not in mine)

    def prompt_hint(self) -> str:
        """Context for the model describing the expected vocabulary."""
        parts = []
        written = [w for _, w in self.sounds_like if w not in self.acronyms]
        terms = list(dict.fromkeys(self.terms + written))
        if terms:
            parts.append("Spell these terms exactly as written here: " + ", ".join(terms) + ".")
        if self.acronyms:
            parts.append(
                "These acronyms may be spoken; write them in capital letters without periods: "
                + ", ".join(self.acronyms) + "."
            )
        if self.sounds_like:
            pairs = "; ".join(f'"{s}" is written {w}' for s, w in self.sounds_like)
            parts.append("Pronunciation guide (what is heard, then how to write it): " + pairs + ".")
        return " ".join(parts)

    def correct(self, text: str) -> str:
        """Apply sound-alike replacements and acronym capitalisation to a transcript."""
        for pattern, written in self._rules():
            text = pattern.sub(lambda m, w=written: _keep_final_period(m, w), text)
        return text

    def _rules(self) -> list[tuple[re.Pattern[str], str]]:
        rules = []
        # Longest phrases first so "new york times" wins over "new york".
        for sound, written in sorted(self.sounds_like, key=lambda p: -len(p[0])):
            words = re.findall(r"\w+", sound)
            if words:
                body = r"[\s\-]+".join(map(re.escape, words))
                rules.append((re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE), written))
        for acro in self.acronyms:
            rules.append((_acronym_pattern(acro), acro))
        return rules


def _acronym_pattern(acro: str) -> re.Pattern[str]:
    letters = re.sub(r"[^A-Za-z0-9]", "", acro)
    if letters != acro or len(letters) < 2:
        # Something like "COVID-19": fix the case only.
        return re.compile(rf"(?<!\w){re.escape(acro)}(?!\w)", re.IGNORECASE)
    # One separator between letters, so "plan A. I think" is not read as "AI".
    spelled = r"[\s.\-]".join(map(re.escape, letters))
    if letters.lower() in COMMON_WORDS:
        # Spelled out ("I T", "I.T.") in any case, or the exact acronym.
        body = rf"(?i:{spelled}\.?)|{re.escape(acro)}"
    else:
        # Any case, with or without separators: "Nasa", "n a s a", "N.A.S.A."
        loose = r"[\s.\-]?".join(map(re.escape, letters))
        body = rf"(?i:{loose}\.?)"
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)")


def _keep_final_period(match: re.Match[str], written: str) -> str:
    # "...at N.A.S.A." at the end of a chunk: the last dot also ends the sentence.
    if match.group(0).endswith(".") and match.end() == len(match.string):
        return written + "."
    return written


# "#" starts a comment at the start of a line or after a space, so "C#" is a term.
_COMMENT = re.compile(r"(?:^|\s)#.*")


def _split_items(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def parse(text: str, source: str = "<vocabulary>") -> Vocabulary:
    vocab = Vocabulary()
    section = "terms"
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _COMMENT.sub("", raw).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            if section not in SECTIONS:
                raise VocabError(
                    f"{source}:{lineno}: unknown section [{section}] (use [terms] or [acronyms])"
                )
            continue
        target = vocab.acronyms if section == "acronyms" else vocab.terms
        sep = "->" if "->" in line else "=" if "=" in line else None
        if sep is None:
            for item in _split_items(line):
                if item not in target:
                    target.append(item)
            continue
        sounds, written = (s.strip() for s in line.split(sep, 1))
        if not written or not sounds:
            raise VocabError(f"{source}:{lineno}: expected 'sounds like = written', got {raw.strip()!r}")
        if section == "acronyms" and written not in vocab.acronyms:
            vocab.acronyms.append(written)
        for sound in _split_items(sounds):
            pair = (sound, written)
            if pair not in vocab.sounds_like:
                vocab.sounds_like.append(pair)
    return vocab


def load(path: Path) -> Vocabulary:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        raise VocabError(f"cannot read vocabulary file {path}: {e}") from e
    except UnicodeDecodeError as e:
        raise VocabError(f"vocabulary file {path} is not UTF-8 text") from e
    return parse(text, str(path))
