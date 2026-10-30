"""Builds the instruction sent with each audio chunk."""

from __future__ import annotations

from .vocab import Vocabulary

DEFAULT_PROMPT = (
    "Transcribe the speech in this audio clip verbatim. "
    "Output only the spoken words as plain text, with punctuation. "
    "Do not add commentary, labels, translations or descriptions. "
    "If there is no intelligible speech, output nothing."
)

TRANSLATE_PROMPT = (
    "Listen to the speech in this audio clip and translate it. "
    "Output only the translation as plain text, with punctuation. "
    "Do not add commentary, labels or descriptions. "
    "If there is no intelligible speech, output nothing."
)

ACRONYM_RULE = (
    "Write acronyms and initialisms in capital letters without periods or spaces "
    "(for example FBI, NASA, API), even when they are spoken letter by letter."
)

PHONETIC_RULE = (
    "When letters are spelled with the NATO phonetic alphabet (Alpha, Bravo, Charlie ...), "
    "write the letters themselves, for example \"alpha bravo charlie\" as ABC."
)

# Common ISO 639-1 codes; anything else is passed to the model as written.
LANGUAGES = {
    "ar": "Arabic", "bn": "Bengali", "ca": "Catalan", "cs": "Czech", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "fa": "Persian",
    "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian",
    "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sk": "Slovak", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "vi": "Vietnamese", "zh": "Chinese",
}


def language_name(value: str | None) -> str | None:
    """Turn "es" or "spanish" into "Spanish"; "auto" or empty means detect."""
    if not value or value.strip().lower() == "auto":
        return None
    value = value.strip()
    return LANGUAGES.get(value.lower(), value[:1].upper() + value[1:])


def build_prompt(
    base: str = DEFAULT_PROMPT,
    language: str | None = None,
    translate_to: str | None = None,
    vocab: Vocabulary | None = None,
    acronyms: bool = True,
    phonetic: bool = True,
) -> str:
    lang = language_name(language)
    target = language_name(translate_to)
    translating = bool(target) and target != lang
    parts = [TRANSLATE_PROMPT if translating and base == DEFAULT_PROMPT else base]
    if translating:
        heard = f"The speech is in {lang}. " if lang else ""
        parts.append(f"{heard}Translate it faithfully into {target} and write only {target}.")
    elif lang:
        parts.append(f"The speech is in {lang}. Write the transcript in {lang}, in its usual script.")
    if acronyms:
        parts.append(ACRONYM_RULE)
    if phonetic:
        parts.append(PHONETIC_RULE)
    if vocab:
        parts.append("This recording may use specialised vocabulary. " + vocab.prompt_hint())
    return " ".join(parts)
