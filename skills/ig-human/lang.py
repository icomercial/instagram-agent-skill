"""
lang.py - tell Spanish from English, so humanize.py and detect.py load the
right lexicon. Running the English lexicon on Spanish text is worse than not
running it: "crucial" exists in both languages, and the English replacement
("important") lands in the middle of a Spanish sentence.
"""

import re

ES = {
    "el", "la", "los", "las", "un", "una", "que", "de", "del", "y", "en",
    "es", "por", "para", "con", "tu", "te", "mi", "lo", "se", "su", "pero",
    "como", "más", "sin", "porque", "esto", "este", "esta", "hay", "muy",
    "ya", "todo", "también", "cuando", "nos", "les", "son", "está",
}
EN = {
    "the", "a", "an", "and", "of", "to", "is", "in", "for", "with", "you",
    "your", "my", "it", "this", "that", "on", "but", "how", "what", "i",
    "me", "be", "are", "was", "not", "just", "do", "have", "we", "they",
}
WORD_RE = re.compile(r"[^\W\d_]+")


def lang_of(text):
    """'es' or 'en'. Function words, plus characters only Spanish uses."""
    low = [w.lower() for w in WORD_RE.findall(text)]
    es = sum(1 for w in low if w in ES) + 2 * len(re.findall(r"[ñ¿¡]", text.lower()))
    en = sum(1 for w in low if w in EN)
    return "es" if es > en else "en"


def lexicon_path(here, text, lang="auto"):
    """slop.es.json for Spanish, slop.json for everything else."""
    import os
    if lang == "auto":
        lang = lang_of(text)
    return os.path.join(here, "slop.es.json" if lang == "es" else "slop.json")
