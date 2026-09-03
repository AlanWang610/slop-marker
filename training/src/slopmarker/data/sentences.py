"""Sentence splitting.

Hand-written on purpose. This is a cross-language coupling point: the extension has to
produce byte-identical boundaries, and every off-the-shelf splitter fails that test.
pysbd/spaCy have no faithful TypeScript port; nltk.punkt and blingfire are data-driven
and would mean shipping and versioning a model file; Intl.Segmenter and PyICU implement
the same spec but against ICU data that differs per browser build and per OS, so their
boundaries can drift silently between a user's Firefox and our training box.

Input is normally whitespace-collapsed (see normalize.collapse_whitespace), so there
are no newline rules and a single space is the only separator. Nothing here depends on
that being true, so the TypeScript port need not reproduce a precondition.
"""

from __future__ import annotations

from .normalize import ELLIPSIS, WHITESPACE

# Tokens that end in a period without ending a sentence. Compared case-sensitively
# against the word preceding the period, with that period stripped.
ABBREVIATIONS = frozenset(
    """
    Mr Mrs Ms Dr Prof Sr Jr St Mt Rev Gen Col Lt Sgt Capt Gov Sen Rep Adm
    Inc Ltd Co Corp Dept Univ Ave Blvd Rd No Fig Vol Ch Sec
    Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec
    Mon Tue Tues Wed Thu Thurs Fri Sat Sun
    vs etc al approx ca esp min max ibid cf
    e.g i.e a.m p.m U.S U.K Ph.D M.D B.A M.A
    """.split()  # noqa: SIM905 - a word list reads better than a literal
)

_TERMINATORS = frozenset(".!?" + ELLIPSIS)
# Closing punctuation that may sit between the terminator and the space:
# " ' ) ] } and the right curly quotes.
_CLOSERS = frozenset("\"')]}" + chr(0x2019) + chr(0x201D))
# A sentence may only start with a capital, a digit, or one of these openers:
# " ' ( [ { and the left curly quotes.
_OPENERS = frozenset("\"'([{" + chr(0x2018) + chr(0x201C))


def _is_upper(ch: str) -> bool:
    """Portable uppercase test.

    Not `str.isupper()`: its Unicode semantics have no natural JavaScript equivalent.
    This form behaves identically in both languages, including for oddities like the
    German sharp s, whose uppercase is two characters.
    """
    return ch != ch.lower() and ch == ch.upper()


def _is_digit(ch: str) -> bool:
    r"""ASCII digits only. `str.isdigit()` is also true for superscripts and other
    Unicode digit forms, which JavaScript's `\d` is not."""
    return "0" <= ch <= "9"


def _blocks_boundary(text: str, dot: int) -> bool:
    """True if the period at `dot` is part of an abbreviation or an initial."""
    if text[dot] != ".":
        return False  # only periods are ambiguous; ! and ? are not
    start = text.rfind(" ", 0, dot) + 1
    word = text[start:dot]
    is_initial = len(word) == 1 and word.lower() != word.upper()  # a single letter
    return is_initial or word in ABBREVIATIONS


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans, one per sentence, covering all of `text`."""
    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] not in _TERMINATORS:
            i += 1
            continue
        if _blocks_boundary(text, i):
            i += 1
            continue
        end = i + 1
        while end < n and text[end] in _TERMINATORS:  # "?!", "..."
            end += 1
        while end < n and text[end] in _CLOSERS:  # He said "Hello."
            end += 1
        # A boundary needs a space and then something that can open a sentence.
        if end < n and text[end] == " " and end + 1 < n:
            nxt = text[end + 1]
            if _is_upper(nxt) or _is_digit(nxt) or nxt in _OPENERS:
                spans.append((start, end))
                start = end + 1
                i = start
                continue
        i = end
    # Trim trailing whitespace off the last span rather than emitting it. Input is
    # normally already collapsed, but not relying on that keeps the TypeScript port
    # from having to reproduce a precondition.
    end = n
    while end > start and text[end - 1] in WHITESPACE:
        end -= 1
    if end > start:
        spans.append((start, end))
    return spans


def sentences(text: str) -> list[str]:
    return [text[a:b] for a, b in split_sentences(text)]
