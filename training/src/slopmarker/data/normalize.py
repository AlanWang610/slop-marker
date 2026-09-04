r"""Text normalization. Two functions, two different jobs.

`collapse_whitespace` is what the *model* sees: it only regularizes whitespace, so
curly quotes and em dashes survive. Those are real signal for AI text and stripping
them would throw the signal away.

`normalize_for_hash` is what the *score cache key* is built from (scope.md 7.4). It
folds quotes, dashes and invisibles too, so that syndicated text differing only in
typography is scored once.

Both are ported to TypeScript and pinned by fixtures/normalize.json. Every character
class is written out as an explicit codepoint and no regex `\s` is used anywhere:
Python's `\s` matches U+001C-U+001F, JavaScript's matches U+FEFF and not those.
Relying on it would guarantee a drift bug that only fires on scraped web text.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Every character we treat as whitespace. Nothing else counts.
WHITESPACE = frozenset(
    map(
        chr,
        (
            0x09,  # tab
            0x0A,  # line feed
            0x0B,  # vertical tab
            0x0C,  # form feed
            0x0D,  # carriage return
            0x20,  # space
            0x00A0,  # no-break space
            0x1680,  # ogham space mark
            *range(0x2000, 0x200B),  # en quad .. hair space
            0x2028,  # line separator
            0x2029,  # paragraph separator
            0x202F,  # narrow no-break space
            0x205F,  # medium mathematical space
            0x3000,  # ideographic space
        ),
    )
)

# Zero-width and format characters deleted outright. U+200C ZWNJ and U+200D ZWJ are
# deliberately NOT here: they are load-bearing in emoji sequences and in Indic and
# Arabic scripts, and deleting them would mangle real text.
INVISIBLE = frozenset(
    map(
        chr,
        (
            0x00AD,  # soft hyphen
            0x200B,  # zero-width space
            0x2060,  # word joiner
            0xFEFF,  # zero-width no-break space / BOM
        ),
    )
)

_SINGLE_QUOTES = (0x2018, 0x2019, 0x201A, 0x201B, 0x2032)  # curly quotes, prime
_DOUBLE_QUOTES = (0x201C, 0x201D, 0x201E, 0x201F, 0x2033)  # curly quotes, double prime
_DASHES = (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212)  # hyphen .. minus
ELLIPSIS = chr(0x2026)

_FOLD: dict[str, str] = {
    **{chr(cp): "'" for cp in _SINGLE_QUOTES},
    **{chr(cp): '"' for cp in _DOUBLE_QUOTES},
    **{chr(cp): "-" for cp in _DASHES},
}


def collapse_whitespace(text: str) -> str:
    """Map every whitespace character to a space, collapse runs, strip the ends."""
    out: list[str] = []
    pending_space = False
    for ch in text:
        if ch in WHITESPACE:
            pending_space = bool(out)
        else:
            if pending_space:
                out.append(" ")
                pending_space = False
            out.append(ch)
    return "".join(out)


def normalize_for_hash(text: str) -> str:
    """Aggressive normalization used only to build the score-cache key."""
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    for ch in text:
        if ch in INVISIBLE:
            continue
        if ch == ELLIPSIS:
            out.append("...")
        else:
            out.append(_FOLD.get(ch, ch))
    return collapse_whitespace("".join(out))


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def count_words(text: str) -> int:
    """Word count, defined on whitespace-collapsed text so it needs no regex at all.

    This is the one definition used by training, by windowing and by the extension.
    """
    collapsed = collapse_whitespace(text)
    return len(collapsed.split(" ")) if collapsed else 0


# Markdown syntax, removed from both classes before windowing.
#
# This is not "removing model style". At inference the extension reads rendered DOM
# text: a heading arrives as the words inside an <h2>, and bold arrives as the words
# inside a <strong>. The literal `#` and `**` characters cannot reach the model in
# production. Human web text is extracted the same way and so contains none of them,
# which makes a literal asterisk an almost perfect classifier -- measured at 8x more
# frequent in AI text than human before this ran.
#
# The *choice* to emphasise or to use headings survives, as word choice and as
# structure. Only the syntax that production never sees is removed.
_MD_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.M)
_MD_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.S)
_MD_BULLET = re.compile(r"^[ \t]{0,3}([-*+])[ \t]+", re.M)
_MD_RULE = re.compile(r"^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.M)
_MD_CODE_FENCE = re.compile(r"^[ \t]{0,3}```.*$", re.M)
_MD_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_MD_LINK = re.compile(r"\[([^\]\n]+)\]\([^)\n]*\)")
_MD_BLOCKQUOTE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.M)


def strip_markdown(text: str) -> str:
    """Remove markdown syntax, keeping the words it wrapped."""
    text = _MD_CODE_FENCE.sub("", text)
    text = _MD_RULE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BULLET.sub("", text)
    return _MD_BLOCKQUOTE.sub("", text)
