"""Prompt construction for the AI side (scope.md 4.2).

The governing idea is a matched-pairs corpus. Every AI document is seeded from one
human document and inherits its genre, topic, entities, target length and structural
shape. The AI side's topic and length distributions are then identical to the human
side *by construction*, so the classifier cannot separate the classes on subject matter
or document size -- only on style.

That also defuses the markdown problem at the source. Heading and list counts are copied
from the seed, so structure carries no label information and does not have to be
stripped. Stripping it would make the corpus easier than the real web, which shows up as
a good validation number and a bad false-positive rate in production.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

from ..data.genre import Genre

PromptStyle = Literal[
    "continuation", "rewrite", "topic_prompt", "persona_instruction", "structured"
]

PROMPT_VERSION = "p1"

# How each genre is meant to read. Keyed by the scope.md 4.2 genres.
GENRE_REGISTER: dict[Genre, str] = {
    "news": (
        "Write in the register of a wire-service news report: inverted pyramid, "
        "attributed statements, no first person, no editorialising."
    ),
    "product_marketing": (
        "Write as marketing or product copy for a company website. Promotional but "
        "not absurd, aimed at a buyer who is comparing options."
    ),
    "press_release": (
        "Write as a corporate press release, with a dateline, an executive quote, "
        "and a boilerplate 'About' paragraph at the end."
    ),
    "forum_comment": (
        "Write as a single forum comment or reply. Casual, first person, no headings. "
        "It may be a fragment of a longer conversation and may end mid-thought."
    ),
    "blog_personal": (
        "Write as a personal blog post. First person singular, a specific point of "
        "view, willing to digress."
    ),
    "technical_docs": (
        "Write as technical documentation or a technical answer: imperative mood, "
        "concrete steps or specifics, no marketing language."
    ),
    "encyclopedia": (
        "Write as an encyclopedia article: neutral point of view, third person, "
        "no direct address to the reader, no conclusion."
    ),
    "academic_formal": (
        "Write as formal academic or institutional prose: hedged claims, passive "
        "constructions where natural, no direct address to the reader."
    ),
    "other": "Write as ordinary web prose.",
}

# Angles keep 500 documents on one topic from all taking the same stance.
ANGLES = (
    "a sceptical assessment",
    "a practical how-to",
    "an enthusiastic recommendation",
    "a historical overview",
    "a comparison against alternatives",
    "a cautionary account of what goes wrong",
    "an explanation for a complete beginner",
    "an argument against the conventional view",
)

PERSONAS: dict[Genre, tuple[str, ...]] = {
    "news": (
        "a regional newspaper reporter with a fifteen-year beat covering local government",
        "a trade-publication journalist who writes for industry insiders",
    ),
    "product_marketing": (
        "a product marketer at a mid-size software company",
        "a freelance copywriter paid by the word",
    ),
    "press_release": (
        "a corporate communications officer preparing an announcement for the wire",
        "an investor-relations manager at a listed company",
    ),
    "forum_comment": (
        "a hobbyist who has been on this forum for a decade and is mildly annoyed by the question",
        "someone who just solved this exact problem last week and is pleased about it",
    ),
    "blog_personal": (
        "someone writing a personal blog they have kept for years and nobody much reads",
        "a practitioner writing up something they learned the hard way",
    ),
    "technical_docs": (
        "a maintainer writing documentation for a library they wrote",
        "a support engineer writing a knowledge-base article",
    ),
    "encyclopedia": (
        "a volunteer encyclopedia editor working from cited sources",
        "a reference-work contributor writing a short survey entry",
    ),
    "academic_formal": (
        "a second-year PhD student writing a related-work section",
        "a grant administrator writing a project summary",
    ),
    "other": ("someone writing for the web",),
}


@dataclass(frozen=True)
class SeedCard:
    """What an AI generation inherits from its human seed."""

    doc_id: str
    genre: Genre
    topic: str
    entities: tuple[str, ...] = ()
    # Concrete figures lifted from the seed: years, quantities, percentages.
    # Without these, a model given only a topic writes prose with almost no
    # numbers -- measured at 0.13x the human digit density, against 0.77x for the
    # one style that does see the seed text. That gap became the single strongest
    # class signal in the corpus.
    figures: tuple[str, ...] = ()
    target_words: int = 400
    headings: int = 0
    lists: int = 0
    audience: str = "a general web reader"
    publication_type: str = "an article"
    prefix: str = ""  # for continuation style
    text: str = ""  # for rewrite style


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    style: PromptStyle
    seed_doc_id: str
    genre: Genre
    target_words: int
    persona: str | None = None
    angle: str | None = None
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def prompt_id(self) -> str:
        payload = json.dumps(
            {"s": self.system, "u": self.user, "v": PROMPT_VERSION}, sort_keys=True
        )
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def structure_clause(card: SeedCard) -> str:
    """Mirror the seed's shape, so structure carries no label information."""
    if card.headings == 0 and card.lists == 0:
        return "Do not use headings, bullet points or numbered lists."
    parts = []
    if card.headings:
        parts.append(f"Use about {card.headings} section headings.")
    if card.lists:
        parts.append(f"Include about {card.lists} bulleted or numbered list items.")
    return " ".join(parts)


def figures_clause(card: SeedCard) -> str:
    """Ask for concrete specifics, seeded from the human document.

    Real writing on these subjects is full of dates, quantities and percentages. A
    model handed only a topic writes almost none, which makes numeric density a
    near-perfect class signal rather than a property of AI writing.
    """
    if not card.figures:
        return (
            "\nInclude concrete specifics -- dates, quantities, percentages, named "
            "sources -- as a piece of real writing on this subject would."
        )
    return (
        "\nWork in these specific figures where they fit naturally, and add other "
        f"concrete dates and quantities as needed: {', '.join(card.figures)}"
    )


def base_system(card: SeedCard) -> str:
    return (
        f"You are writing a single piece of {card.publication_type} for publication on "
        "the web. Output only the finished text. Do not add a preamble, a title line, a "
        "sign-off, or any commentary about the task.\n"
        f"{GENRE_REGISTER[card.genre]}\n"
        f"{structure_clause(card)}\n"
        f"Target length: about {card.target_words} words."
    )


def build(
    card: SeedCard,
    style: PromptStyle,
    *,
    persona: str | None = None,
    angle: str | None = None,
) -> Prompt:
    system = base_system(card)

    if style == "continuation":
        system = (
            "Continue the following text. Match its voice, tense, formatting and level "
            "of detail exactly, including how often it cites specific dates, quantities "
            "and figures. Do not summarize, do not restate what came before, and "
            "do not conclude the piece -- keep writing from where it stops. "
            f"Write about {card.target_words} more words. Output only the continuation."
        )
        user = card.prefix
    elif style == "rewrite":
        system += (
            "\nRewrite the text below. Keep every fact, all named entities, and the "
            "same order of ideas, at roughly the same length. Change the wording and "
            "sentence structure throughout."
        )
        user = card.text
    elif style == "topic_prompt":
        # The only style where AI wording is not derived from human wording. Without
        # it the classifier can learn to detect paraphrase rather than AI.
        entities = ", ".join(card.entities) if card.entities else "the main points"
        user = (
            f"Write {card.publication_type} about: {card.topic}\n"
            f"Cover: {entities}\n"
            f"Audience: {card.audience}\n"
            f"Angle: {angle or ANGLES[0]}"
            f"{figures_clause(card)}"
        )
    elif style == "persona_instruction":
        chosen = persona or PERSONAS[card.genre][0]
        system = f"You are {chosen}. " + system
        user = f"Write {card.publication_type} about: {card.topic}{figures_clause(card)}"
    elif style == "structured":
        system += (
            "\nStructure the piece as a numbered list of points, each with a short "
            "paragraph of explanation, with a brief introduction and no conclusion."
        )
        user = f"Write {card.publication_type} about: {card.topic}{figures_clause(card)}"
    else:  # pragma: no cover - exhaustive over PromptStyle
        raise ValueError(f"unknown style {style}")

    return Prompt(
        system=system,
        user=user,
        style=style,
        seed_doc_id=card.doc_id,
        genre=card.genre,
        target_words=card.target_words,
        persona=persona,
        angle=angle,
    )
