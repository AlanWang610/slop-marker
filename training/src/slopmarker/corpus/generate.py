"""Drive AI-side generation from harvested human seeds.

Each human document produces one or more AI documents that inherit its genre, topic,
length and structure. Three row kinds come out:

  pure AI      styles other than continuation, ai_fraction 1.0
  continuation human prefix + AI body, ai_fraction measured from the split point
  mixed        a human document spliced with its own AI rewrite, fraction measured

Everything derived from one seed shares a `seed_cluster_id`, which is the split
grouping key. Without that, the same topic appears on both sides of the train/test
boundary and the model can memorize topic -> label.

The topic card is derived from the seed document itself rather than from a separate LLM
pass. It costs nothing, adds no latency, and a title plus entity list is all the prompt
needs; the LLM pass in the plan would have bought marginally better topics for an extra
round trip per seed.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..data.genre import Genre
from ..data.normalize import collapse_whitespace, count_words
from ..data.schema import DocumentRow
from ..data.sentences import split_sentences
from .hygiene import clean_generation
from .mixing import Granularity, draw_target_fraction, splice
from .prompts import ANGLES, PERSONAS, Prompt, PromptStyle, SeedCard, build
from .providers import EFFORT_GRID, SAMPLING_GRID, ModelSpec, generate

HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S|^\s*[A-Z][^.!?\n]{2,60}\n[=-]{3,}\s*$", re.M)
BULLET = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+", re.M)
STOPWORDS = frozenset(
    """
    the a an and or but of in on at to for with from by as is are was were be been
    this that these those it its his her their our your my we you they he she i
    not no all any some more most other such only own same so than too very can will
    just should now about into over after before between during under above
    """.split()  # noqa: SIM905 - a word list reads better than a literal
)

# How generation volume is split across the five prompt styles.
STYLE_MIX: tuple[tuple[PromptStyle, float], ...] = (
    ("topic_prompt", 0.26),
    ("persona_instruction", 0.24),
    ("rewrite", 0.22),
    ("structured", 0.15),
    ("continuation", 0.13),
)


@dataclass(frozen=True)
class GenerationTask:
    seed: DocumentRow
    card: SeedCard
    style: PromptStyle
    model: ModelSpec
    cell: dict[str, object]
    cell_name: str
    seed_cluster_id: str


def _pick[T](rng: random.Random, options: Sequence[tuple[T, float]]) -> T:
    """Weighted choice, walked by hand so it does not depend on a library's RNG."""
    total = sum(w for _, w in options)
    draw = rng.random() * total
    running = 0.0
    for value, weight in options:
        running += weight
        if draw < running:
            return value
    return options[-1][0]


def topic_of(doc: DocumentRow) -> str:
    """A short topic line, taken from the document rather than asked for."""
    text = collapse_whitespace(doc.text)
    spans = split_sentences(text)
    first = text[spans[0][0] : spans[0][1]] if spans else text[:120]
    words = first.split(" ")
    return " ".join(words[:16]).rstrip(".,;:")


def entities_of(doc: DocumentRow, limit: int = 6) -> tuple[str, ...]:
    """Capitalised mid-sentence tokens, as a cheap proper-noun proxy."""
    text = collapse_whitespace(doc.text)
    found: list[str] = []
    for match in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z]{2,})\b", text):
        word = match.group(1)
        if word.lower() in STOPWORDS or word in found:
            continue
        found.append(word)
        if len(found) >= limit:
            break
    return tuple(found)


def seed_card(doc: DocumentRow) -> SeedCard:
    words = doc.n_words or count_words(doc.text)
    # The prefix is sized from the *capped* target, not the raw document. Seeds run to
    # 20k words, and taking 30% of that produced 5000-word continuation rows -- five
    # times the intended length, and enough to dominate the window sampler.
    target_words = max(120, min(900, words))
    return SeedCard(
        doc_id=doc.doc_id,
        genre=doc.genre if doc.genre != "other" else "blog_personal",
        topic=topic_of(doc),
        entities=entities_of(doc),
        # The seed's own length and shape, which is what keeps length and structure
        # from carrying label information.
        target_words=target_words,
        headings=len(HEADING.findall(doc.text)),
        lists=len(BULLET.findall(doc.text)),
        publication_type=_publication_type(doc.genre),
        prefix=_prefix_of(doc.text, target_words),
        text=doc.text,
    )


def _publication_type(genre: Genre) -> str:
    return {
        "news": "a news article",
        "product_marketing": "a product page",
        "press_release": "a press release",
        "forum_comment": "a forum comment",
        "blog_personal": "a blog post",
        "technical_docs": "a documentation page",
        "encyclopedia": "an encyclopedia article",
        "academic_formal": "a formal report section",
        "other": "an article",
    }.get(genre, "an article")


def _prefix_of(text: str, target_words: int) -> str:
    """The opening ~30% of the target length, cut at a sentence boundary."""
    text = collapse_whitespace(text)
    target = max(40, int(target_words * 0.3))
    kept, total = [], 0
    for start, end in split_sentences(text):
        sentence = text[start:end]
        kept.append(sentence)
        total += count_words(sentence)
        if total >= target:
            break
    return " ".join(kept)


def plan_tasks(
    seeds: list[DocumentRow], models: list[ModelSpec], *, seed: int = 0
) -> list[GenerationTask]:
    """Assign each seed a model, style and sampling cell."""
    rng = random.Random(seed)
    model_options = [(m, m.weight) for m in models]
    style_options = [(s, w) for s, w in STYLE_MIX]
    tasks = []
    for doc in seeds:
        model = _pick(rng, model_options)
        style = _pick(rng, style_options)
        cell_name, cell = _cell_for(model, rng)
        tasks.append(
            GenerationTask(
                seed=doc,
                card=seed_card(doc),
                style=style,
                model=model,
                cell=cell,
                cell_name=cell_name,
                seed_cluster_id=doc.doc_id,
            )
        )
    return tasks


def _cell_for(model: ModelSpec, rng: random.Random) -> tuple[str, dict[str, object]]:
    if model.sampling:
        row = _pick(rng, [(entry, entry[3]) for entry in SAMPLING_GRID])
        cell: dict[str, object] = {"temperature": row[1], "top_p": row[2]}
        if model.thinking_off:
            cell["thinking_off"] = True
        return row[0], cell
    if model.effort:
        effort = _pick(rng, EFFORT_GRID)
        return f"effort_{effort}", {"effort": effort}
    return "plain", {}


def run_task(task: GenerationTask) -> DocumentRow | None:
    """Generate one document, clean it, and label it honestly."""
    rng = random.Random(f"{task.seed_cluster_id}:{task.style}")
    persona = None
    angle = None
    if task.style == "persona_instruction":
        persona = rng.choice(PERSONAS[task.card.genre])
    if task.style == "topic_prompt":
        angle = rng.choice(ANGLES)

    prompt = build(task.card, task.style, persona=persona, angle=angle)
    result = generate(task.model, prompt.system, prompt.user, task.card.target_words, task.cell)
    if not result.ok:
        return None

    cleaned = clean_generation(
        result.text,
        target_words=task.card.target_words,
        seed_had_heading=task.card.headings > 0,
    )
    if not cleaned.kept:
        return None

    if task.style == "continuation":
        return _continuation_row(task, prompt, cleaned.text, cleaned.ops)
    return _pure_ai_row(task, prompt, cleaned.text, cleaned.ops)


def _base_row(task: GenerationTask, prompt: Prompt, text: str, ops: list[str]) -> DocumentRow:
    from .sources import doc_id_for

    return DocumentRow(
        doc_id=doc_id_for("generated", f"{task.seed_cluster_id}:{prompt.prompt_id}"),
        text=text,
        source="generated",
        doc_class="ai",
        ai_fraction=1.0,
        genre=task.card.genre,
        genre_source="prompt",
        genre_intended=task.card.genre,
        n_words=count_words(text),
        seed_doc_id=task.seed.doc_id,
        seed_cluster_id=task.seed_cluster_id,
        generator=task.model.name,
        generator_provider=task.model.provider,
        prompt_style=task.style,
        prompt_version=prompt.prompt_id,
        sampling_params={"cell": task.cell_name, **{k: v for k, v in task.cell.items()}},
        hygiene_ops=ops,
        hard_negative_kind=task.seed.hard_negative_kind,
        host=task.seed.host,
    )


def _pure_ai_row(task: GenerationTask, prompt: Prompt, text: str, ops: list[str]) -> DocumentRow:
    row = _base_row(task, prompt, text, ops)
    row.ai_spans = [(0, len(collapse_whitespace(text)))]
    return row


def _continuation_row(
    task: GenerationTask, prompt: Prompt, text: str, ops: list[str]
) -> DocumentRow:
    """Human prefix plus AI body. The fraction is measured, not assumed."""
    prefix = collapse_whitespace(task.card.prefix)
    body = collapse_whitespace(text)
    joined = f"{prefix} {body}"
    row = _base_row(task, prompt, joined, ops)
    row.text = joined
    row.n_words = count_words(joined)
    row.ai_spans = [(len(prefix) + 1, len(joined))]
    row.ai_fraction = (len(joined) - len(prefix) - 1) / len(joined) if joined else 0.0
    row.mix_direction = "human_base"
    return row


def make_mixed_row(
    seed: DocumentRow, ai_row: DocumentRow, rng: random.Random
) -> DocumentRow | None:
    """Splice a human seed with its own AI rewrite at a drawn target fraction."""
    target = draw_target_fraction(rng)
    granularity: Granularity = "paragraph" if rng.random() < 0.7 else "prefix_suffix"
    mixed = splice(
        seed.text, ai_row.text, target, granularity=granularity, seed=rng.randrange(1 << 30)
    )
    if mixed is None:
        return None
    from .sources import doc_id_for

    return DocumentRow(
        doc_id=doc_id_for("mixed", f"{ai_row.doc_id}:{target:.3f}"),
        text=mixed.text,
        source="generated",
        doc_class="ai" if mixed.ai_fraction >= 0.5 else "human",
        ai_fraction=mixed.ai_fraction,
        ai_fraction_target=mixed.ai_fraction_target,
        ai_spans=mixed.ai_spans,
        genre=ai_row.genre,
        genre_source="prompt",
        n_words=count_words(mixed.text),
        seed_doc_id=seed.doc_id,
        seed_cluster_id=ai_row.seed_cluster_id,
        generator=ai_row.generator,
        generator_provider=ai_row.generator_provider,
        prompt_style=ai_row.prompt_style,
        mix_direction="human_base",
        hard_negative_kind=seed.hard_negative_kind,
        host=seed.host,
    )


def run_tasks(
    tasks: list[GenerationTask], *, workers: int = 24, mixed_rate: float = 0.35
) -> Iterator[DocumentRow]:
    """Generate concurrently. API calls are IO-bound, so threads are the right tool."""
    seeds_by_id = {t.seed.doc_id: t.seed for t in tasks}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task, row in zip(tasks, pool.map(run_task, tasks), strict=True):
            if row is None:
                continue
            yield row
            # A rewrite has an aligned human counterpart, which is what makes an
            # honest mixed document possible at all.
            if task.style == "rewrite":
                rng = random.Random(f"mix:{row.doc_id}")
                if rng.random() < mixed_rate:
                    mixed = make_mixed_row(seeds_by_id[task.seed.doc_id], row, rng)
                    if mixed is not None:
                        yield mixed
