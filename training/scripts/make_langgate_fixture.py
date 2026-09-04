"""Emit fixtures/langgate.json -- the language-gate agreement fixture scope.md 7.2 asks for.

    uv run --extra corpus python scripts/make_langgate_fixture.py \
        --windows <path to a processed windows shard>.jsonl.zst

The gate is explicitly NOT a coupling point: the extension bundles a franc-class detector
and the harvest uses py3langid, and they need not agree block for block. scope.md 7.2 asks
for something narrower and more useful:

    "asserts both that the two gates agree on acceptance, and, more importantly, that the
     cases where they disagree are not systematically non-native."

That second half is the whole point. franc is weak on short non-native English, and an
"unknown -> skip" extension gate would silently never score exactly the prose scope.md 4.2
over-samples as its hardest negative -- trained on, never scored, and invisible in every
false-positive number the project has. This fixture is what makes that measurable rather
than merely asserted.

Non-native English is drawn from the corpus itself (`hard_negative_kind == non_native_forum`)
rather than written by hand, because the whole question is whether a detector trips on real
second-language prose, and constructed examples would be answering an easier question.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "training" / "src"))

from slopmarker.corpus.lang import detect  # noqa: E402
from slopmarker.data.normalize import collapse_whitespace, count_words  # noqa: E402

MIN_WORDS = 40
MAX_WORDS = 220
PER_GROUP = 60

# Crawled forum text goes into a tracked file, so drop anything carrying a contact handle,
# an address or a link. This is a privacy filter, not a quality one.
_PII = re.compile(
    r"(https?://|www\.|@[A-Za-z0-9_]{3,}|[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s().-]{8,}\d)"
)

# Non-English text. The corpus is English-gated by construction, so it has none, and these
# are public-domain openings plus plain declarative prose in scripts the detector should
# never mistake for English.
NON_ENGLISH: dict[str, str] = {
    "fr": (
        "Longtemps, je me suis couché de bonne heure. Parfois, à peine ma bougie éteinte, "
        "mes yeux se fermaient si vite que je n'avais pas le temps de me dire: Je m'endors. "
        "Et, une demi-heure après, la pensée qu'il était temps de chercher le sommeil "
        "m'éveillait; je voulais poser le volume que je croyais avoir encore dans les mains "
        "et souffler ma lumière."
    ),
    "de": (
        "Als Gregor Samsa eines Morgens aus unruhigen Träumen erwachte, fand er sich in "
        "seinem Bett zu einem ungeheueren Ungeziefer verwandelt. Er lag auf seinem "
        "panzerartig harten Rücken und sah, wenn er den Kopf ein wenig hob, seinen "
        "gewölbten, braunen, von bogenförmigen Versteifungen geteilten Bauch."
    ),
    "es": (
        "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo "
        "que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y "
        "galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, "
        "duelos y quebrantos los sábados, lentejas los viernes."
    ),
    "it": (
        "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura, ché la "
        "diritta via era smarrita. Ahi quanto a dir qual era è cosa dura esta selva selvaggia "
        "e aspra e forte che nel pensier rinova la paura. Tant' è amara che poco è più morte."
    ),
    "pt": (
        "Algum tempo hesitei se devia abrir estas memórias pelo princípio ou pelo fim, isto "
        "é, se poria em primeiro lugar o meu nascimento ou a minha morte. Suposto o uso "
        "vulgar seja começar pelo nascimento, duas considerações me levaram a adotar "
        "diferente método."
    ),
    "nl": (
        "Het was een zeer warme dag in het midden van de zomer en de straten van de stad "
        "lagen er verlaten bij. De winkels waren gesloten en de mensen bleven binnen waar het "
        "koeler was. Pas tegen de avond kwam er weer beweging in de buurt en gingen de "
        "deuren langzaam open."
    ),
    "ru": (
        "Все счастливые семьи похожи друг на друга, каждая несчастливая семья несчастлива "
        "по-своему. Все смешалось в доме Облонских. Жена узнала, что муж был в связи с "
        "бывшею в их доме француженкою-гувернанткой, и объявила мужу, что не может жить с "
        "ним в одном доме."
    ),
    "pl": (
        "Litwo, Ojczyzno moja! ty jesteś jak zdrowie. Ile cię trzeba cenić, ten tylko się "
        "dowie, kto cię stracił. Dziś piękność twą w całej ozdobie widzę i opisuję, bo "
        "tęsknię po tobie. Panno święta, co jasnej bronisz Częstochowy i w Ostrej świecisz "
        "Bramie!"
    ),
    "sv": (
        "Det var en gång en liten flicka som bodde långt uppe i norr där vintrarna är långa "
        "och mörka. Hon hade aldrig sett havet men hon hade hört talas om det av sin farmor "
        "som en gång i sin ungdom hade rest ända ner till kusten och tillbaka igen."
    ),
    "tr": (
        "Bir zamanlar küçük bir kasabada yaşayan yaşlı bir adam vardı. Her sabah erkenden "
        "kalkar, bahçesindeki ağaçları sular ve sonra uzun bir yürüyüşe çıkardı. Kasabadaki "
        "herkes onu tanır ve selam verirdi çünkü yıllardır aynı yolları yürüyordu."
    ),
    "ja": (
        "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗い"
        "じめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて"
        "人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族"
        "であったそうだ。"
    ),
    "zh": (
        "从前有一个小村庄，村子后面有一座很高的山。山上长满了松树，每到秋天，风吹过树林"
        "的时候就会发出很大的声音。村里的孩子们常常爬到半山腰去玩，然后在天黑之前赶回家"
        "里吃晚饭。"
    ),
    "ar": (
        "كان يا ما كان في قديم الزمان رجل عجوز يعيش في قرية صغيرة قرب النهر. كان يخرج كل "
        "صباح إلى حقله ويعمل حتى غروب الشمس ثم يعود إلى بيته متعبا ولكنه سعيد. وكان أهل "
        "القرية يحبونه كثيرا لأنه كان يساعد كل من يحتاج إليه."
    ),
    "ko": (
        "옛날 옛적에 작은 마을에 한 노인이 살았습니다. 그는 매일 아침 일찍 일어나 밭에 "
        "나가 일을 했고 해가 질 때까지 쉬지 않았습니다. 마을 사람들은 모두 그를 존경했고 "
        "어려운 일이 생기면 언제나 그를 찾아가 조언을 구하곤 했습니다."
    ),
    "hi": (
        "एक समय की बात है, एक छोटे से गाँव में एक किसान रहता था। वह हर सुबह जल्दी उठकर अपने "
        "खेतों में काम करने जाता था और शाम तक वहीं रहता था। गाँव के सभी लोग उसकी मेहनत की "
        "तारीफ करते थे क्योंकि वह कभी आराम नहीं करता था।"
    ),
}


def usable(text: str) -> bool:
    words = count_words(text)
    return MIN_WORDS <= words <= MAX_WORDS and _PII.search(text) is None


def load_windows(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """(non-native English, native English) human windows from a processed shard."""
    import zstandard

    non_native: list[dict[str, object]] = []
    native: list[dict[str, object]] = []
    with open(path, "rb") as fh:
        stream = zstandard.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(stream, encoding="utf-8"):
            row = json.loads(line)
            if row.get("doc_class") != "human":
                continue
            text = collapse_whitespace(str(row.get("text", "")))
            if not usable(text):
                continue
            kind = row.get("hard_negative_kind")
            record = {"text": text, "genre": row.get("genre"), "kind": kind}
            if kind == "non_native_forum":
                non_native.append(record)
            elif kind is None:
                native.append(record)
    return non_native, native


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, required=True, help="processed windows .jsonl.zst")
    ap.add_argument("--out", type=Path, default=REPO / "fixtures" / "langgate.json")
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    non_native, native = load_windows(args.windows)
    rng = random.Random(args.seed)
    rng.shuffle(non_native)
    rng.shuffle(native)

    cases: list[dict[str, object]] = []

    def add(text: str, group: str, note: str | None = None) -> None:
        lang, score = detect(text)
        cases.append(
            {
                "group": group,
                "text": text,
                "note": note,
                "python": {"lang": lang, "score": round(score, 6), "is_english": lang == "en" and score >= 0.85},
            }
        )

    for row in non_native[:PER_GROUP]:
        add(str(row["text"]), "non_native_english", str(row["genre"]))
    for row in native[:PER_GROUP]:
        add(str(row["text"]), "native_english", str(row["genre"]))
    for code, text in NON_ENGLISH.items():
        add(collapse_whitespace(text), "non_english", code)

    payload = {
        "version": 1,
        "note": (
            "scope.md 7.2. The extension's franc-class gate and the harvest's py3langid gate "
            "need not agree block for block -- this is not a coupling point. What is asserted "
            "is that they agree on acceptance, and that where they disagree the disagreement "
            "is not concentrated on non-native English, which is the population scope.md 2 "
            "names as the dominant false-positive risk. Non-native and native English are real "
            "corpus windows (hard_negative_kind == non_native_forum, and human windows with "
            "none); non-English is public-domain prose, since the corpus is English-gated and "
            "contains none."
        ),
        "min_score": 0.85,
        "counts": {
            group: sum(1 for c in cases if c["group"] == group)
            for group in ("non_native_english", "native_english", "non_english")
        },
        "cases": cases,
    }
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    for group in ("non_native_english", "native_english", "non_english"):
        rows = [c for c in cases if c["group"] == group]
        accepted = sum(1 for c in rows if c["python"]["is_english"])  # type: ignore[index]
        print(f"  {group:20} n={len(rows):3}  python accepts {accepted}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
