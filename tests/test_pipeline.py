"""CPU-only tests for the parts of the pipeline that need no model.

They pin the properties the experiment depends on: prompts rebuild exactly,
only the history is translated, every language version keeps the same
conversations, and scoring treats ties as wrong.
"""

import json
from pathlib import Path

import pytest

import pcheck_ml as pm

ROOT = Path(__file__).resolve().parent.parent

HEADER = "##Task\nWrite the user's profile.\n\n"
FOOTER = "[User’s General Profile] (GP) (Write only the profile):"
INTERACTIONS = [
    {"query": "Suggest a weekend trip.", "chosen": "Try Busan. It has beaches.",
     "reject": "Go anywhere."},
    {"query": "How do I cook rice?", "chosen": "- Rinse it.\n- Use 1.2 cups of water.",
     "reject": "Just boil it."},
    {"query": "What is 2+2?", "chosen": "4", "reject": "5"},
]


def gp_input(interactions=INTERACTIONS):
    parsed = {"header": HEADER + "[Interaction History of User]\n", "sep": "\n", "footer": FOOTER}
    return pm.build_gp_input(parsed, interactions)


def fake_caches(interactions=INTERACTIONS):
    """A translation cache that tags every segment, so translated text is easy to spot."""
    segs = pm.segments_to_translate(
        [it[k] for it in interactions for k in ("query", "chosen", "reject")])
    return {"en": {}, **{lang: {s: f"<{lang}>{s}" for s in segs} for lang in pm.NLLB_CODES}}


def test_gp_prompt_round_trips_exactly():
    text = gp_input()
    parsed = pm.parse_gp_input(text)
    assert parsed["interactions"] == INTERACTIONS
    assert pm.build_gp_input(parsed, parsed["interactions"]) == text


def test_dropping_old_conversations_keeps_numbering():
    parsed = pm.parse_gp_input(gp_input())
    rebuilt = pm.build_gp_input(parsed, parsed["interactions"][1:], start_index=2)
    assert "<previous interaction 1>" not in rebuilt
    assert rebuilt.count("<previous interaction") == 2
    assert "<previous interaction 2>:\n[User Query]:How do I cook rice?" in rebuilt


def test_english_segments_join_back_unchanged():
    for it in INTERACTIONS:
        for text in it.values():
            pieces = pm.split_segments(text)
            cache = {s: s for kind, s in pieces if kind == "tr"}
            assert pm.join_segments(pieces, "en", cache) == text


def test_list_markers_and_symbols_are_not_translated():
    pieces = pm.split_segments("- Rinse it.\n2+2=4")
    assert ("keep", "- ") in pieces
    assert ("keep", "2+2=4") in pieces
    assert ("tr", "Rinse it.") in pieces


def test_long_lines_are_split_below_the_limit():
    line = " ".join(["This is one sentence."] * 60)
    pieces = pm.split_segments(line)
    assert all(len(s) <= pm.MAX_SEG_CHARS for kind, s in pieces if kind == "tr")
    cache = {s: s for kind, s in pieces if kind == "tr"}
    assert pm.join_segments(pieces, "en", cache) == line


def test_cjk_drops_the_space_between_chunks():
    pieces = [("tr", "A."), ("keep", " "), ("tr", "B.")]
    cache = {"A.": "甲。", "B.": "乙。"}
    assert pm.join_segments(pieces, "zh", cache) == "甲。乙。"
    assert pm.join_segments(pieces, "ms", cache) == "甲。 乙。"


def test_every_version_keeps_the_same_conversations():
    caches = fake_caches()
    for version in pm.VERSIONS:
        out = pm.version_interactions(INTERACTIONS, version, caches)
        assert len(out) == len(INTERACTIONS)
    assert pm.version_interactions(INTERACTIONS, "en", caches) == INTERACTIONS
    korean = pm.version_interactions(INTERACTIONS, "ko", caches)
    assert korean[0]["query"] == "<ko>Suggest a weekend trip."


def test_mixed_version_cycles_languages_by_conversation():
    caches = fake_caches()
    mixed = pm.version_interactions(INTERACTIONS, "mixed", caches)
    assert mixed[0] == INTERACTIONS[0]  # MIX_CYCLE starts with English
    assert mixed[1]["query"].startswith("<zh>")
    assert mixed[2]["query"].startswith("<ms>")


@pytest.mark.parametrize("text, lang", [
    ("사용자는 짧은 답변을 선호합니다.", "ko"),
    ("用户喜欢简短的回答。", "zh"),
    ("Pengguna lebih suka jawapan yang pendek dan jelas untuk anda.", "ms"),
    ("The user prefers short answers with examples.", "en"),
    ("", "empty"),
])
def test_detect_lang(text, lang):
    assert pm.detect_lang(text) == lang


def test_fill_leaves_json_braces_alone():
    out = pm.fill('{GP} -> {"score": <1-10>}', GP="likes tea")
    assert out == 'likes tea -> {"score": <1-10>}'


def test_ties_count_as_wrong_and_unparsed_pairs_are_counted():
    rows = [
        {"instance_id": 1, "version": "en", "rt": "chosen", "s": 8},
        {"instance_id": 1, "version": "en", "rt": "reject", "s": 3},
        {"instance_id": 2, "version": "en", "rt": "chosen", "s": 5},
        {"instance_id": 2, "version": "en", "rt": "reject", "s": 5},
        {"instance_id": 3, "version": "en", "rt": "chosen", "s": None},
        {"instance_id": 3, "version": "en", "rt": "reject", "s": 2},
    ]
    outcomes, failed = pm.pair_outcomes(rows, lambda r: r["s"])
    assert outcomes[(1, "en")] == {"correct": 1, "tie": 0}
    assert outcomes[(2, "en")] == {"correct": 0, "tie": 1}
    assert (3, "en") not in outcomes
    assert failed["en"] == 1


def test_summary_reports_paired_difference_against_english():
    outcomes = {}
    for iid in range(1, 11):
        outcomes[(iid, "en")] = {"correct": 1, "tie": 0}
        outcomes[(iid, "ko")] = {"correct": int(iid > 3), "tie": 0}
    rows = pm.summarize(outcomes, {}, n_expected=10)
    assert rows["en"]["acc"] == 1.0
    assert rows["en"]["diff_vs_en"] is None
    assert rows["ko"]["acc"] == pytest.approx(0.7)
    assert rows["ko"]["diff_vs_en"] == pytest.approx(-0.3)
    lo, hi = rows["ko"]["diff_ci"]
    assert lo <= -0.3 <= hi <= 0


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = pm.bootstrap_ci([1] * 60 + [0] * 40)
    assert lo < 0.6 < hi
    assert pm.bootstrap_ci([]) == (None, None)


def test_notebook_writes_out_the_same_pipeline():
    """The notebook embeds pcheck_ml.py; the two copies must not drift."""
    nb = json.loads((ROOT / "pcheck_multilingual_kaggle.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    embedded = next(c for c in cells if c.startswith("%%writefile pcheck_ml.py\n"))
    embedded = embedded.split("\n", 1)[1]
    assert embedded.rstrip() == (ROOT / "pcheck_ml.py").read_text(encoding="utf-8").rstrip()
