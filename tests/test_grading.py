import json
from pathlib import Path

from local_inference_bench import grading as g

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "grading_parity.json").read_text())


def test_parity_with_typescript_original():
    for row in FIXTURE["rows"]:
        a = row["answer"]
        assert g.parse_citations(a) == row["citations"], a
        assert g.is_refusal(a) == row["isRefusal"], a
        assert g.normalize(a) == row["normalized"], a
        assert [g.citation_validity(a, set(r)) for r in FIXTURE["retrieved"]] == row["validity"], a
        assert [g.refusal_correctness(e, a) for e in ("answered", "refused")] == row["refusal"], a
        assert [g.citation_recall(c, a) for c in FIXTURE["expectedChunks"]] == row["recall"], a
        assert [g.fact_coverage(f, a) for f in FIXTURE["facts"]] == row["facts"], a


def test_has_thinking_ignores_empty_blocks():
    assert not g.has_thinking("<think>\n\n</think>Antwort")
    assert g.has_thinking("<think>Hm.</think>Antwort")
    assert not g.has_thinking("keine Tags")


def test_strip_think_removes_empty_and_full_blocks():
    assert g.strip_think("<think>\n\n</think>\n\nAntwort [chunk:x]") == "Antwort [chunk:x]"
    assert g.strip_think("<think>lang\nüberlegt</think>[KEINE_GRUNDLAGE]") == "[KEINE_GRUNDLAGE]"
    assert g.strip_think("ohne Block") == "ohne Block"
