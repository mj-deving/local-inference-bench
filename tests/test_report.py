"""Report aggregation on synthetic raw data: no model, no server."""

import copy
import json
from pathlib import Path

import pytest

from local_inference_bench.report import agreement, cost, polarity, polarity_flips, quality, summarize

CASES = {
    "a": {"id": "a", "set": "golden", "expected_outcome": "answered", "expected_chunks": ["c1"],
          "expected_facts": ["informiert"], "retrieved": ["c1", "c2"]},
    "b": {"id": "b", "set": "golden", "expected_outcome": "refused", "expected_chunks": [], "retrieved": ["c2"]},
    "h": {"id": "h", "set": "heldout", "expected_outcome": "refused", "expected_chunks": [], "retrieved": ["c3"]},
}


def _row(id_, text, t0, ttft, n, dt, reasoning="", prompt=100):
    timing = {"t_send": t0, "t_first": t0 + ttft, "t_last": t0 + ttft + dt, "t_done": t0 + ttft + dt + 0.01,
              "output_tokens": n}
    return {"id": id_, "text": text, "reasoning": reasoning, "timing": timing, "ttft": ttft,
            "e2e": ttft + dt + 0.01, "decode_tps": (n - 1) / dt, "output_tokens": n, "prompt_tokens": prompt}


def _raw(answer_b="[KEINE_GRUNDLAGE] x"):
    rows = [_row("a", "<think>\n\n</think>Wird informiert [chunk:c1].", 0, 0.5, 11, 1.0),
            _row("b", answer_b, 2, 0.4, 3, 0.1),
            _row("h", "Antwort [chunk:c9]", 4, 0.6, 21, 2.0)]
    sweep = [{"concurrency": c, "rep": r, "results": [_row("a", "x", 0, 0.5, 100, 1.0), _row("b", "x", 0, 0.5, 100, 1.0)]}
             for c in (1, 8) for r in range(3)]
    return {"config": {"name": "t"}, "settings": {"max_tokens": 512},
            "cold": [{"launch_to_first_token_s": s, "ready_s": 1.0, "peak_rss_mb": 10, "peak_footprint_mb": 20}
                     for s in (9.0, 3.0, 3.0)],
            "warm": [{"rep": r, "results": copy.deepcopy(rows)} for r in range(3)], "warm_memory": {"peak_rss_mb": 5, "peak_footprint_mb": 6},
            "sweep": sweep}


def test_quality_grades_with_ported_rules_and_strips_think():
    q = quality(_raw()["warm"][0]["results"], CASES)
    assert q["golden"]["refusal_correctness"] == 1.0
    assert q["golden"]["citation_validity"] == 1.0
    assert q["golden"]["fact_coverage"] == 1.0
    assert q["golden"]["out_of_corpus_refused"] == 1
    assert q["heldout"]["refusal_correctness"] == 0.0  # answered a refuse case
    assert q["heldout"]["citation_validity"] == 0.0  # cited a chunk that was not retrieved


def test_summarize_shapes_and_values():
    s = summarize(_raw(), CASES)
    assert s["cold"]["launch_to_first_token_s"]["mean"] == pytest.approx(5.0)
    assert s["warm"]["ttft_p50"]["stdev"] == 0.0  # identical reps
    assert s["quality_stable_across_reps"] is True
    assert s["identical_text_share_across_reps"] == 1.0
    assert s["same_decision_share_across_reps"] == 1.0
    assert s["prefill_share_of_e2e_median"] == pytest.approx(0.5 / 1.51)
    assert s["think_share"] == 0.0  # empty <think></think> is not thinking
    assert s["sweep"]["1"]["tok_per_s"]["mean"] == pytest.approx(200 / 1.51)


def test_think_share_counts_reasoning_field():
    raw = _raw()
    raw["warm"][0]["results"][0]["reasoning"] = "überlege"
    assert summarize(raw, CASES)["think_share"] == pytest.approx(1 / 9)


def test_polarity_reads_leading_ja_nein_only():
    assert polarity("Ja, verboten [chunk:x]") == "ja"
    assert polarity("Nein. Nicht verboten") == "nein"
    assert polarity("Jahre später") is None
    assert polarity("") is None


def test_polarity_flips_find_disagreeing_yes_no():
    s1 = summarize(_raw(answer_b="Ja, verboten [chunk:c2]"), CASES)
    s2 = summarize(_raw(answer_b="Nein, nicht verboten [chunk:c2]"), CASES)
    assert polarity_flips({"x": s1, "y": s2}) == {"x vs y": ["b"]}
    assert polarity_flips({"x": s1, "z": s1}) == {"x vs z": []}


def test_yes_no_instability_across_reps():
    raw = _raw(answer_b="Ja, verboten [chunk:c2]")
    raw["warm"][2]["results"][1]["text"] = "Nein, nicht verboten [chunk:c2]"
    assert summarize(raw, CASES)["yes_no_unstable_across_reps"] == ["b"]
    assert summarize(_raw(), CASES)["yes_no_unstable_across_reps"] == []


def test_agreement_counts_same_decisions():
    s1, s2 = summarize(_raw(), CASES), summarize(_raw(answer_b="Antwort [chunk:c2]"), CASES)
    assert agreement({"x": s1, "y": s2})["x vs y"] == pytest.approx(2 / 3)


def test_cost_picks_best_concurrency_and_keeps_assumptions():
    assumptions = json.loads((Path(__file__).parents[1] / "assumptions.json").read_text())
    c = cost({"t": summarize(_raw(), CASES)}, assumptions)
    m = c["configs"]["t"]
    assert c["assumptions"] is assumptions
    assert m["measured"]["mean_prompt_tokens"] == 100
    be = m["break_even_blended_eur_per_m_tokens"]["energy_only"]
    assert be == pytest.approx(m["local_eur_per_1k"]["energy_only"] / (1000 * (100 + 100)) * 1e6)
