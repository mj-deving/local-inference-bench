"""Aggregate results/raw/*.json into results/summary.json, results/cost.json and results/summary.md.

Pure functions (summarize, quality, cost) are unit-tested on synthetic raw data.
"""

import json
import statistics
from pathlib import Path

from . import grading as g
from .metrics import RequestTiming, cost_per_1k_answers, hosted_cost_per_1k_answers, percentile, spread, throughput

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def _sp(values: list[float]) -> dict:
    s = spread(values)
    return {"mean": s.mean, "stdev": s.stdev, "min": s.min, "max": s.max, "n": s.n}


def _timing(r: dict) -> RequestTiming:
    return RequestTiming(**r["timing"])


def quality(rows: list[dict], cases: dict[str, dict]) -> dict:
    """Grade one pass over the cases, split by set. Uses the ported grading rules unchanged."""
    out = {}
    for set_name in ("golden", "heldout"):
        sel = [(cases[r["id"]], g.strip_think(r["text"])) for r in rows if cases[r["id"]]["set"] == set_name]
        if not sel:
            continue
        recall = [x for c, a in sel if (x := g.citation_recall(c["expected_chunks"], a)) is not None]
        facts = [x for c, a in sel if (x := g.fact_coverage(c.get("expected_facts"), a)) is not None]
        ooc = [(c, a) for c, a in sel if c["expected_outcome"] == "refused"]
        out[set_name] = {
            "cases": len(sel),
            "refusal_correctness": statistics.fmean(g.refusal_correctness(c["expected_outcome"], a) for c, a in sel),
            "citation_validity": statistics.fmean(g.citation_validity(a, set(c["retrieved"])) for c, a in sel),
            "out_of_corpus_refused": sum(g.is_refusal(a) for _, a in ooc),
            "out_of_corpus_total": len(ooc),
            "wrongly_refused": sum(g.is_refusal(a) for c, a in sel if c["expected_outcome"] == "answered"),
            "citation_recall": statistics.fmean(recall) if recall else None,
            "fact_coverage": statistics.fmean(facts) if facts else None,
        }
    return out


def summarize(raw: dict, cases: dict[str, dict]) -> dict:
    max_tokens = raw["settings"]["max_tokens"]
    warm = raw["warm"]
    per_rep = []
    for rep in warm:
        rs = rep["results"]
        ttft = [r["ttft"] for r in rs if r["ttft"] is not None]
        e2e = [r["e2e"] for r in rs]
        tps = [r["decode_tps"] for r in rs if r["decode_tps"] is not None]
        per_rep.append({"ttft_p50": percentile(ttft, 50), "ttft_p95": percentile(ttft, 95),
                        "e2e_p50": percentile(e2e, 50), "e2e_p95": percentile(e2e, 95),
                        "decode_tps_median": statistics.median(tps), "quality": quality(rs, cases)})
    texts = [{r["id"]: r["text"] for r in rep["results"]} for rep in warm]
    first = warm[0]["results"]
    sweep = {}
    for c in sorted({s["concurrency"] for s in raw["sweep"]}):
        reps = [s["results"] for s in raw["sweep"] if s["concurrency"] == c]
        tps = [throughput([_timing(r) for r in rs]) for rs in reps]
        sweep[str(c)] = {
            "tok_per_s": _sp([t["tok_per_s"] for t in tps]),
            "req_per_s": _sp([t["req_per_s"] for t in tps]),
            "e2e_p95": _sp([percentile([r["e2e"] for r in rs], 95) for rs in reps]),
            "ttft_p95": _sp([percentile([r["ttft"] for r in rs if r["ttft"] is not None], 95) for rs in reps]),
            "mean_output_tokens": statistics.fmean(r["output_tokens"] for rs in reps for r in rs),
            "mean_prompt_tokens": statistics.fmean(r["prompt_tokens"] for rs in reps for r in rs),
        }
    all_warm = [r for rep in warm for r in rep["results"]]
    return {
        "config": raw["config"],
        "cold": {
            "launch_to_first_token_s": _sp([c["launch_to_first_token_s"] for c in raw["cold"]]),
            "ready_s": _sp([c["ready_s"] for c in raw["cold"]]),
            "peak_rss_mb": max(c["peak_rss_mb"] for c in raw["cold"]),
            "peak_footprint_mb": max(c["peak_footprint_mb"] for c in raw["cold"]),
        },
        "warm": {k: _sp([p[k] for p in per_rep]) for k in ("ttft_p50", "ttft_p95", "e2e_p50", "e2e_p95", "decode_tps_median")},
        "memory": raw["warm_memory"],
        "quality": per_rep[0]["quality"],
        "quality_stable_across_reps": all(p["quality"] == per_rep[0]["quality"] for p in per_rep),
        # Reps differ only in the request id line, so this measures sensitivity to an irrelevant
        # prompt change at temperature 0, not runtime nondeterminism.
        "identical_text_share_across_reps": statistics.fmean(
            all(t[i] == texts[0][i] for t in texts) for i in texts[0]),
        "same_decision_share_across_reps": statistics.fmean(
            len({g.is_refusal(g.strip_think(t[i])) for t in texts}) == 1 for i in texts[0]),
        "yes_no_unstable_across_reps": sorted(
            i for i in texts[0]
            if len({p for t in texts if (p := polarity(g.strip_think(t[i])))}) > 1),
        "prefill_share_of_e2e_median": statistics.median(
            r["ttft"] / r["e2e"] for r in all_warm if r["ttft"] is not None and r["e2e"] > 0),
        "think_share": statistics.fmean(bool(r.get("reasoning", "").strip() or g.has_thinking(r["text"]))
                                        for r in all_warm),
        "truncated_share": statistics.fmean(r["output_tokens"] >= max_tokens for r in all_warm),
        "mean_prompt_tokens": statistics.fmean(r["prompt_tokens"] for r in first if r["prompt_tokens"] is not None),
        "mean_output_tokens": statistics.fmean(r["output_tokens"] for r in first),
        "prompt_tokens_by_case": {r["id"]: r["prompt_tokens"] for r in first},
        "decisions_by_case": {r["id"]: g.is_refusal(g.strip_think(r["text"])) for r in first},
        "polarity_by_case": {r["id"]: polarity(g.strip_think(r["text"])) for r in first},
        "sweep": sweep,
    }


def polarity(answer: str) -> str | None:
    """"ja"/"nein" when an answer opens with it, else None. The grading rules cannot see a flipped
    yes/no: a wrong "Nein" with a valid citation scores like the right "Ja"."""
    first = answer.strip().split(maxsplit=1)[0].strip(",.:!").lower() if answer.strip() else ""
    return first if first in ("ja", "nein") else None


def polarity_flips(summaries: dict[str, dict]) -> dict:
    """Per pair, the case ids where both answers open with ja/nein and disagree."""
    names = sorted(summaries)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = summaries[a]["polarity_by_case"], summaries[b]["polarity_by_case"]
            out[f"{a} vs {b}"] = sorted(k for k in pa if pa[k] and pb.get(k) and pa[k] != pb[k])
    return out


def agreement(summaries: dict[str, dict]) -> dict:
    """Pairwise share of cases with the same answer/refuse decision (warm rep 0)."""
    names = sorted(summaries)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da, db = summaries[a]["decisions_by_case"], summaries[b]["decisions_by_case"]
            out[f"{a} vs {b}"] = statistics.fmean(da[k] == db[k] for k in da)
    return out


def cost(summaries: dict[str, dict], assumptions: dict) -> dict:
    A = {k: v["value"] for k, v in assumptions.items() if not k.startswith("_")}
    out = {"assumptions": assumptions, "configs": {}}
    for name, s in summaries.items():
        best_c, best = max(s["sweep"].items(), key=lambda kv: kv[1]["tok_per_s"]["mean"])
        # Prompt and output tokens come from the same sweep requests as the throughput.
        tok_s, out_tok, in_tok = best["tok_per_s"]["mean"], best["mean_output_tokens"], best["mean_prompt_tokens"]
        base = dict(tok_per_s=tok_s, mean_output_tokens=out_tok, power_w=A["power_w"], eur_per_kwh=A["eur_per_kwh"],
                    hardware_eur=A["hardware_eur"], amortization_years=A["amortization_years"])
        local = {sc: cost_per_1k_answers(**base, utilization=u) for sc, u in A["utilization_scenarios"].items()}
        energy_only = local[next(iter(local))]["energy_eur"]
        tokens_per_1k = 1000 * (in_tok + out_tok)
        out["configs"][name] = {
            "measured": {"best_concurrency": int(best_c), "tok_per_s": tok_s, "mean_output_tokens": out_tok,
                         "mean_prompt_tokens": in_tok, "answers_per_hour": 3600 * tok_s / out_tok},
            "local_eur_per_1k": {"energy_only": energy_only, **{sc: v["total_eur"] for sc, v in local.items()}},
            "hosted_eur_per_1k": {p: hosted_cost_per_1k_answers(input_tokens=in_tok, output_tokens=out_tok,
                                                                eur_per_m_in=v["in"], eur_per_m_out=v["out"])
                                  for p, v in A["hosted_price_points_eur_per_m_tokens"].items()},
            "break_even_blended_eur_per_m_tokens": {k: v / tokens_per_1k * 1_000_000 for k, v in
                                                    {"energy_only": energy_only,
                                                     **{sc: v["total_eur"] for sc, v in local.items()}}.items()},
        }
    return out


def _f(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def markdown(summaries: dict[str, dict], agree: dict, costs: dict, prompt_diff: int | None = None,
             flips: dict | None = None) -> str:
    L = ["# Summary (generated by `uv run bench report`, do not edit)", "",
         "## Speed and quality per config (warm, concurrency 1, mean ± sd over 3 reps)", "",
         "| config | TTFT p50 s | TTFT p95 s | E2E p50 s | E2E p95 s | decode tok/s | refusal ok golden | refusal ok held-out | citation valid golden | citation valid held-out |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summaries.items():
        w, q = s["warm"], s["quality"]
        cell = lambda k, nd=2: f"{_f(w[k]['mean'], nd)} ± {_f(w[k]['stdev'], nd)}"
        L.append(f"| {n} | {cell('ttft_p50')} | {cell('ttft_p95')} | {cell('e2e_p50')} | {cell('e2e_p95')} | "
                 f"{cell('decode_tps_median', 1)} | {_f(100 * q['golden']['refusal_correctness'])} % | "
                 f"{_f(100 * q['heldout']['refusal_correctness'])} % | {_f(100 * q['golden']['citation_validity'])} % | "
                 f"{_f(100 * q['heldout']['citation_validity'])} % |")
    L += ["", "## Throughput under concurrency (output tok/s, mean ± sd over 3 reps, 16 requests per rep)", "",
          "| config | c=1 | c=2 | c=4 | c=8 | E2E p95 at c=8 s |", "|---|---|---|---|---|---|"]
    for n, s in summaries.items():
        sw = s["sweep"]
        L.append(f"| {n} | " + " | ".join(f"{_f(sw[c]['tok_per_s']['mean'])} ± {_f(sw[c]['tok_per_s']['stdev'])}"
                                          for c in ("1", "2", "4", "8")) + f" | {_f(sw['8']['e2e_p95']['mean'], 2)} |")
    L += ["", "## Cold start, memory, hygiene", "",
          "| config | launch→first token s (3 reps, page cache primed; min–max) | peak RSS MB | peak footprint MB | think share | truncated share | identical text across reps | same decision across reps | prefill share of E2E | mean prompt tokens |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summaries.items():
        c = s["cold"]
        L.append(f"| {n} | {_f(c['launch_to_first_token_s']['mean'], 2)} ± {_f(c['launch_to_first_token_s']['stdev'], 2)} "
                 f"({_f(c['launch_to_first_token_s']['min'], 2)}–{_f(c['launch_to_first_token_s']['max'], 2)}) | "
                 f"{max(c['peak_rss_mb'], s['memory']['peak_rss_mb'])} | "
                 f"{max(c['peak_footprint_mb'], s['memory']['peak_footprint_mb'])} | {_f(100 * s['think_share'])} % | "
                 f"{_f(100 * s['truncated_share'])} % | {_f(100 * s['identical_text_share_across_reps'])} % | "
                 f"{_f(100 * s['same_decision_share_across_reps'])} % | {_f(100 * s['prefill_share_of_e2e_median'])} % | "
                 f"{_f(s['mean_prompt_tokens'], 0)} |")
    L += ["", "## Quality detail (warm rep 0)", "",
          "| config | set | refusal ok | out-of-corpus refused | wrongly refused | citation valid | citation recall | fact coverage |",
          "|---|---|---|---|---|---|---|---|"]
    for n, s in summaries.items():
        for set_name, q in s["quality"].items():
            pct = lambda x: "n/a" if x is None else f"{_f(100 * x)} %"
            L.append(f"| {n} | {set_name} | {pct(q['refusal_correctness'])} | {q['out_of_corpus_refused']} / "
                     f"{q['out_of_corpus_total']} | {q['wrongly_refused']} | {pct(q['citation_validity'])} | "
                     f"{pct(q['citation_recall'])} | {pct(q['fact_coverage'])} |")
    L += ["", f"Largest prompt-token difference for one case across configs: {prompt_diff} tokens.", ""]
    L += ["", "## Answer/refuse agreement between configs (48 cases)", "", "| pair | same decision |", "|---|---|"]
    L += [f"| {k} | {_f(100 * v)} % |" for k, v in agree.items()]
    L += ["", "## Yes/no answers that change across the 3 repetitions of one config", "",
          "| config | unstable cases |", "|---|---|"]
    L += [f"| {n} | {len(s['yes_no_unstable_across_reps'])}: {', '.join(s['yes_no_unstable_across_reps']) or '-'} |"
          for n, s in summaries.items()]
    L += ["", "## Yes/no flips the grading rules cannot see (both answers open with Ja/Nein and disagree)", "",
          "| pair | flipped cases |", "|---|---|"]
    L += [f"| {k} | {len(v)}: {', '.join(v) or '-'} |" for k, v in (flips or {}).items()]
    L += ["", "## Cost per 1000 answers (EUR; see assumptions.json, these are assumptions)", "",
          "| config | best c | answers/h | energy only | amortized 24/7 | amortized office hours | break-even EUR/M tokens (24/7) | hosted small | hosted mid |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, c in costs["configs"].items():
        m, lo, h, be = c["measured"], c["local_eur_per_1k"], c["hosted_eur_per_1k"], c["break_even_blended_eur_per_m_tokens"]
        L.append(f"| {n} | {m['best_concurrency']} | {_f(m['answers_per_hour'], 0)} | {_f(lo['energy_only'], 3)} | "
                 f"{_f(lo['dedicated_24_7'], 3)} | {_f(lo['busy_office_hours'], 3)} | {_f(be['dedicated_24_7'], 3)} | "
                 f"{_f(h['small_hosted_model'], 3)} | {_f(h['mid_hosted_model'], 3)} |")
    return "\n".join(L) + "\n"


def report() -> None:
    cases = {}
    for line in (ROOT / "data" / "cases.jsonl").read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            cases[c["id"]] = c
    order = ["ollama-q4", "llamacpp-q4", "mlx-4bit", "ollama-q8", "llamacpp-q8", "mlx-8bit"]
    raws = {p.stem: json.loads(p.read_text()) for p in sorted((RESULTS / "raw").glob("*.json"))}
    summaries = {n: summarize(raws[n], cases) for n in order if n in raws}
    agree = agreement(summaries)
    flips = polarity_flips(summaries)
    by_case = [s["prompt_tokens_by_case"] for s in summaries.values()]
    prompt_diff = max((max(d[k] for d in by_case) - min(d[k] for d in by_case)) for k in by_case[0]) if by_case else None
    costs = cost(summaries, json.loads((ROOT / "assumptions.json").read_text()))
    (RESULTS / "summary.json").write_text(json.dumps({"configs": summaries, "agreement": agree,
                                                   "prompt_tokens_max_diff_across_configs": prompt_diff,
                                                   "yes_no_flips": flips}, indent=1, ensure_ascii=False) + "\n")
    (RESULTS / "cost.json").write_text(json.dumps(costs, indent=1) + "\n")
    (RESULTS / "summary.md").write_text(markdown(summaries, agree, costs, prompt_diff, flips))
    print((RESULTS / "summary.md").read_text())
