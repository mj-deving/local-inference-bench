---
task: "Lokale Serving- und Kostenstudie: Ollama vs llama.cpp vs MLX, zwei Quantisierungen, qualitätsgebunden"
slug: 20261109_local-inference-bench
project: local-inference-bench
effort: advanced
effort_source: inferred
phase: build
progress: 13/14
mode: interactive
started: 2026-11-09T16:45:00+01:00
updated: 2026-11-09T21:10:00+01:00
---

> **Über diese Datei.** Das ist die ISA (Ideal State Artifact), gegen die dieses Repo gebaut wurde:
> was „fertig“ hieß, mit welcher Probe jedes Kriterium widerlegt würde und was bewusst als Grenze
> stehen blieb. Es ist der Arbeitsstand aus dem Bau, nur um private Bezüge bereinigt; Hinweise wie
> „kein Remote“ oder „ISA.md gitignored“ beschreiben die Zeit vor der Veröffentlichung. Commit-Hashes
> stammen aus der lokalen Vorgeschichte, öffentlich wurde das Repo als neuer Verlauf exportiert. Die
> Leak-Probes lesen ihre Begriffe aus einer lokalen, nicht versionierten Datei `.private-terms`.

## Ideal State

### L3 · One sentence

Ein Repo beantwortet die Serving-Architekturfragen (welche Runtime, welche Quantisierung, welche Latenz und welcher Durchsatz, was kostet es) mit Messungen auf einem Apple-Silicon-Mac, bei denen keine Geschwindigkeitszahl ohne ihre Qualitätszahl steht.

### L2 · Summary

Wer lokales Serving plant, liest in fünf Minuten, welche Runtime auf dieser Klasse Hardware gewinnt und was Quantisierung an Qualität kostet.

- Drei Runtimes (Ollama, llama.cpp-Server, mlx-lm-Server) mit demselben Modell (Qwen3-8B) in je zwei Quantisierungen, über denselben OpenAI-kompatiblen Streaming-Client.
- TTFT, Decode-Rate, E2E p50/p95, Durchsatz bei Konkurrenz 1/2/4/8, Peak-Speicher, kalt getrennt von warm, 3 Wiederholungen mit Streuung.
- Qualität je Konfiguration auf dem geteilten Golden- und Held-out-Set mit den portierten Grading-Regeln.
- Entscheidungsabschnitt mit Kosten pro 1000 Antworten aus gemessenem Durchsatz und explizit markierten Annahmen.

### L1 · Full articulation

Korpus, Fälle und Grading kommen aus agentic-rag-mastra (Commit gepinnt, Hashes). Retrieval wird einmal mit bge-m3 fixiert, damit nur die Runtime variiert. Jeder Messlauf hält den geteilten GPU-Lock. Rohdaten jeder Anfrage liegen in results/raw, das README zitiert nur Zahlen aus results/summary.json und results/cost.json.

## Problem

Inference-Infrastruktur und Serving hatten keinen Beleg. Kubernetes und GPU-Cluster sind auf diesem Mac nicht erreichbar; die Architekturfragen lassen sich aber auf einer Maschine ehrlich messen.

## Vision

Das README zeigt eine Tabelle, in der neben jeder Token-Rate die Verweigerungs-Korrektheit und Zitat-Validität derselben Konfiguration steht, und einen Satz, wann lokal billiger ist als eine gehostete API und unter welchen Annahmen.

## Out of Scope

- Kubernetes, GPU-Cluster, Multi-Node, vLLM/TGI auf CUDA.
- Speculative Decoding, KV-Cache-Quantisierung, Feintuning.
- Agentic Tool-Calling (fester Kontext statt Tool, bewusst).

## Constraints

- Keine globale Installation ohne Rückfrage; llama-server war bereits installiert, mlx-lm projektlokal via uv.
- Jede Messung unter einem gemeinsamen Ollama-Lock.
- Kein Remote, kein Push; ISA.md/ISA.html gitignored; autoreview `--engine claude` vor jedem Commit, explizite Pfade.

## Criteria

- [x] ISC-1: Grading-Port ist paritätisch zum TS-Original (Fixture aus grading.ts erzeugt).
- [x] ISC-2: Metrik-Mathematik (Perzentil, Streuung, TTFT, Decode-Rate, Durchsatz, Kosten) ist ohne Modell getestet, inkl. Grenz- und Fehlerfälle.
- [x] ISC-3: Fixe Retrieval stimmt mit Mastras protokollierten Top-5 überein (37/37).
- [ ] ISC-4 (5/6, mlx-8bit offen): Alle 6 Konfigurationen haben eine Rohdatei mit 3 Kaltstarts, 3×48 warmen Anfragen und Sweep 1/2/4/8 × 3.
- [x] ISC-5: Jede Messung lief unter dem Lock (Lock-Owner im Log / GpuLock im Code-Pfad).
- [x] ISC-6: Prompt-Tokens sind je Fall über die Runtimes vergleichbar (kein stilles Abschneiden).
- [x] ISC-7: Thinking ist aus (Anteil Antworten mit nichtleerem <think> berichtet).
- [x] ISC-8: summary.json enthält je Konfiguration Latenz/Durchsatz mit Streuung UND Refusal-Korrektheit/Zitat-Validität.
- [x] ISC-9: Peak-Speicher je Konfiguration (RSS und Footprint) erfasst.
- [x] ISC-10: cost.json rechnet aus gemessenem Durchsatz, Annahmen als Annahmen markiert.
- [x] ISC-11: README mit Quickstart, Method, Results inkl. Überraschungen, Limits; jede Zahl im README steht in einer Results-Datei.
- [x] ISC-12: `uv run pytest` grün ohne Modell und ohne Server.
- [x] ISC-13: autoreview --engine claude ohne P1 vor jedem Commit.
- [x] ISC-14: Repo-Dateien ohne private Bezüge (Liste in `.private-terms`, lokal und nicht versioniert).

Anti-Kriterium: Keine Geschwindigkeitszahl im README ohne Qualitätszahl derselben Konfiguration; keine Einzelzahl ohne Wiederholungen.

## Test Strategy

| isc | type | check | threshold | tool | anchors_to |
|-----|------|-------|-----------|------|------------|
| ISC-1 | bash | Paritätstest | grün | uv run pytest tests/test_grading.py | tests/fixtures/grading_parity.json |
| ISC-2 | bash | Metriktests | grün | uv run pytest tests/test_metrics.py | |
| ISC-3 | bash | Vergleich Top-5 | 37/37 | bun -e Vergleich gegen ../agentic-rag-mastra/results | data/cases.jsonl |
| ISC-4 | bash | Rohdatei-Struktur | 6 Dateien, Zählungen stimmen | uv run pytest tests/test_results.py | results/raw/*.json |
| ISC-5 | bash | Code-Pfad | run_config nur in GpuLock | grep GpuLock bench.py | |
| ISC-6 | bash | prompt_tokens je Fall | Abweichung klein, kein Fall gekappt | report | results/summary.json |
| ISC-7 | bash | think-Anteil | berichtet | report | results/summary.json |
| ISC-8 | bash | Felder je Config | vorhanden | uv run pytest tests/test_results.py | |
| ISC-9 | bash | Felder je Config | > 0 | uv run pytest tests/test_results.py | |
| ISC-10 | bash | Kostendatei | Annahmen-Block vorhanden | uv run pytest tests/test_results.py | results/cost.json |
| ISC-11 | bash | Zahlenabgleich | jede README-Zahl in results | scripts/check_readme_numbers.ts | README.md |
| ISC-12 | bash | Testlauf ohne Server | exit 0 | uv run pytest | |
| ISC-13 | bash | autoreview | kein P1 | autoreview --engine claude | |
| ISC-14 | bash | grep private Bezüge | 0 Treffer | rg | |

## Decisions

- 2026-11-09: Nur 5 Configs committet (Maintainer); mlx-8bit nach Speicher-Vorfall heute nicht neu gestartet.

- 2026-11-09: Python + uv statt TypeScript, weil mlx-lm Python-nativ ist und der Server-Prozess ohnehin Python braucht. Grading portiert und per Fixture an das Original gebunden.
- 2026-11-09: Fester Kontext statt Tool-Call, damit nur die Runtime variiert. llama.cpp lädt die GGUF-Blobs aus dem Ollama-Store: identische Gewichte, Unterschied ist reine Runtime.
- 2026-11-09: Fairness-Regeln aus vier autoreview-Runden (alle an installierten Versionen verifiziert): kein seed (mlx _is_batchable), llama-server --cache-ram 0, eindeutige Anfrage-ID hinter dem System-Prompt, Page-Cache vor Kaltstarts befüllt (echter Disk-Kaltstart braucht sudo purge, out of scope), try/finally, nur am Netzteil.
- 2026-11-09: Ollama /v1 ignoriert /no_think und chat_template_kwargs; nur reasoning_effort "none" schaltet Thinking ab (Smoke: 262 statt 79 Tokens).
- 2026-11-09: MLX 4-bit/8-bit (affine, group 64) ist nicht bitgleich zu Q4_K_M/Q8_0; als Grenze dokumentiert.

## Evidence

- ISC-4: 5 Rohdateien committet (dfe314f). mlx-8bit am 2026-11-09 20:33 im c=8-Sweep abgebrochen: Rechner unter Speicherdruck (Swap 3,6 GB, kein Reboot, keine Panic). Maintainer: heute nicht mehr starten. Vor Neulauf: Speicherdeckel für mlx_lm.server und Low-Memory-Abbruch im Harness.
- ISC-8/9/10: results/summary.json, results/cost.json je Config mit Latenz/Durchsatz/Streuung, Refusal/Zitat, RSS und Footprint, Annahmenblock (tests/test_report.py grün).
- ISC-11: tests/test_readme_numbers.py grün; Gegenprobe mit erfundener 1234 schlägt fehl. README Results, Surprises, Decision, Limits.
- ISC-13: autoreview --engine claude P1 scoped-clean vor dfe314f (Runden 6-8: 4 P1 behoben).
- ISC-14: rg auf private Bezüge in README, results, src, tests: 0 Treffer.
- Befund: a5-predictive-policing kippt Ja/Nein zwischen Q4 und Q8 und innerhalb llamacpp-q4; Grading wertet alle Varianten als korrekt. Polarity-Check gehört ins Gate (Vorschlag, nicht umgesetzt).

- ISC-1, ISC-2: 21 Tests grün (2026-11-09 17:10).
- ISC-3: 37/37 gleiche Reihenfolge und Menge.
- ISC-6: größte Prompt-Token-Differenz je Fall über ollama-q4/llamacpp-q4/mlx-4bit: 4 Tokens (results/summary.md, 19:00).
- ISC-7: think share 0,0 % für alle bisher gemessenen Configs (results/summary.md).
- Zwischenbefunde: Prefill ~70 % der Latenz; Durchsatz-Skalierung c=1→8 ollama 13,3→14,0, llama.cpp 15,2→19,8, MLX 17,4→21,9 tok/s; Ollama erster Kaltstart 26,6 s trotz Page-Cache (Folge-Reps 6 s, ollama-q8 Rep 0 5,4 s); MLX Footprint 30,7 GB.
- ISC-13: Commit ab8babb nach autoreview Runde 5 scoped-clean.
- ISC-5: run_config und smoke laufen nur in `with GpuLock()`; test_gpu_lock_* grün.
- ISC-12: 40 passed, 1 skipped (README-Test ohne Results) ohne Server, Commit f9bdd5e.
- ISC-13: autoreview --engine claude P1 Runde 4 scoped-clean vor f9bdd5e (Runden 1–3: 5 P1-Funde, alle behoben).
