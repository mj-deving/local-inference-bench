# local-inference-bench

**Which runtime, which quantization, what latency and throughput, and what it costs: an honest
serving study of Qwen3-8B on one Apple Silicon Mac, where no speed number appears without the
quality number of the same configuration.**

Three runtimes (Ollama, llama.cpp `llama-server`, `mlx_lm.server`) serve the same model in two
quantizations each. One OpenAI-compatible streaming client sends byte-identical prompts to all of
them. Quality is graded on a shared EU AI Act RAG test set with deterministic rules, so a faster
configuration that refuses wrongly or cites chunks it was never given shows up in the same table.

[Quickstart](#quickstart) · [Method](#method) · [Results](#results) · [Decision](#decision) · [Limits](#limits)

## Quickstart

Requires macOS on Apple Silicon, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com)
and a `llama-server` binary from [llama.cpp](https://github.com/ggml-org/llama.cpp) on `PATH`.
`mlx-lm` is installed into the project venv by uv.

```bash
uv sync
uv run pytest                      # harness, metric math, grading parity: no model, no server

ollama pull qwen3:8b               # Q4_K_M; llama.cpp serves the same GGUF blob
ollama pull qwen3:8b-q8_0          # Q8_0
uv run hf download mlx-community/Qwen3-8B-4bit
uv run hf download mlx-community/Qwen3-8B-8bit

uv run bench smoke ollama-q4       # two requests, prints answer, tokens, TTFT
uv run bench run ollama-q4 llamacpp-q4 mlx-4bit ollama-q8 llamacpp-q8 mlx-8bit
uv run bench report                # results/summary.{md,json}, results/cost.json
```

`data/` already holds the corpus, the 48 cases and their fixed retrieval, so `bench prepare` is only
needed to rebuild them from the source repository (it needs `bge-m3` in Ollama).

Every `bench run` waits for AC power and for the cross-session lock directory
`~/projects/.ollama-eval.lock` before it measures. A benchmark taken while another job uses the
GPU, or on battery, is not comparable.

## Why Python

`mlx-lm` and its server are Python, and the client is a few hundred lines of `httpx` and the
standard library. The grading rules come from a TypeScript project; they are ported to
[`grading.py`](src/local_inference_bench/grading.py) and pinned to the original by a parity fixture
that [`scripts/grading_fixture.ts`](scripts/grading_fixture.ts) generates from the TypeScript module
itself.

## Method

**Model and weights.**

| config | runtime | weights |
|---|---|---|
| `ollama-q4` | Ollama | `qwen3:8b` (GGUF Q4_K_M) |
| `llamacpp-q4` | llama-server | the same GGUF file, read from Ollama's blob store |
| `mlx-4bit` | mlx_lm.server | `mlx-community/Qwen3-8B-4bit` |
| `ollama-q8` | Ollama | `qwen3:8b-q8_0` (GGUF Q8_0) |
| `llamacpp-q8` | llama-server | the same GGUF file |
| `mlx-8bit` | mlx_lm.server | `mlx-community/Qwen3-8B-8bit` |

Ollama and llama.cpp therefore run byte-identical weights: any difference between them is the
runtime. MLX quantization (affine, group size 64) is a different scheme from Q4_K_M and Q8_0, so
MLX rows compare a runtime *and* a quantization format.

**Workload.** 48 questions about the EU AI Act (40 golden, 8 held-out) from the companion project
`agentic-rag-mastra`, with its 66 verbatim paragraphs. Retrieval is fixed once (bge-m3, cosine,
top 5) and matches the top 5 that project logged for all 37 cases it recorded. The five chunks are
pasted into the prompt instead of being fetched by a tool, so the only variable is the server.
Prompts are long and answers mostly short: a prefill-heavy RAG workload.

**Settings shared by every runtime.** 8 parallel slots with 8,192 tokens of context each,
temperature 0, `max_tokens` 512, thinking disabled. Each runtime needs its own switch for that:
`chat_template_kwargs` for llama.cpp and MLX, `reasoning_effort: "none"` for Ollama, whose
OpenAI endpoint ignores both the template switch and Qwen3's `/no_think`.

**Fairness rules** (each found during review, each pinned by a test):

- No `seed` in the request: `mlx_lm.server` does not batch seeded requests, so a seed would have
  serialized MLX under concurrency.
- `llama-server --cache-ram 0`, and every request carries a unique id right after the shared system
  prompt. No runtime can answer a repeated request from an exact-prompt cache; the system prompt
  stays a cacheable prefix, as in a real service.
- Before the cold starts the weight files are read once, so every cold rep is
  "process-cold, page-cache-warm" for every runtime.

**Phases per config**, all under the lock:

1. *Cold*: 3 × start the server, wait until it is ready, send one request. Reported as launch to
   first token.
2. *Warm*: 2 warmup requests, then 3 repetitions of all 48 cases, one at a time. TTFT, end-to-end
   latency p50/p95 and median decode rate per repetition, then mean ± sd across repetitions.
   Quality is graded on repetition 0.
3. *Concurrency sweep*: 16 requests at concurrency 1, 2, 4 and 8, 3 repetitions each. Throughput is
   output tokens over the wall span of the batch.

Peak memory is sampled every 0.5 s over the server's process tree as RSS (counts mmapped weights,
which llama.cpp and Ollama use) and as macOS physical footprint (counts Metal allocations, which
MLX uses). Neither alone is comparable across runtimes, so both are reported.

**Metric definitions** ([`metrics.py`](src/local_inference_bench/metrics.py), tested in
[`tests/test_metrics.py`](tests/test_metrics.py)): TTFT is send to first streamed token. Decode
rate is (tokens − 1) / (last token − first token), so prefill is excluded. Percentiles use linear
interpolation. Token counts come from the server's `usage` field.

**Quality** uses the shared grading rules unchanged: *refusal correctness* (refuses exactly when
the case expects it), *citation validity* (every cited chunk was in the prompt, and no answer is
uncited), plus citation recall and fact coverage on answered cases.

## Results

Five of six configs ran to completion. **`mlx-8bit` is missing**: its run drove the machine into
memory pressure during the c=8 sweep and was stopped from outside. What is known about that is in
[`results/mlx-8bit-incident.md`](results/mlx-8bit-incident.md). Every number below comes from
[`results/summary.md`](results/summary.md) (generated from `results/raw/`) or
[`results/cost.json`](results/cost.json); a test fails if the README cites a number that is not there.

**Speed and quality, warm, one request at a time** (speed: mean ± sd over 3 repetitions; quality:
repetition 0):

| config | TTFT p50 s | E2E p95 s | decode tok/s | refusal ok golden / held-out | citation valid golden / held-out |
|---|---|---|---|---|---|
| ollama-q4 | 3.38 ± 0.19 | 8.79 ± 0.15 | 35.1 ± 1.5 | 97.5 % / 87.5 % | 100.0 % / 100.0 % |
| llamacpp-q4 | 3.17 ± 0.09 | 8.40 ± 0.10 | 38.2 ± 2.7 | 97.5 % / 87.5 % | 100.0 % / 100.0 % |
| mlx-4bit | 2.93 ± 0.08 | 6.90 ± 0.05 | 44.5 ± 1.1 | 97.5 % / 87.5 % | 100.0 % / 100.0 % |
| ollama-q8 | 3.15 ± 0.39 | 11.17 ± 0.76 | 26.4 ± 1.4 | 97.5 % / 87.5 % | 100.0 % / 100.0 % |
| llamacpp-q8 | 2.88 ± 0.09 | 9.05 ± 1.11 | 27.4 ± 1.3 | 95.0 % / 87.5 % | 100.0 % / 100.0 % |

**Throughput under concurrency** (output tok/s over the batch, mean ± sd over 3 repetitions):

| config | c=1 | c=8 | E2E p95 at c=8 s |
|---|---|---|---|
| ollama-q4 | 13.3 ± 0.3 | 14.0 ± 0.4 | 66.66 |
| llamacpp-q4 | 15.2 ± 0.4 | 19.8 ± 0.1 | 50.52 |
| mlx-4bit | 17.4 ± 0.8 | 21.9 ± 0.6 | 59.98 |
| ollama-q8 | 12.6 ± 0.8 | 17.7 ± 0.2 | 74.71 |
| llamacpp-q8 | 12.2 ± 0.1 | 19.8 ± 1.3 | 61.34 |

**Memory and cold start:**

| config | peak RSS MB | peak footprint MB | launch → first token s (min–max) |
|---|---|---|---|
| ollama-q4 | 16277 | 19483 | 6.04–26.57 |
| llamacpp-q4 | 14410 | 9499 | 5.05–5.78 |
| mlx-4bit | 4716 | 30720 | 5.87–7.73 |
| ollama-q8 | 23498 | 19481 | 4.99–5.36 |
| llamacpp-q8 | 17867 | 9490 | 4.93–5.53 |

Thinking stayed off in every response (think share 0.0 %), no answer hit the token limit, and the
largest prompt-token difference for one case between runtimes was 4 tokens (cause not checked;
the runtimes apply the chat template differently).

### What surprised me

1. **The grading rules missed the most important quality difference.** On "Are AI systems that
   predict crime risk from profiling alone prohibited?", both Q8 configs answer "Ja" in all 3
   repetitions. `ollama-q4` and `mlx-4bit` answer "Nein" in all 3, the legally opposite answer.
   `llamacpp-q4`, with the same weights as `ollama-q4`, answers "Ja" twice and "Nein" once, flipping
   on nothing but the request id line. Every one of these answers scores full marks: the citation is
   valid and the expected fact "Profiling" is present. The scorecard says Q4 and Q8 are equal
   (identical refusal correctness and citation validity, apart from one extra wrong refusal for
   `llamacpp-q8`); the yes/no tables in `summary.md` say Q4 is less stable on one of 48 cases. One
   case is not a rate, but it is exactly the failure a deterministic citation gate cannot see.
2. **Q8 had the faster prefill on llama.cpp.** TTFT p50 was 2.88 ± 0.09 s for Q8 against
   3.17 ± 0.09 s for Q4, while decode was slower (27.4 vs 38.2 tok/s). On Ollama the TTFT gap is
   within one standard deviation. A plausible reason, not measured here: dequantizing Q4_K blocks
   costs compute in a compute-bound phase. The two configs ran at different times, so drift is not
   ruled out. Under load they meet: both llama.cpp configs reach 19.8 tok/s at c=8.
3. **Concurrency barely pays on this workload.** Prefill is 61.1 % to 71.2 % of end-to-end latency
   (median). Going from c=1 to c=8 lifts throughput modestly, while the p95 latency at c=8 grows to
   50.52 to 74.71 s. Ollama Q4 gains almost nothing (13.3 to 14.0 tok/s).
4. **Same weights, different runtime, different behaviour.** Ollama and llama.cpp load the identical
   GGUF file. For Q4, llama.cpp is faster in every table above and scales better under load
   (19.8 vs 14.0 tok/s at c=8). For Q8 the two are within one standard deviation in TTFT, decode
   rate and throughput at c=1, but llama.cpp has the lower E2E p95 (9.05 vs 11.17 s) and leads at
   c=8 (19.8 vs 17.7 tok/s). Answer-or-refuse decisions agree on 100.0 % of
   cases for Q4 and 97.9 % for Q8, and point 1 shows a yes/no flip between them: bit-identical
   weights do not guarantee identical answers.
5. **MLX trades memory for speed, and the price is real.** `mlx-4bit` is the fastest config
   (44.5 tok/s decode, 21.9 tok/s at c=8), but its physical footprint peaked at 30720 MB against an
   RSS of 4716 MB. The machine ran out of memory during the 8-bit run of the same server.
6. **RSS and footprint disagree in opposite directions.** llama.cpp shows 14410 MB RSS but 9499 MB
   footprint, MLX the reverse. The likely reason (not verified): llama.cpp maps the weight file into
   memory, which RSS counts and footprint does not, while MLX allocates through Metal, which
   footprint counts and RSS does not. Reporting only one of them would have made either runtime look
   small.
7. **Ollama's very first start was slow** (26.57 s) even with the weights in the page cache; the two
   restarts after it were in line with the other runtimes. Cause not investigated.
8. **Answers move with irrelevant prompt changes.** Repetitions differ only in the request id line.
   At temperature 0, 83.3 % to 89.6 % of answer texts stayed identical across repetitions. The
   answer-or-refuse decision was stable for 100.0 % of cases; point 1 is the one yes/no that was not.

## Decision

The cost model is in [`assumptions.json`](assumptions.json); every value there is an assumption
(power draw, electricity price, hardware price, amortization, utilization, hosted price tiers),
not a measurement. The measured inputs are throughput and answer length at the best concurrency
per config, so a config that writes shorter answers also looks cheaper.

| | llamacpp-q4 (cheapest local) |
|---|---|
| answers per hour, measured at c=8 | 937 |
| EUR per 1000 answers, energy only (machine already owned) | 0.022 |
| EUR per 1000 answers, amortized, serving 24/7 | 0.136 |
| EUR per 1000 answers, amortized, serving office hours | 0.477 |
| break-even hosted price, EUR per million tokens (24/7) | 0.078 |
| hosted, small-model price tier | 0.197 |
| hosted, mid-model price tier | 1.966 |

**When local serving on a machine like this wins:**

- The hardware exists anyway. Then the marginal cost is electricity, and local is cheaper than
  either hosted tier.
- Data must not leave the machine. The cost question is then secondary.
- The load is low and mostly one request at a time. At c=1 the E2E p95 is between 6.90 and
  11.17 s; the throughput above needs c=8, where tail latency is close to a minute.

**When a hosted API wins:**

- Interactive traffic with several users at once (E2E p95 at c=8 between 50.52 and 74.71 s).
- Hardware bought for this purpose and not busy around the clock. At office-hour utilization the
  amortized local cost is above the small hosted tier.
- Long prompts at scale. Prefill takes most of the latency here and does not speed up with batching.

**Which runtime and quantization:** on this machine, llama.cpp with Q4_K_M has the best measured
cost per answer, no cold-start outlier and no memory incident. Q8 costs decode speed but answered
the one yes/no question in point 1 consistently, so for a domain where a flipped yes/no matters,
Q8 is the safer default until a larger test set measures the difference. MLX 4-bit is faster at
c=1 and for decode, but needs a memory ceiling that `mlx_lm.server` does not offer. Ollama is the
convenient default and the slowest under load. Whatever is chosen, the quality gate needs a check
for answer polarity, not only for citations.

## Limits

- One machine: an M4 Pro with 48 GB unified memory. No GPU cluster, no Kubernetes, no CUDA
  runtimes such as vLLM or TGI. The architecture questions transfer; the absolute numbers do not.
- One model family at 8B, one workload shape (long prompt, short answer).
- Cold start is process-cold, page-cache-warm. A true disk-cold start needs `sudo purge` and is not
  measured.
- Power draw is not measured (`powermetrics` needs root); the cost section uses a stated
  assumption.
- The grading rules see only single citations like `[chunk:a]`. A model that writes
  `[chunk:a, chunk:b]` scores as uncited. This is the shared rule of the companion projects and is
  kept unchanged so numbers stay comparable.
- MLX quantization differs from GGUF quantization, see Method.
- The grading rules check refusals, citations and key terms, not whether a yes/no answer is right.
  `summary.md` lists yes/no flips separately for that reason; a polarity check belongs in the gate.
- `mlx-8bit` is not measured: its run exhausted memory on the shared machine and was not repeated.
  The harness has no memory ceiling or low-memory abort yet; both are needed before a rerun.
