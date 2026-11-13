# mlx-8bit: run aborted, machine under memory pressure

There is no `results/raw/mlx-8bit.json`. The run was stopped from outside on 2026-11-09 at 20:33
during the concurrency sweep at c=8, after cold starts, all 3 warm repetitions and the sweep levels
c=1, c=2 and c=4 had finished. The machine had become unresponsive.

What the server log shows (`results/raw/mlx-8bit.server.log`, not committed):

- Server start 20:01:01, 336 completed chat requests before the stop.
- The prompt cache was not the cause: it held 10 sequences, 2.51 GB and then 2.38 GB in the last
  minute, the same size as in the completed mlx-4bit run (2.61 GB at most).

What was reported but not measured by this harness: the server's memory had grown to about 31 GB,
for 8-bit weights of about 9 GB. The completed mlx-4bit run had already shown a peak physical
footprint of 30720 MB against a peak RSS of 4716 MB (see `summary.md`).

Most likely cause (inferred, not proven): MLX keeps freed Metal buffers in its own cache instead of
returning them to the system, and `mlx_lm.server` 0.32.2 exposes no flag to cap that cache. At c=8
the batched KV caches and activations are largest, so the cache grows to its high-water mark. On a
48 GB machine shared with other work, that is enough to push the system into swap.

Not rerun on purpose. A rerun needs a memory ceiling for the server process and a harness guard that
aborts when free memory drops below a threshold.
