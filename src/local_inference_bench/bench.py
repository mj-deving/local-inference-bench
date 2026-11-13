"""Benchmark runner: prepare cases, run one runtime/quant config under the shared GPU lock.

  uv run bench prepare ../agentic-rag-mastra   # copies corpus + cases, fixes top-5 retrieval (bge-m3)
  uv run bench run ollama-q4                    # one config: cold starts, warm eval, concurrency sweep
  uv run bench report                           # aggregate results/raw/*.json -> results/summary.*
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .metrics import RequestTiming
from .prompt import messages

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = ROOT / "results" / "raw"
LOCK = Path.home() / "projects" / ".ollama-eval.lock"
SESSION = "local-inference-bench"

CTX_PER_SLOT = 8192
PARALLEL = 8
MAX_TOKENS = 512
TOP_K = 5
REPS = 3
SWEEP = (1, 2, 4, 8)
SWEEP_REQUESTS = 16


@dataclass(frozen=True)
class Config:
    name: str
    runtime: str  # ollama | llamacpp | mlx
    quant: str
    model: str  # ollama tag or HF repo
    port: int


CONFIGS = {
    c.name: c
    for c in [
        Config("ollama-q4", "ollama", "Q4_K_M", "qwen3:8b", 11500),
        Config("ollama-q8", "ollama", "Q8_0", "qwen3:8b-q8_0", 11500),
        Config("llamacpp-q4", "llamacpp", "Q4_K_M", "qwen3:8b", 11501),
        Config("llamacpp-q8", "llamacpp", "Q8_0", "qwen3:8b-q8_0", 11501),
        Config("mlx-4bit", "mlx", "MLX 4-bit", "mlx-community/Qwen3-8B-4bit", 11502),
        Config("mlx-8bit", "mlx", "MLX 8-bit", "mlx-community/Qwen3-8B-8bit", 11502),
    ]
}


# ---------------------------------------------------------------- shared lock


def on_ac_power() -> bool:
    return "AC Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout


def wait_for_ac_power() -> None:
    """Battery and AC numbers are not comparable, and a multi-hour GPU run drains the battery."""
    if not on_ac_power():
        print("on battery power - waiting for AC", flush=True)
        while not on_ac_power():
            time.sleep(30)


class GpuLock:
    """The cross-session mkdir lock. Waits rather than measuring under contention."""

    def __enter__(self):
        wait_for_ac_power()
        announced = False
        while True:
            try:
                LOCK.mkdir()
                (LOCK / "owner").write_text(f"{SESSION} {time.ctime()}\n")
                return self
            except FileExistsError:
                if not announced:
                    owner = (LOCK / "owner").read_text() if (LOCK / "owner").exists() else "?"
                    print(f"lock held by: {owner.strip()} - waiting", flush=True)
                    announced = True
                time.sleep(15)

    def __exit__(self, *exc):
        owner = (LOCK / "owner").read_text() if (LOCK / "owner").exists() else ""
        if owner.startswith(SESSION):
            shutil.rmtree(LOCK)


# ---------------------------------------------------------------- prepare


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(mastra: Path) -> None:
    DATA.mkdir(exist_ok=True)
    files = {"chunks.json": mastra / "corpus" / "chunks.json", "golden.jsonl": mastra / "golden" / "v1.jsonl",
             "heldout.jsonl": mastra / "golden" / "heldout.jsonl"}
    commit = subprocess.run(["git", "-C", str(mastra), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for dst, src in files.items():
        shutil.copyfile(src, DATA / dst)
    chunks = json.loads((DATA / "chunks.json").read_text())
    cases = []
    for set_name, fname in (("golden", "golden.jsonl"), ("heldout", "heldout.jsonl")):
        for line in (DATA / fname).read_text().splitlines():
            if line.strip():
                cases.append({**json.loads(line), "set": set_name})
    with GpuLock():
        vec = _embed([c["text"] for c in chunks])
        qvec = _embed([c["question"] for c in cases])
    for case, q in zip(cases, qvec):
        scores = sorted(((_dot(q, v), c["id"]) for v, c in zip(vec, chunks)), reverse=True)
        case["retrieved"] = [cid for _, cid in scores[:TOP_K]]
    with (DATA / "cases.jsonl").open("w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (DATA / "SOURCE.json").write_text(json.dumps({
        "from": "agentic-rag-mastra", "commit": commit,
        "files": {k: {"source": str(v.relative_to(mastra)), "sha256": _sha(DATA / k)} for k, v in files.items()},
        "retrieval": {"embedding": "bge-m3 (Ollama)", "similarity": "cosine", "top_k": TOP_K, "query": "full question"},
    }, indent=2) + "\n")
    print(f"{len(cases)} cases -> {DATA / 'cases.jsonl'}")


def _embed(texts: list[str]) -> list[list[float]]:
    r = httpx.post("http://127.0.0.1:11434/api/embed", json={"model": "bge-m3", "input": texts}, timeout=600)
    r.raise_for_status()
    return [_unit(v) for v in r.json()["embeddings"]]


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_cases() -> tuple[list[dict], dict[str, dict]]:
    chunks = {c["id"]: c for c in json.loads((DATA / "chunks.json").read_text())}
    cases = [json.loads(line) for line in (DATA / "cases.jsonl").read_text().splitlines() if line.strip()]
    return cases, chunks


# ---------------------------------------------------------------- servers


def gguf_blob(tag: str) -> Path:
    """The GGUF file Ollama stores for a tag, so llama.cpp serves byte-identical weights."""
    name, _, version = tag.partition(":")
    manifest = Path.home() / ".ollama/models/manifests/registry.ollama.ai/library" / name / (version or "latest")
    layers = json.loads(manifest.read_text())["layers"]
    digest = next(lyr["digest"] for lyr in layers if lyr["mediaType"] == "application/vnd.ollama.image.model")
    return Path.home() / ".ollama/models/blobs" / digest.replace(":", "-")


def server_cmd(cfg: Config) -> tuple[list[str], dict[str, str]]:
    """Command line and extra environment for one config. Same slots and context per runtime."""
    if cfg.runtime == "ollama":
        env = {"OLLAMA_HOST": f"127.0.0.1:{cfg.port}", "OLLAMA_NUM_PARALLEL": str(PARALLEL), "OLLAMA_KEEP_ALIVE": "-1",
               "OLLAMA_CONTEXT_LENGTH": str(CTX_PER_SLOT), "OLLAMA_MAX_LOADED_MODELS": "1"}
        return ["ollama", "serve"], env
    if cfg.runtime == "llamacpp":
        return ["llama-server", "-m", str(gguf_blob(cfg.model)), "--port", str(cfg.port), "-np", str(PARALLEL),
                "-c", str(CTX_PER_SLOT * PARALLEL), "-ngl", "99", "--jinja", "--cache-ram", "0"], {}
    if cfg.runtime == "mlx":
        return [str(Path(sys.executable).parent / "mlx_lm.server"), "--model", cfg.model, "--port", str(cfg.port),
                "--decode-concurrency", str(PARALLEL), "--prompt-concurrency", str(PARALLEL),
                "--max-tokens", str(MAX_TOKENS)], {}
    raise ValueError(f"unknown runtime {cfg.runtime!r}")


def weight_files(cfg: Config) -> list[Path]:
    if cfg.runtime in ("ollama", "llamacpp"):
        return [gguf_blob(cfg.model)]
    from huggingface_hub import snapshot_download

    return sorted(Path(snapshot_download(cfg.model, local_files_only=True)).glob("*.safetensors"))


def prime_page_cache(files: list[Path]) -> int:
    """Read the weights once so every cold rep starts from the same disk state: process-cold,
    page-cache-warm. A true disk-cold start needs `sudo purge`, which this harness does not run."""
    total = 0
    for f in files:
        with f.open("rb") as fh:
            while chunk := fh.read(64 << 20):
                total += len(chunk)
    return total


def launch(cfg: Config, log) -> subprocess.Popen:
    cmd, extra = server_cmd(cfg)
    return subprocess.Popen(cmd, env={**os.environ, **extra}, stdout=log, stderr=subprocess.STDOUT)


def wait_ready(cfg: Config, proc: subprocess.Popen, timeout: float = 600) -> None:
    path = {"ollama": "/api/version", "llamacpp": "/health", "mlx": "/v1/models"}[cfg.runtime]
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{cfg.name}: server exited with {proc.returncode}")
        try:
            if httpx.get(f"http://127.0.0.1:{cfg.port}{path}", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"{cfg.name}: not ready after {timeout}s")


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _tree(pid: int) -> list[int]:
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        kids = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()
        todo += [int(k) for k in kids]
    return out


_FOOTPRINT = re.compile(r"Footprint:\s+([\d.]+)\s+(KB|MB|GB)")


def _mem_mb(pid: int) -> tuple[float, float]:
    """(rss_mb, phys_footprint_mb). Footprint counts Metal allocations; RSS counts mmapped weights."""
    rss = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    fp = _FOOTPRINT.search(subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout)
    scale = {"KB": 1 / 1024, "MB": 1, "GB": 1024}
    return (int(rss) / 1024 if rss else 0.0, float(fp.group(1)) * scale[fp.group(2)] if fp else 0.0)


class MemSampler(threading.Thread):
    def __init__(self, pid: int, period: float = 0.5):
        super().__init__(daemon=True)
        self.pid, self.period, self.peak_rss, self.peak_fp = pid, period, 0.0, 0.0
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            pids = _tree(self.pid)
            samples = [_mem_mb(p) for p in pids]
            self.peak_rss = max(self.peak_rss, sum(s[0] for s in samples))
            self.peak_fp = max(self.peak_fp, sum(s[1] for s in samples))
            self._halt.wait(self.period)

    def stop(self) -> dict:
        self._halt.set()
        self.join()
        return {"peak_rss_mb": round(self.peak_rss), "peak_footprint_mb": round(self.peak_fp)}


# ---------------------------------------------------------------- client


def request_body(cfg: Config, msgs: list[dict]) -> dict:
    return {"model": cfg.model, "messages": msgs, "stream": True, "stream_options": {"include_usage": True},
            # No "seed": mlx_lm.server refuses to batch seeded requests (_is_batchable), and
            # temperature 0 is greedy anyway.
            "temperature": 0, "max_tokens": MAX_TOKENS,
            # Thinking off, per runtime dialect: chat_template_kwargs (llama.cpp, MLX),
            # reasoning_effort (Ollama /v1 ignores /no_think and the template kwarg).
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}


def chat(client: httpx.Client, cfg: Config, msgs: list[dict]) -> dict:
    body = request_body(cfg, msgs)
    parts, reasoning, usage, chunks = [], [], None, 0
    t_first = t_last = None
    t_send = time.perf_counter()
    with client.stream("POST", f"http://127.0.0.1:{cfg.port}/v1/chat/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            usage = ev.get("usage") or usage
            for choice in ev.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if content or thought:
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    chunks += 1
                    parts.append(content)
                    reasoning.append(thought)
    t_done = time.perf_counter()
    tokens = usage.get("completion_tokens") if usage else None
    timing = RequestTiming(t_send, t_first, t_last, t_done, tokens if tokens is not None else chunks)
    return {"text": "".join(parts), "reasoning": "".join(reasoning), "timing": asdict(timing), "ttft": timing.ttft, "e2e": timing.e2e,
            "decode_tps": timing.decode_tps, "output_tokens": timing.output_tokens,
            "token_source": "usage" if tokens is not None else "chunks", "stream_chunks": chunks,
            "prompt_tokens": usage.get("prompt_tokens") if usage else None}


# ---------------------------------------------------------------- run


def run_config(cfg: Config) -> Path:
    cases, chunks = load_cases()

    def msg(i: int, request_id: str) -> list[dict]:
        c = cases[i]
        return messages(c["question"], [chunks[k] for k in c["retrieved"]], request_id)

    RAW.mkdir(parents=True, exist_ok=True)
    out = {"config": asdict(cfg), "settings": {"ctx_per_slot": CTX_PER_SLOT, "parallel": PARALLEL,
           "max_tokens": MAX_TOKENS, "reps": REPS, "sweep": SWEEP, "sweep_requests": SWEEP_REQUESTS},
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cold": [], "warm": [], "sweep": []}
    proc = sampler = None
    with GpuLock(), httpx.Client() as client, (RAW / f"{cfg.name}.server.log").open("w") as log:
        out["host"] = _host()
        try:
            # Cold: launch -> ready -> first token, process restarted each rep. The weights are read
            # once beforehand, so all reps are "process-cold, page-cache-warm" for every runtime.
            out["primed_bytes"] = prime_page_cache(weight_files(cfg))
            for rep in range(REPS):
                if proc is not None:
                    stop(proc)
                t0 = time.perf_counter()
                proc = launch(cfg, log)
                sampler = MemSampler(proc.pid)
                sampler.start()
                wait_ready(cfg, proc)
                t_ready = time.perf_counter()
                r = chat(client, cfg, msg(0, f"k{rep}"))
                out["cold"].append({"rep": rep, "ready_s": t_ready - t0,
                                    "launch_to_first_token_s": r["timing"]["t_first"] - t0, "first_request": r,
                                    **sampler.stop()})
                sampler = None
                print(f"{cfg.name} cold {rep}: {out['cold'][-1]['launch_to_first_token_s']:.1f}s", flush=True)
            sampler = MemSampler(proc.pid)
            sampler.start()
            for i in range(2):
                chat(client, cfg, msg(1, f"u{i}"))  # warmup
            for rep in range(REPS):
                rows = [{"id": c["id"], **chat(client, cfg, msg(i, f"w{rep}-{i}"))} for i, c in enumerate(cases)]
                out["warm"].append({"rep": rep, "results": rows})
                print(f"{cfg.name} warm {rep}: done", flush=True)
            for c in SWEEP:
                for rep in range(REPS):
                    batch = [msg(i, f"s{c}-{rep}-{i}") for i in range(SWEEP_REQUESTS)]
                    with cf.ThreadPoolExecutor(c) as pool:
                        rs = list(pool.map(lambda m: chat(client, cfg, m), batch))
                    out["sweep"].append({"concurrency": c, "rep": rep, "results": rs})
                print(f"{cfg.name} sweep c={c}: done", flush=True)
            out["warm_memory"] = sampler.stop()
            sampler = None
        finally:
            # Never leave a loaded server behind: the lock is released right after this block.
            if sampler is not None:
                sampler.stop()
            if proc is not None:
                stop(proc)
        out["ac_power_at_end"] = on_ac_power()
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path = RAW / f"{cfg.name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    return path


def smoke(cfg: Config) -> None:
    """Two requests through the real server, printed. Checks thinking-off and token accounting."""
    cases, chunks = load_cases()
    with GpuLock(), httpx.Client() as client, open(os.devnull, "w") as log:
        proc = launch(cfg, log)
        try:
            wait_ready(cfg, proc)
            for case in (cases[0], cases[-1]):
                r = chat(client, cfg, messages(case["question"], [chunks[i] for i in case["retrieved"]], "smoke"))
                print(json.dumps({k: r[k] for k in ("text", "output_tokens", "token_source", "stream_chunks",
                                                    "prompt_tokens", "ttft", "decode_tps")}, ensure_ascii=False))
        finally:
            stop(proc)


def _host() -> dict:
    def sh(*cmd):
        p = subprocess.run(cmd, capture_output=True, text=True)
        return (p.stdout + p.stderr).strip()

    return {"chip": sh("sysctl", "-n", "machdep.cpu.brand_string"), "mem_bytes": int(sh("sysctl", "-n", "hw.memsize")),
            "macos": sh("sw_vers", "-productVersion"), "ollama": sh("ollama", "--version"),
            "llama_server": next(ln for ln in sh("llama-server", "--version").splitlines() if ln.startswith("version")) if shutil.which("llama-server") else None,
            "mlx_lm": __import__("mlx_lm").__version__, "power": sh("pmset", "-g", "batt").splitlines()[0]}


def main() -> None:
    ap = argparse.ArgumentParser(prog="bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").add_argument("mastra", type=Path)
    sub.add_parser("run").add_argument("configs", nargs="+", choices=sorted(CONFIGS))
    sub.add_parser("smoke").add_argument("config", choices=sorted(CONFIGS))
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.mastra.resolve())
    elif a.cmd == "run":
        for name in a.configs:
            print(run_config(CONFIGS[name]), flush=True)
    elif a.cmd == "smoke":
        smoke(CONFIGS[a.config])
    else:
        from .report import report

        report()
