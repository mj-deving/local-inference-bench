"""Runner pieces that need no model: server command lines, memory sampler, lock."""

import os
from pathlib import Path

import pytest

from local_inference_bench import bench


@pytest.mark.parametrize("name", sorted(bench.CONFIGS))
def test_server_cmd_for_every_config(name, monkeypatch):
    monkeypatch.setattr(bench, "gguf_blob", lambda tag: Path(f"/blobs/{tag}"))
    cfg = bench.CONFIGS[name]
    cmd, env = bench.server_cmd(cfg)
    assert str(cfg.port) in " ".join(cmd) + env.get("OLLAMA_HOST", "")
    if cfg.runtime == "ollama":
        assert env["OLLAMA_NUM_PARALLEL"] == str(bench.PARALLEL)
        assert env["OLLAMA_CONTEXT_LENGTH"] == str(bench.CTX_PER_SLOT)
    elif cfg.runtime == "llamacpp":
        assert cmd[cmd.index("-c") + 1] == str(bench.CTX_PER_SLOT * bench.PARALLEL)
        assert cmd[cmd.index("-m") + 1] == f"/blobs/{cfg.model}"
        assert cmd[cmd.index("--cache-ram") + 1] == "0"  # no host-RAM cache of repeated prompts
    else:
        assert cmd[0].endswith("mlx_lm.server")
        assert cmd[cmd.index("--decode-concurrency") + 1] == str(bench.PARALLEL)


def test_request_body_has_no_seed_and_disables_thinking_for_every_runtime():
    for cfg in bench.CONFIGS.values():
        body = bench.request_body(cfg, [])
        assert "seed" not in body  # mlx_lm.server does not batch seeded requests
        assert body["temperature"] == 0
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["reasoning_effort"] == "none"


def test_request_ids_make_prompts_unique_but_keep_system_prefix():
    from local_inference_bench.prompt import messages

    chunk = [{"id": "art4-p1", "text": "Text"}]
    a, b = messages("Frage?", chunk, "w0-1"), messages("Frage?", chunk, "w1-1")
    assert a[0] == b[0]
    assert a[1]["content"] != b[1]["content"]
    assert a[1]["content"].endswith("Frage: Frage? /no_think")


def test_server_cmd_rejects_unknown_runtime():
    with pytest.raises(ValueError):
        bench.server_cmd(bench.Config("x", "vllm", "q", "m", 1))


def test_mem_sampler_starts_and_stops_on_python_312():
    # Regression: an attribute named _stop shadowed Thread._stop and broke join() on 3.12.
    s = bench.MemSampler(os.getpid(), period=0.05)
    s.start()
    peaks = s.stop()
    assert peaks["peak_rss_mb"] > 0
    assert not s.is_alive()


def test_gpu_lock_waits_for_ac_takes_and_releases(tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    monkeypatch.setattr(bench, "LOCK", lock)
    monkeypatch.setattr(bench, "on_ac_power", lambda: True)
    with bench.GpuLock():
        assert (lock / "owner").read_text().startswith(bench.SESSION)
    assert not lock.exists()


def test_gpu_lock_never_removes_a_foreign_lock(tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    monkeypatch.setattr(bench, "LOCK", lock)
    monkeypatch.setattr(bench, "on_ac_power", lambda: True)
    g = bench.GpuLock()
    lock.mkdir()
    (lock / "owner").write_text("other-session now\n")
    g.__exit__(None, None, None)
    assert lock.exists()


def test_prime_page_cache_reads_every_byte(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"x" * 1000)
    b.write_bytes(b"")
    assert bench.prime_page_cache([a, b]) == 1000
