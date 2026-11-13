import math

import pytest

from local_inference_bench.metrics import (
    RequestTiming,
    cost_per_1k_answers,
    hosted_cost_per_1k_answers,
    percentile,
    spread,
    throughput,
)


def test_percentile_matches_linear_definition():
    xs = [10, 1, 4, 3, 2]  # sorted: 1 2 3 4 10
    assert percentile(xs, 50) == 3
    assert percentile(xs, 0) == 1
    assert percentile(xs, 100) == 10
    assert percentile(xs, 95) == pytest.approx(4 + 0.8 * 6)  # rank 3.8
    assert percentile([7.0], 95) == 7.0


@pytest.mark.parametrize("p", [-0.1, 100.1])
def test_percentile_rejects_out_of_range(p):
    with pytest.raises(ValueError):
        percentile([1, 2], p)


def test_percentile_rejects_empty():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_spread_single_value_has_zero_stdev():
    s = spread([2.5])
    assert (s.mean, s.stdev, s.n) == (2.5, 0.0, 1)


def test_spread_uses_sample_stdev():
    assert spread([1, 2, 3]).stdev == pytest.approx(1.0)


def test_request_timing_derivations():
    t = RequestTiming(t_send=10.0, t_first=10.5, t_last=12.5, t_done=12.6, output_tokens=41)
    assert t.ttft == pytest.approx(0.5)
    assert t.e2e == pytest.approx(2.6)
    assert t.decode_tps == pytest.approx(40 / 2.0)  # first token excluded


@pytest.mark.parametrize(
    "first,last,n",
    [(None, None, 0), (1.0, 1.0, 1), (1.0, 1.0, 5), (2.0, 1.0, 5)],
)
def test_decode_tps_undefined_cases(first, last, n):
    assert RequestTiming(0.0, first, last, 3.0, n).decode_tps is None


def test_throughput_uses_wall_span_not_sum_of_latencies():
    # Two fully overlapping 2 s requests: 2x the tokens in the same 2 s.
    a = RequestTiming(0.0, 0.5, 1.9, 2.0, 100)
    b = RequestTiming(0.0, 0.6, 1.9, 2.0, 100)
    tp = throughput([a, b])
    assert tp["tok_per_s"] == pytest.approx(100.0)
    assert tp["req_per_s"] == pytest.approx(1.0)


def test_throughput_rejects_degenerate():
    with pytest.raises(ValueError):
        throughput([])
    with pytest.raises(ValueError):
        throughput([RequestTiming(1.0, None, None, 1.0, 0)])


def test_local_cost_hand_computed():
    # 1000 answers * 360 tokens / 100 tok/s = 3600 s = 1 h
    c = cost_per_1k_answers(
        tok_per_s=100, mean_output_tokens=360, power_w=50, eur_per_kwh=0.40,
        hardware_eur=8760, amortization_years=1, utilization=1.0,
    )
    assert c["hours"] == pytest.approx(1.0)
    assert c["energy_eur"] == pytest.approx(0.02)
    assert c["hardware_eur"] == pytest.approx(1.0)
    assert math.isclose(c["total_eur"], 1.02)


@pytest.mark.parametrize("bad", [{"tok_per_s": 0}, {"utilization": 0}, {"utilization": 1.5}, {"power_w": -1}])
def test_local_cost_rejects_invalid(bad):
    args = dict(tok_per_s=100, mean_output_tokens=360, power_w=50, eur_per_kwh=0.4,
                hardware_eur=1000, amortization_years=3, utilization=0.5)
    args.update(bad)
    with pytest.raises(ValueError):
        cost_per_1k_answers(**args)


def test_hosted_cost():
    assert hosted_cost_per_1k_answers(input_tokens=2000, output_tokens=100, eur_per_m_in=0.1, eur_per_m_out=0.4) == pytest.approx(0.24)
