"""Metric math. No I/O, no model: everything here is unit-tested in tests/test_metrics.py."""

import math
import statistics
from dataclasses import dataclass


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (numpy's default 'linear' method). p in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty list")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    xs = sorted(values)
    rank = (len(xs) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


@dataclass(frozen=True)
class Spread:
    mean: float
    stdev: float  # sample stdev; 0.0 for a single value
    min: float
    max: float
    n: int


def spread(values: list[float]) -> Spread:
    if not values:
        raise ValueError("spread of empty list")
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return Spread(statistics.fmean(values), sd, min(values), max(values), len(values))


@dataclass(frozen=True)
class RequestTiming:
    """Wall-clock timestamps of one streamed request, in seconds (time.perf_counter)."""

    t_send: float
    t_first: float | None  # first content token; None if nothing was generated
    t_last: float | None  # last content token
    t_done: float  # response fully closed
    output_tokens: int

    @property
    def ttft(self) -> float | None:
        return None if self.t_first is None else self.t_first - self.t_send

    @property
    def e2e(self) -> float:
        return self.t_done - self.t_send

    @property
    def decode_tps(self) -> float | None:
        """Tokens/s after the first token: (n - 1) / (t_last - t_first).

        The first token is excluded because its time is prefill (TTFT). Undefined for fewer than
        two tokens or a zero interval.
        """
        if self.t_first is None or self.t_last is None or self.output_tokens < 2:
            return None
        dt = self.t_last - self.t_first
        return None if dt <= 0 else (self.output_tokens - 1) / dt


def throughput(timings: list[RequestTiming]) -> dict[str, float]:
    """Aggregate output tokens/s and requests/s over the wall span of a concurrent batch."""
    if not timings:
        raise ValueError("throughput of empty batch")
    span = max(t.t_done for t in timings) - min(t.t_send for t in timings)
    if span <= 0:
        raise ValueError("non-positive wall span")
    tokens = sum(t.output_tokens for t in timings)
    return {"wall_s": span, "output_tokens": tokens, "tok_per_s": tokens / span, "req_per_s": len(timings) / span}


def cost_per_1k_answers(
    *,
    tok_per_s: float,
    mean_output_tokens: float,
    power_w: float,
    eur_per_kwh: float,
    hardware_eur: float,
    amortization_years: float,
    utilization: float,
) -> dict[str, float]:
    """Local cost of 1000 answers at a measured aggregate decode rate.

    Hardware is amortized over the hours the machine actually serves (years * 8760 * utilization).
    Prefill time is folded into tok_per_s because it is measured end to end under load.
    """
    for name, v in {"tok_per_s": tok_per_s, "power_w": power_w, "amortization_years": amortization_years}.items():
        if v <= 0:
            raise ValueError(f"{name} must be > 0")
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    hours = 1000 * mean_output_tokens / tok_per_s / 3600
    energy = hours * power_w / 1000 * eur_per_kwh
    hardware = hours * hardware_eur / (amortization_years * 8760 * utilization)
    return {"hours": hours, "energy_eur": energy, "hardware_eur": hardware, "total_eur": energy + hardware}


def hosted_cost_per_1k_answers(*, input_tokens: float, output_tokens: float, eur_per_m_in: float, eur_per_m_out: float) -> float:
    return 1000 * (input_tokens * eur_per_m_in + output_tokens * eur_per_m_out) / 1_000_000
