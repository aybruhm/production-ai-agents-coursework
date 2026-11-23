"""Turn one run's metrics.json into the cost numbers Homework 1 asks for (lesson 05 method).

    projected cost per asset = F / N + c
    c = price per second x median generation time       (paid once per asset)
    F = price per second x fixed time per run           (startup + model load + idle)
    breakeven N = ceil(F / (p - c)) against an API price p; 0 means self-hosting never wins

Usage:
    uv run python costs.py runs/<run_id>/metrics.json [--pricing pricing.toml]
                           [--stopped-at 2026-11-24T10:20:00Z]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_COUNTS = (1, 10, 100, 1000)


@dataclass(frozen=True)
class CostProjection:
    price_per_hour_usd: float
    fixed_s: float
    median_latency_s: float
    marginal_usd: float  # c
    fixed_usd: float  # F
    per_asset_usd: dict[int, float]  # F/N + c
    api_price_usd: float | None = None
    breakeven_assets: int | None = None  # None: no API price given; 0: never
    this_run_usd: float | None = None  # F + sum of the actual latencies, at the run's price
    notes: list[str] = field(default_factory=list)


def _check(name: str, value: float) -> None:
    if value is None or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")


def breakeven(fixed_usd: float, marginal_usd: float, api_price_usd: float) -> int:
    """Smallest N per run from which self-hosting is cheaper per asset. 0 when it never is."""
    if api_price_usd <= marginal_usd:
        return 0
    return max(1, math.ceil(fixed_usd / (api_price_usd - marginal_usd)))


def project(
    fixed_s: float,
    latency_s: list[float],
    price_per_hour_usd: float,
    api_price_usd: float | None = None,
    counts: tuple[int, ...] = DEFAULT_COUNTS,
) -> CostProjection:
    if not latency_s:
        raise ValueError("no generation times: the run produced no assets")
    for i, v in enumerate(latency_s):
        _check(f"latency_s[{i}]", v)
    _check("fixed_s", fixed_s)
    _check("price_per_hour_usd", price_per_hour_usd)
    if api_price_usd is not None:
        _check("api_price_usd", api_price_usd)
    if not counts or min(counts) < 1:
        raise ValueError("counts must be positive")

    rate = price_per_hour_usd / 3600.0
    median = statistics.median(latency_s)
    c = rate * median
    f = rate * fixed_s
    return CostProjection(
        price_per_hour_usd=price_per_hour_usd,
        fixed_s=fixed_s,
        median_latency_s=median,
        marginal_usd=c,
        fixed_usd=f,
        per_asset_usd={n: f / n + c for n in counts},
        api_price_usd=api_price_usd,
        breakeven_assets=None if api_price_usd is None else breakeven(f, c, api_price_usd),
        this_run_usd=f + rate * sum(latency_s),
    )


def api_price_per_asset(api: dict, size: str | None) -> float:
    unit, price = api["unit"], float(api["unit_price_usd"])
    if unit == "asset":
        return price
    if unit == "megapixel":
        if not size:
            raise ValueError("megapixel pricing needs model.size in the run")
        width, height = (int(x) for x in size.lower().split("x", 1))
        return width * height / 1e6 * price
    if unit == "video_second":
        return float(api["video_seconds_per_asset"]) * price
    raise ValueError(f"unknown api.unit {unit!r}: use asset, megapixel or video_second")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp needs a timezone: {value!r}")
    return parsed.astimezone(UTC)


def fixed_seconds(metrics: dict, stopped_at: str | None) -> tuple[float, list[str]]:
    """Fixed billed time: what generate.py measured, plus idle the GPU billed with no work."""
    if metrics.get("fixed_s") is None:
        raise ValueError("metrics.json has no fixed_s: the run never became ready")
    total = float(metrics["fixed_s"])
    notes = [f"fixed time basis: {metrics.get('fixed_basis')}"]
    gap = metrics.get("ready_to_first_request_s") or 0.0
    total += gap
    if gap >= 0.05:
        notes.append(f"includes {gap:.1f} s idle between ready and the first request")
    if stopped_at:
        last = metrics.get("timeline_utc", {}).get("last_response_at")
        if not last:
            raise ValueError("--stopped-at given but the run has no last_response_at")
        tail = (_parse_utc(stopped_at) - _parse_utc(last)).total_seconds()
        if tail < 0:
            raise ValueError("--stopped-at is before the last response")
        total += tail
        notes.append(f"includes {tail:.1f} s idle after the last asset until Stop")
    elif metrics.get("backend") == "endpoint":
        notes.append(
            "assumes you stopped the Endpoint right after the last asset; "
            "pass --stopped-at to bill the real tail"
        )
    return total, notes


def report(metrics: dict, pricing: dict, stopped_at: str | None = None) -> CostProjection:
    if metrics.get("status") != "completed":
        raise ValueError(f"run status is {metrics.get('status')!r}, not 'completed'")
    gpu = pricing["gpu"]
    notes: list[str] = []
    billing = metrics.get("run", {}).get("billing")
    if billing and gpu.get("basis") and billing != gpu["basis"]:
        notes.append(f"WARNING: run billed as {billing!r} but pricing.toml uses {gpu['basis']!r}")
    fixed_s, fixed_notes = fixed_seconds(metrics, stopped_at)
    api_price = api_price_per_asset(pricing["api"], metrics.get("model", {}).get("size"))
    proj = project(fixed_s, metrics["latency_s"], float(gpu["price_per_hour_usd"]), api_price)
    proj.notes.extend(notes + fixed_notes)
    if len(metrics["latency_s"]) != 10:
        proj.notes.append(f"homework asks for 10 assets; this run has {len(metrics['latency_s'])}")
    return proj


def to_markdown(proj: CostProjection, metrics: dict, pricing: dict) -> str:
    gpu, api, run = pricing["gpu"], pricing["api"], metrics.get("run", {})
    never = proj.breakeven_assets == 0
    verdict = "(never: the API is cheaper per asset)" if never else "assets per run"
    lines = [
        "| Quantity | Value |",
        "|---|---|",
        f"| Model | `{metrics.get('model', {}).get('id')}` |",
        f"| How run | {metrics.get('backend')} |",
        f"| GPU | {run.get('gpu')} ({run.get('platform')}, {run.get('preset')}, "
        f"{run.get('region')}) |",
        f"| Price used | ${gpu['price_per_hour_usd']}/h, {gpu['basis']}, read {gpu['read_at']} |",
        f"| Assets | {len(metrics['latency_s'])} |",
        f"| Fixed time per run | {proj.fixed_s:.1f} s |",
        f"| Median generation time | {proj.median_latency_s:.3f} s |",
        f"| Marginal cost per asset, c | ${proj.marginal_usd:.6f} |",
        f"| Fixed cost per run, F | ${proj.fixed_usd:.4f} |",
    ]
    lines += [f"| Cost per asset at N={n} | ${v:.6f} |" for n, v in proj.per_asset_usd.items()]
    lines += [
        f"| This run, projected | ${proj.this_run_usd:.4f} |",
        f"| API compared | {api['service']} `{api['model']}`, ${proj.api_price_usd:.6f} per asset "
        f"({api['unit_price_usd']} per {api['unit']}, read {api['read_at']}), {api['url']} |",
        f"| Breakeven N | {proj.breakeven_assets} {verdict} |",
    ]
    if proj.notes:
        lines += ["", *[f"- {n}" for n in proj.notes]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ is not None else ""
    )
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--pricing", type=Path, default=HERE / "pricing.toml")
    parser.add_argument("--stopped-at", help="UTC time you stopped the Endpoint, e.g. ...Z")
    args = parser.parse_args(argv)

    metrics = json.loads(args.metrics.read_text())
    with args.pricing.open("rb") as fh:
        pricing = tomllib.load(fh)
    proj = report(metrics, pricing, args.stopped_at)

    out = args.metrics.parent
    data = asdict(proj) | {"pricing": pricing, "stopped_at": args.stopped_at}
    data["per_asset_usd"] = {str(k): v for k, v in proj.per_asset_usd.items()}
    (out / "cost_report.json").write_text(json.dumps(data, indent=2) + "\n")
    md = to_markdown(proj, metrics, pricing)
    (out / "cost_report.md").write_text(md)
    print(md)
    print(f"wrote {out / 'cost_report.json'} and {out / 'cost_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
