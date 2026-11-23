"""Homework 1: generate N media assets in ONE run and record every time lesson 05 needs.

One run = one startup, one model load, N generations. The output folder holds the assets and
metrics.json; costs.py turns metrics.json into c, F, cost per asset and the API breakeven.

Endpoint (laptop -> Nebius Endpoint), start it right after you click Create:
    uv run python generate.py --created-at 2026-11-24T10:00:00Z

Local GPU (JupyterLab / DevLab), with the startup you read off the console:
    uv run python generate.py --backend local --startup-s 95
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import sys
import time
import tomllib
from datetime import UTC, datetime
from getpass import getpass
from pathlib import Path

from backends import Backend, EndpointImageBackend, ImageRequest, LocalSanaBackend

HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_utc(value: str) -> datetime:
    """Accept '2026-11-24T10:00:00Z' or any ISO 8601 with an offset; naive times are refused."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("--created-at needs a timezone, e.g. ...Z or +00:00")
    return parsed.astimezone(UTC)


def load_config(path: Path) -> dict:
    with path.open("rb") as fh:
        cfg = tomllib.load(fh)
    subjects = cfg["prompt"]["subjects"]
    if not subjects:
        raise ValueError("prompt.subjects is empty")
    if "{subject}" not in cfg["prompt"]["template"]:
        raise ValueError("prompt.template must contain {subject}")
    return cfg


def build_requests(cfg: dict) -> list[ImageRequest]:
    m, p = cfg["model"], cfg["prompt"]
    return [
        ImageRequest(
            prompt=p["template"].format(subject=subject),
            seed=int(m["seed"]) + i,
            size=m["size"],
            num_inference_steps=int(m["num_inference_steps"]),
            guidance_scale=float(m["guidance_scale"]),
            negative_prompt=m.get("negative_prompt", ""),
        )
        for i, subject in enumerate(p["subjects"])
    ]


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("'\"")
    return values


def endpoint_credentials(env_file: Path) -> tuple[str, str]:
    """Environment, then .env, then prompt. The token is never echoed or written to metrics."""
    saved = read_dotenv(env_file)
    base = os.getenv("BASE_URL") or saved.get("BASE_URL") or ""
    token = os.getenv("TOKEN") or saved.get("TOKEN") or ""
    if not base:
        base = input("Endpoint HTTPS URL: ").strip()
    if not token:
        token = getpass("Endpoint token: ").strip()
    if not (saved.get("BASE_URL") and saved.get("TOKEN")):
        env_file.write_text(f"BASE_URL={base}\nTOKEN={token}\n")
        env_file.chmod(0o600)
        print(f"Saved URL and token to {env_file.name} (git-ignored, mode 600)")
    return base, token


def make_backend(cfg: dict, backend_name: str) -> Backend:
    model = cfg["model"]["id"]
    if backend_name == "endpoint":
        base, token = endpoint_credentials(HERE / ".env")
        return EndpointImageBackend(base, token, model)
    if backend_name == "local":
        return LocalSanaBackend(model, revision=cfg["model"].get("revision"))
    raise ValueError(f"unknown backend {backend_name!r}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)  # atomic: a crash never leaves half a metrics file


def run(
    cfg: dict,
    backend: Backend,
    out_dir: Path,
    *,
    created_at: datetime | None = None,
    startup_s: float | None = None,
    ready_deadline_s: float = 900.0,
) -> dict:
    """Wait for readiness (or load), then generate every request once, timing each one."""
    requests = build_requests(cfg)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = out_dir / "metrics.json"

    metrics: dict = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "backend": backend.name,
        "run": cfg["run"],
        "model": cfg["model"],
        "prompt_template": cfg["prompt"]["template"],
        "n_requested": len(requests),
        "timeline_utc": {
            "endpoint_created_at": created_at.isoformat() if created_at else None,
            "script_started_at": utc_now().isoformat(),
        },
        "startup_s": startup_s,
        "ready_wait_s": None,
        "load_s": None,
        "create_to_ready_s": None,
        "ready_to_first_request_s": None,
        "fixed_s": None,
        "fixed_basis": None,
        "latency_s": [],
        "median_latency_s": None,
        "assets": [],
        "env": {},
        "error": None,
    }
    write_json(metrics_path, metrics)

    try:
        # --- fixed part: paid once per run ---------------------------------------------
        metrics["ready_wait_s"] = backend.wait_ready(ready_deadline_s)
        metrics["load_s"] = backend.load()
        ready_at = utc_now()
        metrics["timeline_utc"]["ready_at"] = ready_at.isoformat()

        if backend.name == "endpoint":
            if created_at is not None:
                # Create -> model answers. Covers provisioning, image pull, weight download, load.
                metrics["create_to_ready_s"] = (ready_at - created_at).total_seconds()
                metrics["fixed_s"] = metrics["create_to_ready_s"]
                metrics["fixed_basis"] = "endpoint created_at -> /v1/models ready"
            else:
                metrics["fixed_s"] = metrics["ready_wait_s"]
                metrics["fixed_basis"] = (
                    "script start -> ready (LOWER BOUND: time before the script started is "
                    "missing; pass --created-at next time)"
                )
        else:
            metrics["fixed_s"] = (startup_s or 0.0) + (metrics["load_s"] or 0.0)
            metrics["fixed_basis"] = (
                "startup_s (manual) + measured load_s"
                if startup_s is not None
                else "measured load_s only (startup missing; pass --startup-s)"
            )
        write_json(metrics_path, metrics)

        # --- variable part: paid once per asset ----------------------------------------
        first_request_mono: float | None = None
        ready_mono = time.monotonic()
        for i, req in enumerate(requests):
            start = time.monotonic()
            if first_request_mono is None:
                first_request_mono = start
                metrics["ready_to_first_request_s"] = start - ready_mono
            png = backend.generate(req)
            latency = time.monotonic() - start

            path = assets_dir / f"asset-{i:02d}.png"
            path.write_bytes(png)
            metrics["latency_s"].append(round(latency, 4))
            metrics["assets"].append(
                {
                    "index": i,
                    "file": f"assets/{path.name}",
                    "prompt": req.prompt,
                    "seed": req.seed,
                    "latency_s": round(latency, 4),
                    "bytes": len(png),
                    "sha256": sha256(png),
                }
            )
            print(f"asset {i + 1}/{len(requests)}  {latency:6.2f} s  {path.name}")
            write_json(metrics_path, metrics)  # keep the record current if the run dies later

        metrics["timeline_utc"]["last_response_at"] = utc_now().isoformat()
        metrics["median_latency_s"] = round(statistics.median(metrics["latency_s"]), 4)
        metrics["status"] = "completed"
    except BaseException as exc:  # includes Ctrl-C: always leave an honest record
        metrics["status"] = "failed"
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            metrics["env"] = backend.env()
        except Exception as exc:  # env is best-effort; never mask the real error
            metrics["env"] = {"error": repr(exc)}
        write_json(metrics_path, metrics)
    return metrics


def _sigterm_to_exit(signum: int, _frame: object) -> None:
    # Preemption, `timeout` and `kill` send SIGTERM; turn it into an exception so run()
    # records status "failed" instead of leaving "running" behind.
    raise SystemExit(f"terminated by signal {signum}")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)  # progress shows live in pipes and Job logs
    signal.signal(signal.SIGTERM, _sigterm_to_exit)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=HERE / "run.toml")
    parser.add_argument("--backend", choices=["endpoint", "local"], help="overrides run.toml")
    parser.add_argument(
        "--out",
        type=Path,
        help="run folder (default: runs/<UTC timestamp>); must not exist",
    )
    parser.add_argument(
        "--created-at",
        type=parse_utc,
        help="Endpoint creation time in UTC, from the console or `nebius ai endpoint get`",
    )
    parser.add_argument(
        "--startup-s",
        type=float,
        help="local backend: seconds from DevLab Create until the kernel was usable",
    )
    parser.add_argument("--ready-deadline-s", type=float, default=900.0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    backend_name = args.backend or cfg["run"]["backend"]
    cfg["run"]["backend"] = backend_name
    out_dir = args.out or HERE / "runs" / utc_now().strftime("%Y%m%dT%H%M%SZ")
    if out_dir.exists():
        parser.error(f"{out_dir} already exists: every run gets a fresh folder")

    backend = make_backend(cfg, backend_name)
    print(f"{backend_name}: {len(cfg['prompt']['subjects'])} assets -> {out_dir}")
    try:
        metrics = run(
            cfg,
            backend,
            out_dir,
            created_at=args.created_at,
            startup_s=args.startup_s,
            ready_deadline_s=args.ready_deadline_s,
        )
    finally:
        if backend_name == "endpoint":
            print("\nSTOP THE ENDPOINT NOW: it bills while RUNNING, even when idle.")
    print(
        f"\nmedian {metrics['median_latency_s']:.3f} s | fixed {metrics['fixed_s']:.1f} s "
        f"({metrics['fixed_basis']})\nnext: uv run python costs.py {out_dir / 'metrics.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
