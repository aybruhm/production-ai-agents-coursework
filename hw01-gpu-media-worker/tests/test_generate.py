import base64
import hashlib
import json
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from backends import EndpointImageBackend
from costs import report
from generate import load_config, run

TOKEN = "test-token-abcd"
FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
HERE = Path(__file__).resolve().parents[1]


class FakeEndpoint(BaseHTTPRequestHandler):
    not_ready_left = 2  # first two /v1/models calls answer 502, like a loading model
    bodies: list[dict] = []

    def log_message(self, format, *args):
        pass

    def _send(self, code: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self):
        if not self._authorized():
            return
        if type(self).not_ready_left > 0:
            type(self).not_ready_left -= 1
            self._send(502, {"error": "failed to connect to local service"})
            return
        self._send(200, {"data": [{"id": "sana"}]})

    def do_POST(self):
        if not self._authorized():
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        png = FAKE_PNG + body["prompt"].encode()  # different bytes per prompt
        self._send(200, {"data": [{"b64_json": base64.b64encode(png).decode()}]})


@pytest.fixture
def endpoint():
    FakeEndpoint.not_ready_left = 2
    FakeEndpoint.bodies = []
    server = HTTPServer(("127.0.0.1", 0), FakeEndpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def small_config():
    cfg = load_config(HERE / "run.toml")
    cfg["prompt"]["subjects"] = cfg["prompt"]["subjects"][:3]
    return cfg


def test_run_records_everything(endpoint, tmp_path):
    cfg = small_config()
    backend = EndpointImageBackend(endpoint, TOKEN, cfg["model"]["id"], poll_s=0.01)
    created = datetime.now(UTC) - timedelta(seconds=60)
    m = run(cfg, backend, tmp_path / "run", created_at=created)

    assert m["status"] == "completed"
    assert len(m["latency_s"]) == len(m["assets"]) == 3
    assert m["create_to_ready_s"] >= 60
    assert m["fixed_s"] == m["create_to_ready_s"]
    assert m["ready_wait_s"] > 0
    assert [a["seed"] for a in m["assets"]] == [7, 8, 9]  # seed + i, recorded per asset
    for asset in m["assets"]:
        data = (tmp_path / "run" / asset["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == asset["sha256"]
    assert FakeEndpoint.bodies[0]["seed"] == 7 and FakeEndpoint.bodies[0]["size"] == "512x512"
    on_disk = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert on_disk == m
    assert TOKEN not in (tmp_path / "run" / "metrics.json").read_text()


def test_without_created_at_fixed_time_is_flagged_lower_bound(endpoint, tmp_path):
    cfg = small_config()
    backend = EndpointImageBackend(endpoint, TOKEN, cfg["model"]["id"], poll_s=0.01)
    m = run(cfg, backend, tmp_path / "run")
    assert "LOWER BOUND" in m["fixed_basis"]
    assert m["fixed_s"] == m["ready_wait_s"]


def test_bad_token_fails_fast_and_leaves_failed_record(endpoint, tmp_path):
    cfg = small_config()
    backend = EndpointImageBackend(endpoint, "wrong", cfg["model"]["id"], poll_s=0.01)
    with pytest.raises(PermissionError):
        run(cfg, backend, tmp_path / "run")
    m = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert m["status"] == "failed" and "PermissionError" in m["error"]


def test_metrics_feed_costs(endpoint, tmp_path):
    cfg = small_config()
    backend = EndpointImageBackend(endpoint, TOKEN, cfg["model"]["id"], poll_s=0.01)
    m = run(cfg, backend, tmp_path / "run", created_at=datetime.now(UTC) - timedelta(seconds=120))
    pricing = {
        "gpu": {"price_per_hour_usd": 0.7492, "basis": "preemptible", "read_at": "x"},
        "api": {"unit": "megapixel", "unit_price_usd": 0.001},
    }
    proj = report(m, pricing)
    assert proj.fixed_s >= 120
    assert proj.breakeven_assets is not None
    assert any("10 assets" in n for n in proj.notes)  # 3-asset test run is flagged
