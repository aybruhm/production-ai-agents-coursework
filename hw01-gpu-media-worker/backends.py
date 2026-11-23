"""Two ways to make an image: call an Endpoint over HTTP, or run the model on the local GPU.

Both expose the same interface, so generate.py measures them the same way:

    backend.wait_ready(deadline_s) -> float | None   seconds spent waiting (Endpoint only)
    backend.load() -> float | None                   seconds to load weights (local only)
    backend.generate(request) -> bytes               one PNG
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
RETRYABLE_HTTP = {429, 502, 503, 504}  # 502 = container up, model still loading


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    seed: int
    size: str
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: str = ""

    @property
    def width_height(self) -> tuple[int, int]:
        try:
            width, height = (int(part) for part in self.size.lower().split("x", 1))
        except ValueError as exc:
            raise ValueError(f"size must look like 512x512, got {self.size!r}") from exc
        for side in (width, height):
            if not 256 <= side <= 2048 or side % 32:
                raise ValueError(f"each side must be a multiple of 32 in 256..2048: {self.size!r}")
        return width, height


class Backend(Protocol):
    name: str

    def wait_ready(self, deadline_s: float) -> float | None: ...
    def load(self) -> float | None: ...
    def generate(self, request: ImageRequest) -> bytes: ...
    def env(self) -> dict[str, object]: ...


class EndpointImageBackend:
    """OpenAI-style image Endpoint: GET /v1/models, POST /v1/images/generations (b64_json)."""

    name = "endpoint"

    def __init__(
        self,
        base_url: str,
        token: str,
        model: str,
        timeout_s: float = 300.0,
        poll_s: float = 2.0,
    ):
        base = base_url.strip().rstrip("/").removesuffix("/v1")
        if not base.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError("BASE_URL must be the Endpoint's https:// URL")
        if not token:
            raise ValueError("TOKEN is empty")
        self.base_url = base
        self._token = token
        self.model = model
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    def _call(self, route: str, body: dict | None = None, timeout: float = 30.0) -> dict:
        headers = {"Authorization": f"Bearer {self._token}"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + route, headers=headers, data=data)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    def wait_ready(self, deadline_s: float = 900.0) -> float:
        """Poll /v1/models until it lists a model. RUNNING alone doesn't mean it is loaded."""
        start = time.monotonic()
        last_error = "no response yet"
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= deadline_s:
                raise TimeoutError(
                    f"Not ready after {deadline_s:.0f} s ({last_error}). "
                    "Check the Endpoint logs, and stop it if it's stuck: it bills while RUNNING."
                )
            try:
                data = self._call("/v1/models", timeout=min(30.0, deadline_s - elapsed))
                if isinstance(data.get("data"), list) and data["data"]:
                    return time.monotonic() - start
                last_error = "empty model list"
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise PermissionError(f"HTTP {exc.code}: wrong or missing token") from exc
                if exc.code not in RETRYABLE_HTTP:
                    raise
                last_error = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = type(exc).__name__
            time.sleep(self.poll_s)

    def load(self) -> None:
        return None  # happens inside the Endpoint; folded into the readiness wait

    def generate(self, request: ImageRequest) -> bytes:
        _ = request.width_height  # validate before paying for a request
        body = {
            "model": self.model,
            "prompt": request.prompt,
            "size": request.size,
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "negative_prompt": request.negative_prompt,
            "seed": request.seed,
        }
        try:
            payload = self._call("/v1/images/generations", body, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            detail = exc.read(500).decode(errors="replace") if exc.fp else ""
            raise RuntimeError(f"HTTP {exc.code} from /v1/images/generations: {detail}") from exc
        png = base64.b64decode(payload["data"][0]["b64_json"], validate=True)
        if not png.startswith(PNG_MAGIC):
            raise ValueError("Endpoint response is not a PNG")
        return png

    def env(self) -> dict[str, object]:
        return {"base_url_host": self.base_url.split("//", 1)[-1].split("/", 1)[0]}


class LocalSanaBackend:
    """Sana loaded with Diffusers on the local CUDA GPU (JupyterLab / DevLab).

    Same recipe as the course: fp16 transformer, text encoder and VAE in bf16
    (the model card says they must stay bf16 or fp32, or images come out black).
    """

    name = "local"

    def __init__(self, model: str, revision: str | None = None, variant: str = "fp16"):
        self.model = model
        self.revision = revision
        self.variant = variant
        self._pipe = None
        self.peak_vram_gb: float | None = None

    def wait_ready(self, deadline_s: float = 0.0) -> None:
        return None

    def load(self) -> float:
        import torch
        from diffusers import SanaPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: use the GPU kernel (uv sync --extra gpu)")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        pipe = SanaPipeline.from_pretrained(
            self.model, revision=self.revision, variant=self.variant, torch_dtype=torch.float16
        ).to("cuda")
        pipe.text_encoder.to(torch.bfloat16)
        pipe.vae.to(torch.bfloat16)
        pipe.set_progress_bar_config(disable=True)
        torch.cuda.synchronize()
        self._pipe = pipe
        return time.perf_counter() - start

    def generate(self, request: ImageRequest) -> bytes:
        import io

        import torch

        if self._pipe is None:
            raise RuntimeError("call load() first")
        width, height = request.width_height
        rng = torch.Generator(device="cuda").manual_seed(request.seed)
        image = self._pipe(
            prompt=request.prompt,
            width=width,
            height=height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            negative_prompt=request.negative_prompt,
            generator=rng,
        ).images[0]
        torch.cuda.synchronize()  # GPU work is async: sync so the timer covers the real work
        self.peak_vram_gb = max(self.peak_vram_gb or 0.0, torch.cuda.max_memory_allocated() / 1e9)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def env(self) -> dict[str, object]:
        import diffusers
        import torch

        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "peak_vram_gb": self.peak_vram_gb,
        }
