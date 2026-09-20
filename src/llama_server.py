"""Raw-completion inference against a persistent llama-server. See RDR-004.

/completion applies no chat template, unlike llama-cli. Requests run
sequentially on a single slot for deterministic decoding.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

DEFAULT_BIN = "./llama.cpp/build/bin/llama-server"


class LlamaServerError(RuntimeError):
    pass


def _post(url: str, payload: dict, timeout: int = 300, retries: int = 3) -> dict:
    """POST with a short retry. A multi-seed matrix issues tens of thousands of
    requests; one transient failure should not lose a whole run."""
    data = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
    raise LlamaServerError(f"{url} failed after {retries} attempts: {last}")


def _wait_for_health(port: int, proc: subprocess.Popen, timeout_s: int = 180) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise LlamaServerError(
                f"llama-server exited early with code {proc.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise LlamaServerError(f"llama-server did not become healthy within {timeout_s}s")


@contextmanager
def llama_server(
    model_path: str | Path,
    port: int = 8099,
    n_gpu_layers: int = 99,
    ctx_size: int = 1024,
    binary: str = DEFAULT_BIN,
    extra_args: list[str] | None = None,
):
    """Run one server for the lifetime of the block."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    cmd = [
        binary,
        "-m", str(model_path),
        "--port", str(port),
        "--host", "127.0.0.1",
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx_size),
        "--parallel", "1",          # deterministic decode
        "--no-webui",
    ] + (extra_args or [])

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_health(port, proc)
        yield LlamaClient(port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


class LlamaClient:
    """Client for the raw /completion endpoint."""

    def __init__(self, port: int):
        self.url = f"http://127.0.0.1:{port}/completion"

    def complete(self, prompt: str, n_predict: int = 10) -> str:
        """Greedy raw completion. No chat template."""
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "seed": 42,
            "cache_prompt": False,   # no cross-sample state
            "stream": False,
            "stop": ["\n"],
        }
        result = _post(self.url, payload)
        return result.get("content", "")

    def tokenize(self, text: str) -> list[int]:
        """Token ids as the engine sees them."""
        url = self.url.replace("/completion", "/tokenize")
        return _post(url, {"content": text}).get("tokens", [])
