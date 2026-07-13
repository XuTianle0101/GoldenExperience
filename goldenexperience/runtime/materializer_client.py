"""Lifecycle client for the resident cached-KV materializer worker."""

from __future__ import annotations

import json
import select
import subprocess
from pathlib import Path
from typing import Any, TextIO


class MaterializerClientError(RuntimeError):
    """Raised when the resident worker cannot return a valid response."""


class ResidentMaterializerClient:
    """Keep bridge tensors resident and exchange one JSON object per line."""

    def __init__(
        self,
        *,
        python_bin: str,
        cwd: str | Path,
        stderr_path: str | Path,
        timeout_sec: float = 600.0,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("materializer timeout must be positive")
        self.python_bin = python_bin
        self.cwd = Path(cwd)
        self.stderr_path = Path(stderr_path)
        self.timeout_sec = timeout_sec
        self._process: subprocess.Popen[str] | None = None
        self._stderr: TextIO | None = None
        self._request_count = 0

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def start(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                return
            raise MaterializerClientError(
                f"materializer worker already exited with code {self._process.returncode}"
            )
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = self.stderr_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            [
                self.python_bin,
                "-m",
                "goldenexperience.runtime.cross_model_materializer",
                "--serve-jsonl",
            ],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.start()
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        if self._process.poll() is not None:
            raise MaterializerClientError(
                f"materializer worker exited with code {self._process.returncode}"
            )
        try:
            line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MaterializerClientError("could not write to materializer worker") from exc

        ready, _, _ = select.select([self._process.stdout], [], [], self.timeout_sec)
        if not ready:
            raise MaterializerClientError(
                f"materializer worker timed out after {self.timeout_sec:.1f} seconds"
            )
        response_line = self._process.stdout.readline()
        if not response_line:
            returncode = self._process.poll()
            raise MaterializerClientError(
                f"materializer worker closed stdout with code {returncode}"
            )
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise MaterializerClientError("materializer worker returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise MaterializerClientError("materializer worker response must be an object")
        self._request_count += 1
        return response

    def process_info(self) -> dict[str, Any]:
        process = self._process
        return {
            "mode": "resident_jsonl",
            "pid": process.pid if process is not None else None,
            "alive": process is not None and process.poll() is None,
            "returncode": process.poll() if process is not None else None,
            "request_count": self._request_count,
            "stderr_path": str(self.stderr_path),
        }

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
        self._process = None
        self._stderr = None

    def __enter__(self) -> ResidentMaterializerClient:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
