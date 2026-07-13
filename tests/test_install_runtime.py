import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_runtime.sh"


def _fake_python(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "python-commands.log"
    executable = tmp_path / "python"
    executable.write_text(
        """#!/usr/bin/env bash
printf '%s ' "$@" >> "$GE_TEST_COMMAND_LOG"
printf '\n' >> "$GE_TEST_COMMAND_LOG"
if [ "${1:-}" = "-" ]; then
  cat >/dev/null
fi
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, log


def _install_env(tmp_path: Path, cuda_major: str) -> tuple[dict[str, str], Path]:
    python, log = _fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(python),
            "GE_TEST_COMMAND_LOG": str(log),
            "GE_USE_UV": "0",
            "GE_CUDA_MAJOR": cuda_major,
            "GE_PATCH_MOONCAKE_RUNTIME": "0",
            "GE_RUNTIME_CHECK": "skip",
        }
    )
    return env, log


def test_package_install_uses_verified_stack_without_forcing_cupy(tmp_path: Path) -> None:
    env, log = _install_env(tmp_path, "13")

    completed = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--mode", "package", "--no-dev"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    commands = log.read_text(encoding="utf-8")
    assert "vllm==0.24.0" in commands
    assert "lmcache==0.4.6" in commands
    assert "cupy" not in commands
    assert "uninstall" not in commands
    assert "--no-deps" not in commands


def test_package_install_rejects_unsupported_cuda_before_pip(tmp_path: Path) -> None:
    env, log = _install_env(tmp_path, "12")

    completed = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--mode", "package"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "No packages were changed" in completed.stderr
    commands = log.read_text(encoding="utf-8")
    assert "pip" not in commands


def test_install_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(INSTALL_SCRIPT)], cwd=REPO_ROOT, check=True)
