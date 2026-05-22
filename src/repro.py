from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

DEFAULT_EXPERIMENT_SEED = 42


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    seed: int
    git_sha: str
    python_version: str
    platform: str


def _safe_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def resolve_experiment_seed(
    *,
    cli_seed: Optional[int] = None,
    env_var: str = "EXPERIMENT_SEED",
    default_seed: int = DEFAULT_EXPERIMENT_SEED,
) -> int:
    if cli_seed is not None:
        return int(cli_seed)
    env_seed = os.getenv(env_var, "").strip()
    if env_seed:
        try:
            return int(env_seed)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {env_var} value {env_seed!r}; expected an integer."
            ) from exc
    return int(default_seed)


def set_global_seed(seed: int) -> dict[str, Any]:
    """
    Best-effort global seeding for reproducibility.
    Note: exact reproducibility can still vary across OS/BLAS/threading.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    reproducibility = {
        "experiment_seed": int(seed),
        "numpy_seed": int(seed),
        "python_random_seed": int(seed),
        "torch_seed": None,
        "torch_deterministic": False,
    }
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
            reproducibility["torch_deterministic"] = True
        except Exception:
            reproducibility["torch_deterministic"] = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        reproducibility["torch_seed"] = int(seed)
    except Exception:
        pass
    return reproducibility


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def build_run_config(seed: int, run_id: Optional[str] = None) -> RunConfig:
    rid = run_id or make_run_id()
    return RunConfig(
        run_id=rid,
        seed=int(seed),
        git_sha=_safe_git_sha(),
        python_version=platform.python_version(),
        platform=platform.platform(),
    )


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
