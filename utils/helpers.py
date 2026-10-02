"""
utils/helpers.py
================
Boring but necessary infrastructure: seeding, results directory, CSV writing
and logging. Nothing quantum happens in this file - it exists so the
scientific modules stay readable.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import config

# Module-level cache so repeated setup_logger() calls do not attach duplicate
# handlers (which would print every line two, three, four times).
_CONFIGURED_LOGGERS: set[str] = set()


def set_seed(seed: int = config.SEED) -> int:
    """Seed every random number generator we rely on.

    What it does
    ------------
    Seeds Python's `random`, NumPy's global RNG, and the `PYTHONHASHSEED`
    environment variable (for dict/set ordering in subprocesses).

    A note on Stim
    --------------
    Stim deliberately has NO global seed. Its samplers take a `seed=` argument
    instead, because a global RNG would make multi-threaded sampling
    non-reproducible. Every place we create a Stim sampler in this project
    therefore passes an explicit seed derived from config.SEED. This function
    returns the seed so callers can forward it.

    Parameters
    ----------
    seed : int
        The seed to install. Defaults to config.SEED.

    Returns
    -------
    int
        The seed that was set (handy for `seed = set_seed(...)`).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def setup_results_dir(path: str = config.RESULTS_DIR) -> str:
    """Create the results directory if it does not exist.

    Parameters
    ----------
    path : str
        Directory to create. Defaults to config.RESULTS_DIR.

    Returns
    -------
    str
        The absolute path of the directory.
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    # .gitkeep so the (otherwise empty) folder survives a git clone.
    keep = os.path.join(abs_path, ".gitkeep")
    if not os.path.exists(keep):
        open(keep, "a").close()
    return abs_path


def results_path(filename: str) -> str:
    """Join a filename onto the results directory.

    Every artefact in the project goes through this function so that changing
    config.RESULTS_DIR relocates all outputs at once.

    Parameters
    ----------
    filename : str
        Bare file name, e.g. "drift_results.csv".

    Returns
    -------
    str
        Path inside the results directory.
    """
    setup_results_dir()
    return os.path.join(config.RESULTS_DIR, filename)


def save_to_csv(data: Any, filename: str) -> str:
    """Write tabular data to results/<filename> as CSV.

    Parameters
    ----------
    data : pandas.DataFrame | list[dict] | dict[str, list]
        Anything pandas can turn into a DataFrame.
    filename : str
        Bare file name (no directory), e.g. "baseline_results.csv".

    Returns
    -------
    str
        The full path that was written.
    """
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    path = results_path(filename)
    df.to_csv(path, index=False)
    return path


def save_json(data: Mapping[str, Any], filename: str) -> str:
    """Write a dictionary to results/<filename> as pretty-printed JSON.

    Used for small metadata blobs (learned policies, verification results)
    that do not fit a table.

    Parameters
    ----------
    data : mapping
    filename : str

    Returns
    -------
    str
        The full path that was written.
    """
    path = results_path(filename)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_default)
    return path


def setup_logger(name: str = "quantum_rl_decoder") -> logging.Logger:
    """Create (or fetch) a logger that writes to console and results/run.log.

    Why both: the console keeps you informed while an experiment runs, the
    file gives you an artefact you can paste into a report or diff between
    runs.

    Parameters
    ----------
    name : str
        Logger name, conventionally the module doing the logging.

    Returns
    -------
    logging.Logger
        Logger at INFO level: INFO and above go to results/run.log, WARNING
        and above also reach the console.
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console shows warnings and errors only. The experiments print their own
    # formatted progress and tables to stdout, and interleaved INFO lines from
    # the logging module would corrupt those tables. The full INFO trace still
    # goes to results/run.log.
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(results_path(config.LOG_FILE), mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger


def banner(title: str, width: int = 70, char: str = "=") -> str:
    """Return a centred banner string for readable console sections.

    Parameters
    ----------
    title : str
    width : int
    char : str

    Returns
    -------
    str
        Three lines: rule, centred title, rule.
    """
    line = char * width
    return f"\n{line}\n{title.center(width)}\n{line}"


def format_policy(policy: Sequence[int]) -> str:
    """Render a policy (state -> action) as a readable one-liner.

    Parameters
    ----------
    policy : sequence of 4 ints

    Returns
    -------
    str
        e.g. "[0,0]->no-op | [0,1]->flip q2 | [1,0]->flip q0 | [1,1]->flip q1"
    """
    parts = []
    for state, action in enumerate(policy):
        parts.append(f"{config.STATE_LABELS[state]}->{config.ACTION_LABELS[int(action)]}")
    return " | ".join(parts)


def seeds_for_experiment(num_seeds: int = config.NUM_SEEDS, base: int = config.SEED) -> list[int]:
    """Deterministically derive a list of distinct seeds from the base seed.

    Multi-seed repetition is how we separate "the agent learned something"
    from "the agent got lucky". Deriving the seeds from config.SEED keeps the
    whole sweep reproducible from a single number.

    Parameters
    ----------
    num_seeds : int
    base : int

    Returns
    -------
    list of int
        e.g. [42, 1042, 2042, 3042, 4042]
    """
    return [int(base + 1000 * i) for i in range(int(num_seeds))]


def as_percent(x: float, digits: int = 4) -> str:
    """Format a fraction as a percentage string.

    Parameters
    ----------
    x : float
    digits : int

    Returns
    -------
    str
    """
    return f"{100.0 * float(x):.{digits}f}%"


def iter_chunks(items: Iterable[Any], size: int):
    """Yield successive chunks of an iterable (small utility for batching).

    Parameters
    ----------
    items : iterable
    size : int

    Yields
    ------
    list
        Chunks of at most `size` elements.
    """
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
