"""
evaluation/metrics.py
=====================
All of the project's maths-on-results and all of its plotting.

-----------------------------------------------------------------------
THE METRIC
-----------------------------------------------------------------------
Logical Error Rate (LER)
    The fraction of shots on which the decoder's prediction of the logical
    observable disagreed with reality. In plain terms: how often the encoded
    bit was corrupted despite error correction. Lower is better, and it is
    plotted on a log axis because it spans orders of magnitude across the
    physical error rates we sweep.

Why error bars matter here
    LER is a binomial proportion. With NUM_SHOTS = 10,000 shots at p = 0.01,
    the true LER is ~3e-4, so you expect only ~3 failures per run. Quoting
    "LER = 0.0003" without an interval would be dishonest at that resolution.
    We report a bootstrap percentile interval, which needs no normal
    approximation and behaves sensibly even when the count of failures is 0.

pseudo-threshold (a term you will meet in the QEC literature)
    The physical error rate below which the encoded logical qubit is more
    reliable than a single unencoded physical qubit (LER < p). For a
    single-round 3-qubit repetition code the LER is ~3p^2, so the crossover
    sits near p = 1/3 in the ideal case; with only two parity checks and one
    round, the useful regime is small p.
-----------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib

# Rule 10: figures are written to disk, never shown. The Agg backend makes
# the code work on headless machines (servers, CI, WSL without an X server).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402

import config  # noqa: E402
from training.qlearner import moving_average  # noqa: E402
from utils.helpers import results_path  # noqa: E402



# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------
def compute_ler(predictions, actuals) -> float:
    """Logical error rate = fraction of shots where the prediction was wrong.

    Quantum concepts
    ----------------
    `actuals` are the sampled values of the circuit's logical OBSERVABLE:
    1 means the encoded bit came out flipped by the physical noise.
    `predictions` are what the decoder believed, given only the syndrome.
    A mismatch is a *logical error*: the correction the decoder would apply
    leaves the encoded information wrong.

    Parameters
    ----------
    predictions : array-like, shape (num_shots, 1) or (num_shots,)
    actuals : array-like, shape (num_shots, 1) or (num_shots,)

    Returns
    -------
    float
        LER in [0, 1].
    """
    predictions = np.asarray(predictions).astype(bool).reshape(-1)
    actuals = np.asarray(actuals).astype(bool).reshape(-1)
    if predictions.shape != actuals.shape:
        raise ValueError(
            f"shape mismatch: predictions {predictions.shape} vs actuals {actuals.shape}"
        )
    if predictions.size == 0:
        raise ValueError("cannot compute LER from zero shots")
    return float(np.mean(predictions != actuals))


def compute_ler_with_confidence(
    predictions,
    actuals,
    num_bootstrap: int = config.NUM_BOOTSTRAP,
    seed: int = config.SEED,
    confidence: float = config.CONFIDENCE_LEVEL,
) -> tuple[float, float, float]:
    """LER plus a bootstrap confidence interval.

    Method: resample the per-shot outcome array with replacement
    `num_bootstrap` times, recompute the LER each time, and take percentiles
    of the resulting distribution. This is the standard non-parametric way to
    put uncertainty on a Monte-Carlo estimate, and unlike a normal
    approximation it does not produce negative lower bounds when the LER is
    tiny.

    Parameters
    ----------
    predictions : array-like, shape (num_shots, 1) or (num_shots,)
    actuals : array-like, shape (num_shots, 1) or (num_shots,)
    num_bootstrap : int
        Number of resamples. Defaults to config.NUM_BOOTSTRAP.
    seed : int
        Seed for the resampling RNG (reproducible error bars).
    confidence : float
        Confidence level, e.g. 0.95 for a 95% interval.

    Returns
    -------
    mean_ler : float
        The point estimate (computed on the original data, not the bootstrap
        mean, so it matches compute_ler exactly).
    lower_ci : float
    upper_ci : float
    """
    predictions = np.asarray(predictions).astype(bool).reshape(-1)
    actuals = np.asarray(actuals).astype(bool).reshape(-1)
    errors = (predictions != actuals).astype(np.float64)
    n = errors.size
    if n == 0:
        raise ValueError("cannot compute LER from zero shots")

    point = float(errors.mean())

    rng = np.random.default_rng(int(seed))
    # Resample indices in one vectorised draw: (num_bootstrap, n).
    idx = rng.integers(0, n, size=(int(num_bootstrap), n))
    boot = errors[idx].mean(axis=1)

    alpha = (1.0 - float(confidence)) / 2.0
    lower = float(np.percentile(boot, 100 * alpha))
    upper = float(np.percentile(boot, 100 * (1 - alpha)))
    return point, lower, upper


def theoretical_ler(p: float) -> float:
    """Analytic LER of the optimal decoder for one round of this code.

    Derivation: the minimum-weight decoder succeeds exactly when at most one
    of the three data qubits was hit, so

        LER(p) = 1 - [(1-p)^3 + 3p(1-p)^2] = 3p^2(1-p) + p^3 = 3p^2 - 2p^3

    The leading 3p^2 term is the textbook statement that a distance-3 code
    (which corrects one error) suppresses the error rate quadratically.

    Parameters
    ----------
    p : float
        Physical error rate.

    Returns
    -------
    float
        The exact logical error rate an optimal decoder achieves.
    """
    p = float(p)
    return 3 * p**2 - 2 * p**3


def relative_degradation(ler: Sequence[float], reference: Sequence[float]) -> list[float]:
    """Ratio LER / reference_LER, element-wise, guarding against division by 0.

    Used by the drift experiment to answer "how many times worse than the
    oracle decoder is this decoder at each noise rate?".

    Parameters
    ----------
    ler : sequence of float
    reference : sequence of float

    Returns
    -------
    list of float
        NaN wherever the reference LER is exactly 0 (undefined ratio).
    """
    out = []
    for a, b in zip(ler, reference):
        out.append(float("nan") if b == 0 else float(a) / float(b))
    return out


# ---------------------------------------------------------------------------
# Plotting (rule 10: always savefig, never show)
# ---------------------------------------------------------------------------
def plot_ler_comparison(
    noise_rates: Sequence[float],
    ler_dict: Mapping[str, Sequence[float]],
    title: str,
    filename: str,
    ci_dict: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    show_physical_line: bool = True,
    show_theory: bool = True,
    xlabel: str = r"physical error rate  $p$",
) -> str:
    """Plot one or more LER-vs-noise curves on a log y-axis and save the figure.

    Parameters
    ----------
    noise_rates : sequence of float
        X-axis: physical error rates.
    ler_dict : mapping
        {"MWPM Fixed": [...], "RL": [...], "MWPM Oracle": [...]} - one entry
        per curve, each the same length as `noise_rates`.
    title : str
        Figure title.
    filename : str
        Bare file name; the figure is written into config.RESULTS_DIR.
    ci_dict : mapping, optional
        {label: (lower_list, upper_list)} bootstrap bounds, drawn as error
        bars. Labels must match those in `ler_dict`.
    show_physical_line : bool
        Draw the y = p reference line ("no error correction at all"). Curves
        below this line mean the encoding is helping.
    show_theory : bool
        Draw the analytic optimum 3p^2 - 2p^3 as a dashed reference.
    xlabel : str
        X-axis label. Override it when the x-axis is not a noise rate (the
        bias study sweeps an asymmetry strength instead).

    Returns
    -------
    str
        Path of the saved PNG.

    Notes
    -----
    A LER of exactly 0 (possible at small p with finite shots) cannot be drawn
    on a log axis. Such points are lifted to half of the smallest resolvable
    rate, 1/(2 * NUM_SHOTS), and marked hollow, which is the standard way QEC
    papers display "no failures observed".
    """
    noise_rates = np.asarray(noise_rates, dtype=float)
    floor = 1.0 / (2.0 * config.NUM_SHOTS)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Distinct, colour-blind-safe-ish styles so curves stay distinguishable in
    # a printed report.
    # Curves in this study very often COINCIDE exactly (that is itself a
    # result). Identical curves would hide each other, so each successive
    # curve is drawn thinner, with a different dash pattern and a different
    # marker: where they overlap you see a striped, multi-marker line rather
    # than a single line that looks like a missing decoder.
    styles = [
        {"color": "#0072B2", "marker": "o", "linestyle": "-", "linewidth": 4.0,
         "markersize": 11, "alpha": 0.45},
        {"color": "#D55E00", "marker": "s", "linestyle": "--", "linewidth": 2.4,
         "markersize": 7.5, "alpha": 0.95},
        {"color": "#009E73", "marker": "^", "linestyle": ":", "linewidth": 1.6,
         "markersize": 5.5, "alpha": 1.0},
        {"color": "#CC79A7", "marker": "D", "linestyle": "-.", "linewidth": 1.2,
         "markersize": 4.5, "alpha": 1.0},
        {"color": "#56B4E9", "marker": "v", "linestyle": "-", "linewidth": 1.0,
         "markersize": 4.0, "alpha": 1.0},
    ]

    for i, (label, values) in enumerate(ler_dict.items()):
        values = np.asarray(values, dtype=float)
        plotted = np.where(values <= 0, floor, values)
        style = styles[i % len(styles)]

        yerr = None
        if ci_dict and label in ci_dict:
            lower = np.asarray(ci_dict[label][0], dtype=float)
            upper = np.asarray(ci_dict[label][1], dtype=float)
            lower = np.clip(lower, floor, None)
            upper = np.clip(upper, floor, None)
            # Error bars must be *distances* from the point, and non-negative.
            yerr = np.vstack(
                [
                    np.clip(plotted - lower, 0, None),
                    np.clip(upper - plotted, 0, None),
                ]
            )

        ax.errorbar(
            noise_rates,
            plotted,
            yerr=yerr,
            label=label,
            capsize=3,
            **style,
        )

        # Hollow overlay for "zero failures observed" points.
        zero_mask = values <= 0
        if np.any(zero_mask):
            ax.scatter(
                noise_rates[zero_mask],
                np.full(int(zero_mask.sum()), floor),
                facecolors="white",
                edgecolors=style["color"],
                zorder=5,
                s=70,
                linewidths=1.5,
            )

    if show_physical_line:
        ax.plot(
            noise_rates,
            noise_rates,
            color="grey",
            linestyle=":",
            linewidth=1.4,
            label="no encoding (LER = p)",
        )

    if show_theory:
        dense = np.linspace(min(noise_rates), max(noise_rates), 200)
        ax.plot(
            dense,
            [theoretical_ler(p) for p in dense],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=r"theory: $3p^2-2p^3$",
        )

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("logical error rate  (LER)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = results_path(filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_curve(
    reward_history: Sequence[float],
    filename: str,
    window: int = 500,
    optimal_reward: float | None = None,
    epsilon_history: Sequence[float] | None = None,
    title: str = "Tabular Q-learning on the 3-qubit repetition code",
) -> str:
    """Plot the smoothed training reward curve (and the epsilon schedule).

    How to read it
    --------------
    The y-axis is the average reward per episode, where +1 = the logical bit
    survived and -1 = it was lost. It starts near the random-policy value and
    should climb towards - but not quite reach - the greedy optimum, because
    EPSILON_END = 0.05 keeps 5% of actions random forever.

    Parameters
    ----------
    reward_history : sequence of float
        Per-episode rewards from TabularQLearner.train.
    filename : str
        Bare file name inside the results directory.
    window : int
        Moving-average window in episodes.
    optimal_reward : float, optional
        Horizontal reference line, e.g. 2 * P(success) - 1 for the optimal
        greedy policy.
    epsilon_history : sequence of float, optional
        If given, plotted on a twin axis.
    title : str

    Returns
    -------
    str
        Path of the saved PNG.
    """
    from training.qlearner import moving_average  # local import avoids a cycle

    rewards = np.asarray(reward_history, dtype=float)
    smoothed = moving_average(rewards, window=window)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(smoothed)) + window
    ax.plot(x, smoothed, color="#0072B2", linewidth=1.6, label=f"mean reward ({window}-episode window)")

    if optimal_reward is not None:
        ax.axhline(
            optimal_reward,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"greedy optimum ({optimal_reward:.4f})",
        )

    ax.set_xlabel("episode")
    ax.set_ylabel("average reward  (+1 survived / -1 logical error)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if epsilon_history is not None:
        ax2 = ax.twinx()
        ax2.plot(
            np.arange(len(epsilon_history)),
            np.asarray(epsilon_history, dtype=float),
            color="#D55E00",
            alpha=0.5,
            linewidth=1.2,
            label="epsilon",
        )
        ax2.set_ylabel("epsilon (exploration rate)", color="#D55E00")
        ax2.tick_params(axis="y", labelcolor="#D55E00")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="lower right", fontsize=9)
    else:
        ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    path = results_path(filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bar_comparison(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    filename: str,
) -> str:
    """Simple bar chart helper (used for policy-agreement summaries).

    Parameters
    ----------
    labels : sequence of str
    values : sequence of float
    title : str
    ylabel : str
    filename : str

    Returns
    -------
    str
        Path of the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color="#0072B2", alpha=0.85)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = results_path(filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------
def print_results_table(results_dict: Mapping[str, Sequence], float_fmt: str = "{:.6f}") -> None:
    """Print a dictionary of equal-length columns as an aligned text table.

    Parameters
    ----------
    results_dict : mapping
        {column name: column values}. All columns must have the same length.
    float_fmt : str
        Format string applied to floats.

    Returns
    -------
    None
    """
    if not results_dict:
        print("(empty results)")
        return

    columns = list(results_dict.keys())
    lengths = {len(v) for v in results_dict.values()}
    if len(lengths) != 1:
        raise ValueError(f"columns have inconsistent lengths: { {k: len(v) for k, v in results_dict.items()} }")
    num_rows = lengths.pop()

    def fmt(x) -> str:
        if isinstance(x, float):
            return float_fmt.format(x)
        return str(x)

    cells = [[fmt(results_dict[c][r]) for c in columns] for r in range(num_rows)]
    widths = [
        max(len(str(columns[i])), *(len(row[i]) for row in cells)) if cells else len(columns[i])
        for i in range(len(columns))
    ]

    header = " | ".join(str(c).rjust(widths[i]) for i, c in enumerate(columns))
    print(header)
    print("-" * len(header))
    for row in cells:
        print(" | ".join(row[i].rjust(widths[i]) for i in range(len(columns))))


# ---------------------------------------------------------------------------
# Adaptive RL & Overhead Metrics
# ---------------------------------------------------------------------------
def compute_adaptation_speed(
    ler_timeline: Sequence[float],
    drift_step: int,
    target_ler: float,
    window: int = 100,
) -> int:
    """Compute adaptation speed: timesteps required after drift to reach target LER.

    Parameters
    ----------
    ler_timeline : sequence of float
        Per-step binary logical error indicators (1 = error, 0 = success).
    drift_step : int
        Timestep index where drift occurred.
    target_ler : float
        Target LER threshold to recover to.
    window : int
        Moving window for LER calculation.

    Returns
    -------
    int
        Number of steps to adaptation; len(ler_timeline) if target not reached.
    """
    ler_timeline = np.asarray(ler_timeline, dtype=float)
    if drift_step >= len(ler_timeline):
        return 0

    post_drift = ler_timeline[drift_step:]
    if len(post_drift) < window:
        return len(post_drift)

    for i in range(len(post_drift) - window):
        rolling_ler = np.mean(post_drift[i : i + window])
        if rolling_ler <= target_ler * 1.1:
            return i + window
    return len(post_drift)


def compute_recovery_after_drift(
    pre_drift_ler: float,
    peak_drift_ler: float,
    post_drift_ler: float,
) -> float:
    """Compute recovery fraction after drift: (peak_ler - post_ler) / (peak_ler - pre_ler).

    Returns
    -------
    float
        1.0 = 100% recovery to pre-drift baseline; 0.0 = no recovery.
    """
    denom = peak_drift_ler - pre_drift_ler
    if abs(denom) < 1e-8:
        return 1.0
    recovery = (peak_drift_ler - post_drift_ler) / denom
    return float(np.clip(recovery, 0.0, 1.0))


def plot_adaptive_drift_comparison(
    timesteps: Sequence[int],
    ler_dict: Mapping[str, Sequence[float]],
    updates_dict: Mapping[str, Sequence[float]],
    action_counts: Mapping[str, int],
    drift_events: Sequence[int],
    filename: str = config.ADAPTIVE_PLOT,
) -> str:
    """Plot multi-panel figure for adaptive RL evaluation.

    Panel 1: Moving LER over time for all strategies under dynamic drift.
    Panel 2: Cumulative online update overhead over time.
    Panel 3: Meta-controller decision breakdown (Selective Adaptation).

    Parameters
    ----------
    timesteps : sequence of int
    ler_dict : mapping
    updates_dict : mapping
    action_counts : mapping
    drift_events : sequence of int
    filename : str

    Returns
    -------
    str
        Path to saved PNG plot.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Panel 1: LER over time
    colors = {
        "Fixed MWPM": "#56B4E9",
        "Continuous RL Fine-Tune": "#D55E00",
        "Selective Adaptation (RL)": "#009E73",
        "Oracle MWPM": "#0072B2",
    }

    for label, ler_curve in ler_dict.items():
        c = colors.get(label, "#999999")
        ax1.plot(timesteps, ler_curve, label=label, color=c, linewidth=1.8)

    for event_t in drift_events:
        ax1.axvline(event_t, color="red", linestyle="--", alpha=0.6, label="Drift Event" if event_t == drift_events[0] else "")

    ax1.set_yscale("log")
    ax1.set_ylabel("Logical Error Rate (LER)")
    ax1.set_title("Dynamic Noise Drift: LER & Online Update Overhead Comparison")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Cumulative Online Updates
    for label, upd_curve in updates_dict.items():
        c = colors.get(label, "#999999")
        ax2.plot(timesteps, upd_curve, label=label, color=c, linewidth=1.8)

    for event_t in drift_events:
        ax2.axvline(event_t, color="red", linestyle="--", alpha=0.6)

    ax2.set_ylabel("Cumulative Online Updates")
    ax2.set_title("Online Updating Overhead (Lower is Better)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    # Panel 3: Meta-action breakdown bar chart / timeline summary
    act_labels = list(action_counts.keys())
    act_vals = list(action_counts.values())
    ax3.barh(act_labels, act_vals, color=["#0072B2", "#E69F00", "#D55E00"], alpha=0.85)
    ax3.set_xlabel("Occurrences / Count")
    ax3.set_title("Selective Meta-Controller Action Distribution")
    ax3.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    path = results_path(filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

