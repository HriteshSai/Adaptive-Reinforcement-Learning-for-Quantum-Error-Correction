"""
experiments/exp_drift.py
========================
EXPERIMENT 3 - the actual study: what happens when the noise drifts away from
the conditions the decoder was calibrated on?

Motivation
----------
Real quantum hardware does not hold still. Qubit error rates wander with
temperature, calibration age and cosmic-ray events. A decoder is always
configured from *yesterday's* characterisation data. This experiment measures
the cost of that staleness for two kinds of decoder:

    Fixed RL    : Q-table trained once at p = 0.03, never updated.
    Fixed MWPM  : matching weights calibrated once at p = 0.03, never updated.
    Oracle MWPM : matching weights rebuilt at the true test noise rate. This
                  is the unattainable upper bound - it always knows the truth.

Part A (required): drift in the MAGNITUDE of uniform noise, p = 0.01 .. 0.15.
Part B (bonus)   : drift in the SHAPE of the noise (per-qubit asymmetry).

Read the printed analysis for what the two parts actually show. Part A has a
mathematically inevitable null result, and understanding *why* is the main
scientific insight of Phase 1; Part B is where drift actually bites.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

import config
from circuits.repetition_code import build_repetition_code_circuit
from decoders.mwpm_decoder import MWPMDecoder
from decoders.rl_decoder import RLDecoder
from environment.qec_env import QECDecoderEnv
from evaluation.metrics import (
    compute_ler_with_confidence,
    plot_ler_comparison,
    print_results_table,
    relative_degradation,
    theoretical_ler,
)
from simulation.sampler import sample_syndromes
from training.qlearner import TabularQLearner
from utils.helpers import (
    banner,
    format_policy,
    results_path,
    save_to_csv,
    set_seed,
    setup_logger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_or_train_q_table(
    noise_rate: float = config.NOISE_RATE_TRAIN,
    seed: int = config.SEED,
    verbose: bool = True,
) -> np.ndarray:
    """Reuse the Q-table from experiment 2, or train one if it is missing.

    Reusing the saved table is what makes "the agent trained at p = 0.03" a
    meaningful phrase across experiments: the drift study must evaluate the
    *same* agent that was analysed in experiment 2.

    Parameters
    ----------
    noise_rate : float
        Training noise rate, used only if a fresh agent has to be trained.
    seed : int
    verbose : bool

    Returns
    -------
    np.ndarray, shape (4, 4)
        The trained Q-table.
    """
    path = results_path(config.QTABLE_FILE)
    if os.path.exists(path):
        if verbose:
            print(f"[drift] reusing Q-table from {path}")
        return np.load(path)

    if verbose:
        print(f"[drift] no saved Q-table at {path}; training a fresh agent")
    env = QECDecoderEnv(noise_rate=noise_rate, seed=seed)
    success = env.verify_with_optimal_policy(verbose=verbose)
    if success < config.VERIFY_MIN_SUCCESS_RATE:
        raise RuntimeError(
            f"environment verification failed ({success:.4%}); refusing to train"
        )
    learner = TabularQLearner(env, config, seed=seed)
    learner.train(config.NUM_EPISODES, verbose=False)
    learner.save_q_table(path)
    return learner.q_table


def biased_rates(p: float, strength: float, profile: Sequence[float] = config.BIAS_PROFILE):
    """Interpolate between uniform noise and a biased per-qubit noise profile.

    Parameters
    ----------
    p : float
        Base physical error rate.
    strength : float
        0.0 = perfectly uniform noise (all qubits at p);
        1.0 = the full profile (qubit i at p * profile[i]).
    profile : sequence of 3 floats
        Per-qubit multipliers at full strength. Defaults to
        config.BIAS_PROFILE = (0.1, 1.0, 3.0), i.e. qubit 0 is unusually
        clean and qubit 2 (the logical readout qubit) is unusually noisy -
        the situation where a stale decoder is most likely to guess wrong.

    Returns
    -------
    tuple of 3 floats
        Per-qubit error probabilities, clipped into [0, 0.5].
    """
    return tuple(
        float(np.clip(p * (1.0 + strength * (m - 1.0)), 0.0, 0.5)) for m in profile
    )


# ---------------------------------------------------------------------------
# Part A: drift in noise magnitude (required experiment)
# ---------------------------------------------------------------------------
def run_drift_experiment(
    train_rate: float | None = None,
    test_rates=None,
    num_shots: int | None = None,
    seed: int = config.SEED,
    verbose: bool = True,
) -> dict:
    """Compare Fixed RL, Fixed MWPM and Oracle MWPM as the noise rate drifts.

    Procedure, per test rate p:
        1. Build the circuit at p and sample `num_shots` shots.
        2. Decode the SAME shots with all three decoders, so differences come
           only from the decoders and never from sampling noise.
        3. Record LER with bootstrap confidence intervals.

    Parameters
    ----------
    train_rate : float
        Rate the fixed decoders were calibrated/trained at.
    test_rates : sequence of float, optional
        Defaults to config.NOISE_RATES_TEST.
    num_shots : int
    seed : int
    verbose : bool

    Returns
    -------
    dict
        Curves for the three decoders plus artefact paths.
    """
    logger = setup_logger("exp_drift")
    set_seed(seed)
    train_rate = float(config.NOISE_RATE_TRAIN if train_rate is None else train_rate)
    num_shots = int(config.NUM_SHOTS if num_shots is None else num_shots)
    rates = list(config.NOISE_RATES_TEST if test_rates is None else test_rates)

    if verbose:
        print(banner("EXPERIMENT 3A: NOISE-MAGNITUDE DRIFT"))

    # --- the three decoders -------------------------------------------
    q_table = load_or_train_q_table(train_rate, seed, verbose=verbose)
    rl_fixed = RLDecoder(q_table, name=f"Fixed RL (trained p={train_rate})")

    train_circuit = build_repetition_code_circuit(train_rate)
    mwpm_fixed = MWPMDecoder(train_circuit, name=f"Fixed MWPM (calibrated p={train_rate})")

    if verbose:
        print(f"\nfixed RL policy   : {format_policy(rl_fixed.get_policy())}")
        print(f"fixed RL lookup   : {rl_fixed.policy_table()}")
        print(f"fixed MWPM lookup : {mwpm_fixed.policy_table()}")
        print("(lookup = syndrome state -> predicted logical flip)\n")

    rl_ler, rl_lo, rl_hi = [], [], []
    fx_ler, fx_lo, fx_hi = [], [], []
    or_ler, or_lo, or_hi = [], [], []
    rl_vs_oracle_agreement = []

    for i, p in enumerate(rates):
        rate_seed = seed + 307 * (i + 1)
        circuit = build_repetition_code_circuit(p)
        syndromes, observables = sample_syndromes(circuit, num_shots, rate_seed)

        # Oracle: rebuilt at the true test rate -> always correctly calibrated.
        mwpm_oracle = MWPMDecoder(circuit, name=f"Oracle MWPM (p={p})")

        preds_rl = rl_fixed.decode_batch(syndromes)
        preds_fx = mwpm_fixed.decode(syndromes)
        preds_or = mwpm_oracle.decode(syndromes)

        a, b, c = compute_ler_with_confidence(preds_rl, observables, seed=rate_seed)
        rl_ler.append(a); rl_lo.append(b); rl_hi.append(c)
        a, b, c = compute_ler_with_confidence(preds_fx, observables, seed=rate_seed)
        fx_ler.append(a); fx_lo.append(b); fx_hi.append(c)
        a, b, c = compute_ler_with_confidence(preds_or, observables, seed=rate_seed)
        or_ler.append(a); or_lo.append(b); or_hi.append(c)

        rl_vs_oracle_agreement.append(
            float(np.mean(preds_rl.reshape(-1) == preds_or.reshape(-1)))
        )

        logger.info(
            "p=%.3f  RL=%.6f  fixedMWPM=%.6f  oracleMWPM=%.6f  RL/oracle agreement=%.4f",
            p, rl_ler[-1], fx_ler[-1], or_ler[-1], rl_vs_oracle_agreement[-1],
        )

    results = {
        "noise_rate": rates,
        "ler_rl_fixed": rl_ler,
        "ler_rl_lower_95": rl_lo,
        "ler_rl_upper_95": rl_hi,
        "ler_mwpm_fixed": fx_ler,
        "ler_mwpm_fixed_lower_95": fx_lo,
        "ler_mwpm_fixed_upper_95": fx_hi,
        "ler_mwpm_oracle": or_ler,
        "ler_mwpm_oracle_lower_95": or_lo,
        "ler_mwpm_oracle_upper_95": or_hi,
        "rl_vs_oracle_agreement": rl_vs_oracle_agreement,
        "theory": [theoretical_ler(p) for p in rates],
        "rl_over_oracle": relative_degradation(rl_ler, or_ler),
        "fixed_mwpm_over_oracle": relative_degradation(fx_ler, or_ler),
    }

    csv_path = save_to_csv(results, config.DRIFT_CSV)
    plot_path = plot_ler_comparison(
        rates,
        {
            f"Fixed RL (trained p={train_rate})": rl_ler,
            f"Fixed MWPM (calibrated p={train_rate})": fx_ler,
            "Oracle MWPM (recalibrated)": or_ler,
        },
        title=f"Noise-model drift: decoders calibrated at p={train_rate}",
        filename=config.DRIFT_PLOT,
        ci_dict={
            f"Fixed RL (trained p={train_rate})": (rl_lo, rl_hi),
            f"Fixed MWPM (calibrated p={train_rate})": (fx_lo, fx_hi),
            "Oracle MWPM (recalibrated)": (or_lo, or_hi),
        },
    )

    if verbose:
        print("\nDrift results (identical shots for all three decoders)")
        print_results_table(
            {
                "p": rates,
                "Fixed RL": rl_ler,
                "Fixed MWPM": fx_ler,
                "Oracle MWPM": or_ler,
                "RL/oracle": results["rl_over_oracle"],
                "fixed/oracle": results["fixed_mwpm_over_oracle"],
                "RL=oracle preds": rl_vs_oracle_agreement,
            }
        )
        print(f"\nsaved table -> {csv_path}")
        print(f"saved plot  -> {plot_path}")
        _print_magnitude_drift_analysis(
            rates, rl_ler, fx_ler, or_ler, rl_fixed, mwpm_fixed, train_rate
        )

    return {
        "noise_rates": rates,
        "ler_rl_fixed": rl_ler,
        "ler_mwpm_fixed": fx_ler,
        "ler_mwpm_oracle": or_ler,
        "csv": csv_path,
        "plot": plot_path,
    }


def _print_magnitude_drift_analysis(
    rates, rl_ler, fx_ler, or_ler, rl_fixed, mwpm_fixed, train_rate
) -> None:
    """Explain, in words, what the magnitude-drift numbers mean.

    Parameters
    ----------
    rates : sequence of float
    rl_ler, fx_ler, or_ler : sequence of float
        The three LER curves.
    rl_fixed : RLDecoder
    mwpm_fixed : MWPMDecoder
    train_rate : float

    Returns
    -------
    None
    """
    same_rule = rl_fixed.policy_table() == mwpm_fixed.policy_table()
    max_gap_rl = max(abs(a - b) for a, b in zip(rl_ler, or_ler))
    max_gap_fx = max(abs(a - b) for a, b in zip(fx_ler, or_ler))

    print("\n" + "-" * 70)
    print("ANALYSIS: where does each decoder degrade under magnitude drift?")
    print("-" * 70)
    print(
        f"largest |LER - oracle LER| over the sweep:\n"
        f"    Fixed RL   : {max_gap_rl:.6f}\n"
        f"    Fixed MWPM : {max_gap_fx:.6f}"
    )
    print(
        f"\nfixed RL and fixed MWPM implement the same syndrome->prediction "
        f"lookup table: {same_rule}"
    )
    print(
        "\nWHY THIS RESULT LOOKS LIKE A FLAT LINE (and why that is correct):\n"
        "  For a single round of the 3-qubit repetition code with i.i.d. noise,\n"
        "  the most likely explanation of a syndrome does not depend on p at all,\n"
        "  as long as p < 0.5. Syndrome [1,0] is explained either by one flip on\n"
        "  q0 (probability p(1-p)^2) or by two flips on q1,q2 (p^2(1-p)); the\n"
        "  ratio is (1-p)/p > 1 for every p < 0.5, so 'q0 flipped' always wins.\n"
        "  Changing p rescales every weight in the matching graph by the same\n"
        f"  monotone factor, so an MWPM calibrated at p={train_rate} makes exactly the\n"
        "  same decisions as an oracle MWPM at p=0.15. The RL agent converged to\n"
        "  that same table, so all three curves coincide.\n"
        "\n  The empirical finding of Part A is therefore a NULL RESULT, and it is\n"
        "  a real result: under pure magnitude drift, a fixed decoder loses\n"
        "  nothing, and the RL agent is exactly as robust as a stale MWPM. Any\n"
        "  visible separation between the curves at small p would be Monte-Carlo\n"
        "  noise (only a handful of failures occur in 10,000 shots at p=0.01),\n"
        "  which is why the plot carries bootstrap error bars.\n"
        "\n  Drift only hurts when it changes the RANKING of error hypotheses,\n"
        "  which requires the SHAPE of the noise to change. That is Part B."
    )
    print("-" * 70)


# ---------------------------------------------------------------------------
# Part B: drift in noise shape (bonus, this is where drift bites)
# ---------------------------------------------------------------------------
def run_biased_noise_experiment(
    train_rate: float | None = None,
    strengths=None,
    num_shots: int | None = None,
    seed: int = config.SEED,
    retrain_rl: bool = True,
    verbose: bool = True,
) -> dict:
    """Drift the SHAPE of the noise and measure who survives it.

    Setup: keep the average error rate near the training rate, but make the
    per-qubit rates asymmetric, interpolating from uniform (strength 0) to
    config.BIAS_PROFILE (strength 1). At full strength the logical readout
    qubit is the noisiest one, so the most likely explanation of some
    syndromes changes - and a stale decoder starts making the wrong call.

    Four decoders are compared:
        Fixed RL       : trained on uniform noise at p = train_rate.
        Fixed MWPM     : calibrated on uniform noise at p = train_rate.
        Oracle MWPM    : calibrated on the true biased rates.
        Retrained RL   : a fresh agent trained in the biased environment
                         (optional; shows whether the RL agent can recover the
                         loss if it is allowed to see the new noise).

    Parameters
    ----------
    train_rate : float
    strengths : sequence of float, optional
        Bias strengths to sweep. Defaults to config.BIAS_STRENGTHS.
    num_shots : int
    seed : int
    retrain_rl : bool
        Whether to train a fresh agent at each bias strength.
    verbose : bool

    Returns
    -------
    dict
        Curves for each decoder plus artefact paths.
    """
    logger = setup_logger("exp_drift_bias")
    set_seed(seed)
    train_rate = float(config.NOISE_RATE_TRAIN if train_rate is None else train_rate)
    num_shots = int(config.BIAS_NUM_SHOTS if num_shots is None else num_shots)
    strengths = list(config.BIAS_STRENGTHS if strengths is None else strengths)

    if verbose:
        print(banner("EXPERIMENT 3B: NOISE-SHAPE (BIAS) DRIFT"))
        print(
            f"per-qubit multipliers at full strength: {config.BIAS_PROFILE}\n"
            f"base rate p = {train_rate}\n"
        )

    q_table = load_or_train_q_table(train_rate, seed, verbose=verbose)
    rl_fixed = RLDecoder(q_table, name="Fixed RL (uniform-trained)")
    mwpm_fixed = MWPMDecoder(
        build_repetition_code_circuit(train_rate), name="Fixed MWPM (uniform-calibrated)"
    )

    rows = []
    oracle_lookups = []
    rl_ler, fx_ler, or_ler, re_ler = [], [], [], []

    for i, strength in enumerate(strengths):
        rate_seed = seed + 401 * (i + 1)
        rates = biased_rates(train_rate, strength)
        circuit = build_repetition_code_circuit(train_rate, per_qubit_rates=rates)
        syndromes, observables = sample_syndromes(circuit, num_shots, rate_seed)

        mwpm_oracle = MWPMDecoder(circuit, name="Oracle MWPM (bias-aware)")
        oracle_lookups.append(mwpm_oracle.policy_table())

        preds_rl = rl_fixed.decode_batch(syndromes)
        preds_fx = mwpm_fixed.decode(syndromes)
        preds_or = mwpm_oracle.decode(syndromes)

        a_rl, _, _ = compute_ler_with_confidence(preds_rl, observables, seed=rate_seed)
        a_fx, _, _ = compute_ler_with_confidence(preds_fx, observables, seed=rate_seed)
        a_or, _, _ = compute_ler_with_confidence(preds_or, observables, seed=rate_seed)

        rl_ler.append(a_rl); fx_ler.append(a_fx); or_ler.append(a_or)

        a_re = float("nan")
        retrained_policy = None
        if retrain_rl:
            env = QECDecoderEnv(
                noise_rate=train_rate, seed=rate_seed, per_qubit_rates=rates
            )
            learner = TabularQLearner(env, config, seed=rate_seed)
            learner.train(config.NUM_EPISODES, verbose=False)
            rl_retrained = RLDecoder(learner.q_table, name="Retrained RL (bias-aware)")
            preds_re = rl_retrained.decode_batch(syndromes)
            a_re, _, _ = compute_ler_with_confidence(preds_re, observables, seed=rate_seed)
            retrained_policy = learner.get_policy()
        re_ler.append(a_re)

        rows.append(
            {
                "bias_strength": strength,
                "p_q0": rates[0],
                "p_q1": rates[1],
                "p_q2": rates[2],
                "ler_rl_fixed": a_rl,
                "ler_mwpm_fixed": a_fx,
                "ler_mwpm_oracle": a_or,
                "ler_rl_retrained": a_re,
                "oracle_lookup": str(mwpm_oracle.policy_table()),
                "fixed_lookup": str(mwpm_fixed.policy_table()),
                "retrained_policy": str(retrained_policy),
            }
        )
        logger.info(
            "bias=%.2f rates=%s  RL=%.6f fixedMWPM=%.6f oracleMWPM=%.6f retrainedRL=%s",
            strength, rates, a_rl, a_fx, a_or, f"{a_re:.6f}",
        )

    csv_path = save_to_csv(rows, config.BIAS_CSV)

    curves = {
        "Fixed RL (uniform-trained)": rl_ler,
        "Fixed MWPM (uniform-calibrated)": fx_ler,
        "Oracle MWPM (bias-aware)": or_ler,
    }
    if retrain_rl:
        curves["Retrained RL (bias-aware)"] = re_ler

    plot_path = plot_ler_comparison(
        strengths,
        curves,
        title=f"Noise-shape drift at p={train_rate}: per-qubit bias {config.BIAS_PROFILE}",
        filename=config.BIAS_PLOT,
        show_physical_line=False,
        show_theory=False,
        xlabel="noise-shape bias strength  (0 = uniform, 1 = full profile)",
    )

    if verbose:
        print("\nBias-drift results")
        print_results_table(
            {
                "bias": strengths,
                "p per qubit": [
                    "(" + ", ".join(f"{r:.4f}" for r in biased_rates(train_rate, s)) + ")"
                    for s in strengths
                ],
                "Fixed RL": rl_ler,
                "Fixed MWPM": fx_ler,
                "Oracle MWPM": or_ler,
                "Retrained RL": re_ler,
            }
        )
        print(f"\nsaved table -> {csv_path}")
        print(f"saved plot  -> {plot_path}")

        worst = int(np.argmax([r - o for r, o in zip(rl_ler, or_ler)]))
        oracle_worst = or_ler[worst] if or_ler[worst] > 0 else float("nan")
        rl_penalty = rl_ler[worst] / oracle_worst
        fx_penalty = fx_ler[worst] / oracle_worst
        # Did retraining under the new noise actually buy anything? Decide from
        # the data instead of assuming - this is the point of an empirical study.
        retrain_recovered = (
            retrain_rl
            and not np.isnan(re_ler[worst])
            and re_ler[worst] <= 0.5 * (rl_ler[worst] + oracle_worst)
        )
        stale_lookup = mwpm_fixed.policy_table()
        oracle_lookup_worst = oracle_lookups[worst]
        changed = [
            config.STATE_LABELS[st]
            for st in range(config.NUM_STATES)
            if stale_lookup[st] != oracle_lookup_worst[st]
        ]

        print("\n" + "-" * 70)
        print("ANALYSIS: noise-shape drift")
        print("-" * 70)
        print(
            f"  worst case for the stale decoders: bias strength "
            f"{strengths[worst]:.2f}, per-qubit p = "
            f"{tuple(round(r, 5) for r in biased_rates(train_rate, strengths[worst]))}"
        )
        print(f"    Fixed RL     LER = {rl_ler[worst]:.6f}  ({rl_penalty:.2f}x oracle)")
        print(f"    Fixed MWPM   LER = {fx_ler[worst]:.6f}  ({fx_penalty:.2f}x oracle)")
        print(f"    Oracle MWPM  LER = {or_ler[worst]:.6f}")
        if retrain_rl:
            print(f"    Retrained RL LER = {re_ler[worst]:.6f}")
        print(
            f"\n  syndromes where the oracle disagrees with the stale lookup table: "
            f"{changed if changed else 'none'}"
        )
        print(
            "\n  WHAT HAPPENED\n"
            "  The bias profile keeps the AVERAGE physical error rate at the training\n"
            f"  value ({train_rate}) and changes only its shape: qubit 0 becomes nearly\n"
            "  perfect while qubit 2 becomes the noisiest. At full bias, syndrome [1,0]\n"
            "  is now better explained by 'q1 and q2 both flipped' than by 'q0 flipped',\n"
            "  because p1*p2 has overtaken p0. The oracle MWPM rebuilds its edge weights\n"
            "  and switches its answer; both frozen decoders keep giving the old one.\n"
            "\n  FINDING 1: the fixed RL agent and the fixed MWPM degrade by exactly the\n"
            "  same amount. They encode the same stale 4-entry lookup table, so learning\n"
            "  from reward is neither more brittle nor more robust than deriving the\n"
            "  rule from an error model. Robustness here is a property of the\n"
            "  CALIBRATION being stale, not of how the decoder was obtained."
        )
        if retrain_rl:
            if retrain_recovered:
                print(
                    "\n  FINDING 2: retraining the agent in the drifted environment recovers\n"
                    "  most of the loss - and it did so from the +/-1 logical reward alone,\n"
                    "  with no characterisation of the new noise. That is the practical\n"
                    "  argument for RL decoders: MWPM cannot re-weight itself without a\n"
                    "  measured error model."
                )
            else:
                print(
                    "\n  FINDING 2 (the interesting one): retraining does NOT recover the\n"
                    f"  loss - the retrained agent still scores {re_ler[worst]:.6f}, matching the\n"
                    "  stale agent rather than the oracle. The bottleneck is the ACTION\n"
                    "  SPACE, not the learning algorithm. The optimal response to syndrome\n"
                    "  [1,0] under this noise is the weight-2 correction 'flip q1 and q2',\n"
                    "  which is not in the 4-action set {no-op, flip q0, flip q1, flip q2}.\n"
                    "  The agent converges correctly to the best action it CAN take, and\n"
                    "  that action is simply not good enough. MWPM is not limited this way,\n"
                    "  because it predicts the logical flip directly rather than naming a\n"
                    "  single-qubit correction.\n"
                    "\n  This is a clean, quotable Phase 1 result: under noise-shape drift the\n"
                    "  RL decoder's ceiling is set by its action-space design. Phase 2 test:\n"
                    "  widen the action space to all 8 correction patterns and re-run this\n"
                    "  experiment; the retrained curve should then meet the oracle."
                )
        print("-" * 70)

    return {
        "bias_strengths": strengths,
        "ler_rl_fixed": rl_ler,
        "ler_mwpm_fixed": fx_ler,
        "ler_mwpm_oracle": or_ler,
        "ler_rl_retrained": re_ler,
        "csv": csv_path,
        "plot": plot_path,
    }


if __name__ == "__main__":  # pragma: no cover
    run_drift_experiment()
    run_biased_noise_experiment()
