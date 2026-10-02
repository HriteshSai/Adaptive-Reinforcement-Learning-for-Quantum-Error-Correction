"""
experiments/exp_rl_train.py
===========================
EXPERIMENT 2 - train the tabular Q-learning decoder.

Questions this experiment answers
---------------------------------
1. Is the environment itself correct? (hardcoded-policy verification, rule 9)
2. Can a tabular agent that sees ONLY the syndrome and a +1/-1 logical reward
   discover the minimum-weight decoding rule on its own?
3. Is that discovery reliable, or seed-dependent? (config.NUM_SEEDS runs)
4. How does the trained agent compare with MWPM at every test noise rate,
   including rates it never trained on?

The agent is trained at a single noise rate (config.NOISE_RATE_TRAIN = 0.03).
Testing it elsewhere is the first taste of the drift study in experiment 3.
"""

from __future__ import annotations

import numpy as np

import config
from circuits.repetition_code import build_repetition_code_circuit
from decoders.mwpm_decoder import MWPMDecoder
from decoders.rl_decoder import RLDecoder
from environment.qec_env import QECDecoderEnv
from evaluation.metrics import (
    compute_ler_with_confidence,
    plot_ler_comparison,
    plot_training_curve,
    print_results_table,
    theoretical_ler,
)
from simulation.sampler import sample_syndromes
from training.qlearner import TabularQLearner
from utils.helpers import (
    banner,
    format_policy,
    results_path,
    save_json,
    save_to_csv,
    seeds_for_experiment,
    set_seed,
    setup_logger,
)


class EnvironmentVerificationError(RuntimeError):
    """Raised when the environment self-test fails (rule 9: stop immediately)."""


def verify_environment(env: QECDecoderEnv, verbose: bool = True) -> float:
    """Run the hardcoded-optimal-policy check and abort if it fails.

    Why: if the reward function, the syndrome encoding or the circuit is
    wrong, a *known-good* decoder will score badly in this environment. That
    is a far faster and far more reliable diagnosis than staring at a
    non-converging learning curve.

    Parameters
    ----------
    env : QECDecoderEnv
    verbose : bool

    Returns
    -------
    float
        The measured success rate of the hardcoded policy.

    Raises
    ------
    EnvironmentVerificationError
        If the success rate is below config.VERIFY_MIN_SUCCESS_RATE.
    """
    success_rate = env.verify_with_optimal_policy(verbose=verbose)

    if success_rate < config.VERIFY_MIN_SUCCESS_RATE:
        message = (
            "\n"
            + "!" * 72
            + "\nENVIRONMENT VERIFICATION FAILED - TRAINING ABORTED\n"
            + "!" * 72
            + f"\n  hardcoded optimal policy success rate : {success_rate:.4%}"
            f"\n  required threshold                    : {config.VERIFY_MIN_SUCCESS_RATE:.2%}"
            f"\n  analytic expectation at p={env.noise_rate}          : "
            f"{env.analytic_optimal_success_rate():.4%}"
            "\n\nThe environment, not the agent, is broken. Check, in this order:"
            "\n  1. environment/qec_env.py  : is the reward computed from the"
            "\n     residual error r = e XOR c, with +1 only when r is all-zero?"
            "\n  2. environment/qec_env.py  : state = 2*s0 + s1 must agree with"
            "\n     config.OPTIMAL_POLICY = (0, 3, 1, 2)."
            "\n  3. circuits/repetition_code.py : detector order must be"
            "\n     (parity q0q1, parity q1q2), and X_ERROR must hit data qubits only."
            "\n  4. config.NOISE_RATE_TRAIN : a very large p makes even the optimal"
            "\n     policy fail often; the threshold assumes p is small.\n"
        )
        print(message)
        raise EnvironmentVerificationError(
            f"environment verification failed: {success_rate:.4%} < "
            f"{config.VERIFY_MIN_SUCCESS_RATE:.2%}"
        )

    return success_rate


def run_rl_training_experiment(
    noise_rate: float | None = None,
    num_episodes: int | None = None,
    num_seeds: int | None = None,
    seed: int = config.SEED,
    verbose: bool = True,
) -> dict:
    """Verify the environment, train the agent, and benchmark it against MWPM.

    Steps
    -----
    1. Build QECDecoderEnv at the training noise rate.
    2. RULE 9: run verify_with_optimal_policy() and stop on failure.
    3. Train `num_seeds` independent agents (the first one is the "primary"
       agent whose Q-table is saved and reused by experiment 3).
    4. Print the primary Q-table and compare its greedy policy with the
       analytic minimum-weight policy.
    5. Plot the training curve.
    6. Evaluate the primary agent and MWPM on fresh Stim samples at every
       config.NOISE_RATES_TEST rate and plot both LER curves.

    Parameters
    ----------
    noise_rate : float
        Training noise rate p.
    num_episodes : int
        Episodes per training run.
    num_seeds : int
        Independent repetitions, to show the result is not seed luck.
    seed : int
        Base seed.
    verbose : bool

    Returns
    -------
    dict
        Verification result, per-seed policies, the primary Q-table path, the
        evaluation table and the paths of every artefact written.
    """
    logger = setup_logger("exp_rl_train")
    set_seed(seed)

    # Resolved here, not in the signature, so that runtime overrides of the
    # config (e.g. main.py --quick) are actually honoured.
    noise_rate = float(config.NOISE_RATE_TRAIN if noise_rate is None else noise_rate)
    num_episodes = int(config.NUM_EPISODES if num_episodes is None else num_episodes)
    num_seeds = int(config.NUM_SEEDS if num_seeds is None else num_seeds)

    if verbose:
        print(banner("EXPERIMENT 2: TRAIN THE TABULAR Q-LEARNING DECODER"))

    # ---------------------------------------------------------------
    # 1 + 2. Environment and its self-test
    # ---------------------------------------------------------------
    env = QECDecoderEnv(noise_rate=noise_rate, seed=seed)
    verification_rate = verify_environment(env, verbose=verbose)
    logger.info("environment verification passed: %.4f", verification_rate)

    # ---------------------------------------------------------------
    # 3. Multi-seed training
    # ---------------------------------------------------------------
    seeds = seeds_for_experiment(num_seeds, seed)
    per_seed = []
    primary_learner = None
    primary_history: list[float] = []

    for run_idx, run_seed in enumerate(seeds):
        run_env = QECDecoderEnv(noise_rate=noise_rate, seed=run_seed)
        learner = TabularQLearner(run_env, config, seed=run_seed)
        _, history = learner.train(num_episodes, verbose=verbose and run_idx == 0)

        greedy_success = learner.evaluate()
        report = learner.convergence_report()

        per_seed.append(
            {
                "seed": run_seed,
                "policy": learner.get_policy(),
                "policy_matches_optimal": report["policy_matches"],
                "greedy_success_rate": greedy_success,
                "max_q_error": report["max_abs_error_optimal_actions"],
                "mean_reward_last_1k": learner.diagnostics["mean_reward_last_1k"],
            }
        )
        logger.info(
            "seed %d: policy %s (optimal: %s), greedy success %.4f",
            run_seed,
            learner.get_policy(),
            report["policy_matches"],
            greedy_success,
        )

        if run_idx == 0:
            primary_learner = learner
            primary_history = history

    assert primary_learner is not None

    # ---------------------------------------------------------------
    # 4. Inspect the primary agent
    # ---------------------------------------------------------------
    if verbose:
        primary_learner.print_q_table()

        print("multi-seed summary (does the result depend on luck?)")
        print_results_table(
            {
                "seed": [r["seed"] for r in per_seed],
                "learned policy": ["".join(str(a) for a in r["policy"]) for r in per_seed],
                "= optimal": ["yes" if r["policy_matches_optimal"] else "NO" for r in per_seed],
                "greedy success": [r["greedy_success_rate"] for r in per_seed],
                "max |dQ|": [r["max_q_error"] for r in per_seed],
            }
        )
        agree = sum(r["policy_matches_optimal"] for r in per_seed)
        print(
            f"\n{agree}/{len(per_seed)} seeds recovered the exact minimum-weight policy."
        )
        print(f"optimal policy: {format_policy(config.OPTIMAL_POLICY)}")
        print(f"learned policy: {format_policy(primary_learner.get_policy())}\n")

    q_path = results_path(config.QTABLE_FILE)
    primary_learner.save_q_table(q_path)
    logger.info("saved Q-table -> %s", q_path)

    # ---------------------------------------------------------------
    # 5. Training curve
    # ---------------------------------------------------------------
    optimal_reward = 2 * env.analytic_optimal_success_rate() - 1
    curve_path = plot_training_curve(
        primary_history,
        filename=config.TRAINING_CURVE_PLOT,
        optimal_reward=optimal_reward,
        epsilon_history=primary_learner.epsilon_history,
        title=f"Tabular Q-learning at p = {noise_rate} (seed {seeds[0]})",
    )

    # ---------------------------------------------------------------
    # 6. RL vs MWPM across the test noise rates
    # ---------------------------------------------------------------
    rl_decoder = RLDecoder(primary_learner.q_table, name=f"RL (trained at p={noise_rate})")

    rates = list(config.NOISE_RATES_TEST)
    rl_lers, rl_lo, rl_hi = [], [], []
    mwpm_lers, mwpm_lo, mwpm_hi = [], [], []

    for i, p in enumerate(rates):
        rate_seed = seed + 211 * (i + 1)
        circuit = build_repetition_code_circuit(p)
        syndromes, observables = sample_syndromes(circuit, config.NUM_SHOTS, rate_seed)

        rl_pred = rl_decoder.decode_batch(syndromes)
        mwpm = MWPMDecoder(circuit, name=f"MWPM (oracle p={p})")
        mwpm_pred = mwpm.decode(syndromes)

        a, b, c = compute_ler_with_confidence(rl_pred, observables, seed=rate_seed)
        rl_lers.append(a); rl_lo.append(b); rl_hi.append(c)

        a, b, c = compute_ler_with_confidence(mwpm_pred, observables, seed=rate_seed)
        mwpm_lers.append(a); mwpm_lo.append(b); mwpm_hi.append(c)

        # Both decoders see the SAME shots here, so any difference is purely
        # a difference in decision rule, not Monte-Carlo noise.
        agreement = float(np.mean(rl_pred.reshape(-1) == mwpm_pred.reshape(-1)))
        logger.info(
            "p=%.3f  RL LER=%.6f  MWPM LER=%.6f  agreement=%.4f",
            p, rl_lers[-1], mwpm_lers[-1], agreement,
        )

    comparison = {
        "noise_rate": rates,
        "ler_rl": rl_lers,
        "ler_rl_lower_95": rl_lo,
        "ler_rl_upper_95": rl_hi,
        "ler_mwpm_oracle": mwpm_lers,
        "ler_mwpm_lower_95": mwpm_lo,
        "ler_mwpm_upper_95": mwpm_hi,
        "theory": [theoretical_ler(p) for p in rates],
    }
    csv_path = save_to_csv(comparison, config.RL_VS_MWPM_CSV)
    plot_path = plot_ler_comparison(
        rates,
        {
            f"RL (trained at p={noise_rate})": rl_lers,
            "MWPM (oracle-calibrated)": mwpm_lers,
        },
        title="RL decoder vs MWPM across physical error rates",
        filename=config.RL_VS_MWPM_PLOT,
        ci_dict={
            f"RL (trained at p={noise_rate})": (rl_lo, rl_hi),
            "MWPM (oracle-calibrated)": (mwpm_lo, mwpm_hi),
        },
    )

    metadata = {
        "training_noise_rate": noise_rate,
        "num_episodes": num_episodes,
        "verification_success_rate": verification_rate,
        "seeds": seeds,
        "per_seed": [
            {**r, "policy": list(r["policy"])} for r in per_seed
        ],
        "primary_policy": list(primary_learner.get_policy()),
        "optimal_policy": list(config.OPTIMAL_POLICY),
        "q_table": primary_learner.q_table.tolist(),
    }
    json_path = save_json(metadata, "rl_training_metadata.json")

    if verbose:
        print("RL vs MWPM (same shots at every point)")
        print_results_table(
            {
                "p": rates,
                "RL LER": rl_lers,
                "MWPM LER": mwpm_lers,
                "difference": [r - m for r, m in zip(rl_lers, mwpm_lers)],
                "theory": comparison["theory"],
            }
        )
        print(f"\nsaved Q-table       -> {q_path}")
        print(f"saved training curve-> {curve_path}")
        print(f"saved comparison    -> {csv_path}")
        print(f"saved plot          -> {plot_path}")
        print(f"saved metadata      -> {json_path}")
        print(
            "\nReading the result: the RL curve should lie on top of the MWPM curve.\n"
            "That is the expected outcome, not a disappointment - the agent was\n"
            "given no physics, only syndromes and a +1/-1 signal, and it still\n"
            "recovered the same 4-entry lookup table that minimum-weight matching\n"
            "derives from the error model. Matching a near-optimal baseline is the\n"
            "strongest result available on a code this small."
        )

    return {
        "verification_rate": verification_rate,
        "per_seed": per_seed,
        "q_table": primary_learner.q_table,
        "q_table_path": q_path,
        "history": primary_history,
        "training_curve": curve_path,
        "noise_rates": rates,
        "ler_rl": rl_lers,
        "ler_mwpm": mwpm_lers,
        "csv": csv_path,
        "plot": plot_path,
        "metadata": json_path,
    }


if __name__ == "__main__":  # pragma: no cover
    run_rl_training_experiment()
