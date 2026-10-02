"""
experiments/exp_baseline.py
===========================
EXPERIMENT 1 - the MWPM baseline.

Question this experiment answers
--------------------------------
"How well does the standard classical decoder protect one logical bit, as the
physical error rate p sweeps from 0.01 to 0.15?"

Everything later in the project is measured against this curve, so it is run
first. It also acts as a sanity check on the whole simulation stack: the
measured LER must track the analytic prediction 3p^2 - 2p^3 for a
minimum-weight decoder on a distance-3 repetition code. If it does not, the
circuit or the sampler is wrong, and nothing downstream can be trusted.

Quantum terms recap
-------------------
physical error rate p : chance that one data qubit suffers a bit flip.
logical error rate    : chance that the *encoded* bit ends up wrong anyway.
MWPM                  : minimum-weight perfect matching, see decoders/.
"""

from __future__ import annotations

import numpy as np

import config
from circuits.repetition_code import build_repetition_code_circuit
from decoders.mwpm_decoder import MWPMDecoder
from evaluation.metrics import (
    compute_ler_with_confidence,
    plot_ler_comparison,
    print_results_table,
    theoretical_ler,
)
from simulation.sampler import print_sample_examples, sample_syndromes
from utils.helpers import banner, save_to_csv, set_seed, setup_logger


def run_baseline_experiment(
    noise_rates=None,
    num_shots: int | None = None,
    seed: int = config.SEED,
    verbose: bool = True,
) -> dict:
    """Measure the MWPM logical error rate across the test noise rates.

    Procedure, per noise rate p:
        1. Build the repetition-code circuit at p.
        2. Sample `num_shots` shots -> syndromes + true logical observables.
        3. Decode every syndrome with MWPM calibrated at that same p.
        4. LER = fraction of shots where the prediction was wrong, with a
           bootstrap 95% confidence interval.

    Parameters
    ----------
    noise_rates : sequence of float, optional
        Defaults to config.NOISE_RATES_TEST.
    num_shots : int
        Shots per noise rate. Defaults to config.NUM_SHOTS.
    seed : int
        Base seed; each noise rate gets a distinct derived seed so the
        different points are statistically independent yet reproducible.
    verbose : bool
        Print progress and the summary table.

    Returns
    -------
    dict
        {"noise_rates": [...], "ler": [...], "ler_lower": [...],
         "ler_upper": [...], "theory": [...], "csv": path, "plot": path}
    """
    logger = setup_logger("exp_baseline")
    set_seed(seed)
    num_shots = int(config.NUM_SHOTS if num_shots is None else num_shots)
    rates = list(config.NOISE_RATES_TEST if noise_rates is None else noise_rates)

    if verbose:
        print(banner("EXPERIMENT 1: MWPM BASELINE"))
        logger.info(
            "sweeping %d noise rates with %d shots each", len(rates), num_shots
        )

    lers, lowers, uppers, theory, num_failures = [], [], [], [], []

    for i, p in enumerate(rates):
        # A distinct seed per rate: reusing one seed would correlate the
        # points and make the curve look smoother than the data justifies.
        rate_seed = seed + 101 * (i + 1)

        circuit = build_repetition_code_circuit(p)
        syndromes, observables = sample_syndromes(circuit, num_shots, rate_seed)
        decoder = MWPMDecoder(circuit, name=f"MWPM (calibrated p={p})")
        predictions = decoder.decode(syndromes)

        ler, lo, hi = compute_ler_with_confidence(
            predictions, observables, seed=rate_seed
        )

        lers.append(ler)
        lowers.append(lo)
        uppers.append(hi)
        theory.append(theoretical_ler(p))
        num_failures.append(int(np.sum(predictions.reshape(-1) != observables.reshape(-1))))

        logger.info(
            "p=%.3f  LER=%.6f  [%.6f, %.6f]  (%d failures / %d shots, theory %.6f)",
            p, ler, lo, hi, num_failures[-1], num_shots, theory[-1],
        )

        # One peek at raw data, at the training noise rate, so the reader can
        # see what a "shot" actually looks like.
        if verbose and np.isclose(p, config.NOISE_RATE_TRAIN):
            print(f"\nExample shots at the training noise rate p = {p}:")
            print_sample_examples(syndromes, observables, n=10)
            print(f"\nMWPM lookup table learned from the error model at p={p}:")
            for state, pred in decoder.policy_table().items():
                print(
                    f"  syndrome {config.STATE_LABELS[state]} -> "
                    f"predicted logical flip = {pred}"
                )
            print()

    results = {
        "noise_rate": rates,
        "ler_mwpm": lers,
        "ler_lower_95": lowers,
        "ler_upper_95": uppers,
        "theory_3p2_minus_2p3": theory,
        "num_failures": num_failures,
        "num_shots": [num_shots] * len(rates),
    }

    csv_path = save_to_csv(results, config.BASELINE_CSV)
    plot_path = plot_ler_comparison(
        rates,
        {"MWPM (oracle-calibrated)": lers},
        title=f"MWPM baseline, 3-qubit repetition code ({num_shots} shots/point)",
        filename=config.BASELINE_PLOT,
        ci_dict={"MWPM (oracle-calibrated)": (lowers, uppers)},
    )

    if verbose:
        print("\nMWPM baseline summary")
        print_results_table(
            {
                "p": rates,
                "LER": lers,
                "CI low": lowers,
                "CI high": uppers,
                "theory": theory,
                "failures": num_failures,
            }
        )
        print(f"\nsaved table -> {csv_path}")
        print(f"saved plot  -> {plot_path}")
        print(
            "\nReading the result: the measured curve should sit on top of the\n"
            "analytic 3p^2 - 2p^3 line. Quadratic suppression is the signature\n"
            "of a distance-3 code correcting exactly one error; the code only\n"
            "helps while 3p^2 < p, and the gap narrows quickly as p grows."
        )

    return {
        "noise_rates": rates,
        "ler": lers,
        "ler_lower": lowers,
        "ler_upper": uppers,
        "theory": theory,
        "csv": csv_path,
        "plot": plot_path,
    }


if __name__ == "__main__":  # pragma: no cover
    run_baseline_experiment()
