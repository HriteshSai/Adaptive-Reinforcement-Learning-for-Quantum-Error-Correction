"""
simulation/sampler.py
=====================
Turns a Stim circuit into data: run the circuit many times ("shots") and
collect what a real experiment would give you - syndromes and the logical
observable.

Terms
-----
shot
    One complete execution of the circuit: reset -> noise -> parity checks ->
    measurements. Shots are independent, so LER estimates are binomial.
detector sampler
    Stim's fast path for QEC: instead of returning raw measurements, it
    returns the DETECTOR values (the syndrome) and the OBSERVABLE values
    (whether the logical bit came out flipped). This is exactly the
    information available to a real decoder.
measurement sampler
    Returns the raw measurement record instead. We use it only inside the
    environment, where the *simulator* (never the agent) needs ground truth.
"""

from __future__ import annotations

import numpy as np
import stim

import config


def sample_syndromes(
    circuit: stim.Circuit,
    num_shots: int = config.NUM_SHOTS,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample syndromes and logical observables from a Stim circuit.

    What it does
    ------------
    Compiles a detector sampler for the circuit and draws `num_shots`
    independent shots from it.

    Quantum concepts
    ----------------
    * syndrome  : values of the circuit's DETECTORs - the parity checks that
                  fired. This is the decoder's input.
    * observable: value of OBSERVABLE_INCLUDE(0) - whether the logical bit
                  was flipped by the noise. This is the ground truth the
                  decoder's prediction is scored against. A decoder is
                  "correct" on a shot when prediction == observable.

    Parameters
    ----------
    circuit : stim.Circuit
        Circuit with 2 detectors and 1 observable (see circuits/).
    num_shots : int
        Number of independent runs. Defaults to config.NUM_SHOTS.
    seed : int
        Seed for Stim's internal RNG, for reproducibility.

    Returns
    -------
    syndromes : np.ndarray, shape (num_shots, 2), dtype bool
    observables : np.ndarray, shape (num_shots, 1), dtype bool

    Notes
    -----
    Stim only guarantees reproducibility for a fixed version + fixed seed +
    fixed circuit; that is enough for a semester project, but it is why we
    also store the resulting numbers in CSV files.
    """
    if num_shots <= 0:
        raise ValueError(f"num_shots must be positive, got {num_shots}")

    sampler = circuit.compile_detector_sampler(seed=int(seed))
    syndromes, observables = sampler.sample(
        shots=num_shots, separate_observables=True
    )

    syndromes = np.asarray(syndromes, dtype=bool)
    observables = np.asarray(observables, dtype=bool)

    # Defensive shape checks: a silent shape change here would corrupt every
    # downstream LER number, and those errors are painful to trace back.
    expected_dets = config.NUM_ANCILLA_QUBITS
    assert syndromes.shape == (num_shots, expected_dets), (
        f"expected syndromes {(num_shots, expected_dets)}, got {syndromes.shape}"
    )
    assert observables.shape == (num_shots, 1), (
        f"expected observables {(num_shots, 1)}, got {observables.shape}"
    )
    return syndromes, observables


def sample_syndromes_and_errors(
    circuit: stim.Circuit,
    num_shots: int = config.NUM_SHOTS,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample syndromes, observables AND the underlying data-qubit error bits.

    What it does
    ------------
    Uses the raw *measurement* sampler so we also get the final data-qubit
    measurements. In our circuit the data qubits are only ever CNOT controls,
    so they are never disturbed after the noise step: their final measurement
    outcomes ARE the error bits (e0, e1, e2).

    Why this exists
    ---------------
    The RL environment needs ground truth to compute a reward ("did the
    correction actually restore the encoded state?"). That ground truth lives
    in the simulator, NOT in the agent's observation (project rule 2). Nothing
    that is returned as `errors` here is ever passed to the agent.

    Measurement record layout for our circuit:
        index 0 -> ancilla a0  (syndrome bit s0)
        index 1 -> ancilla a1  (syndrome bit s1)
        index 2 -> data q0     (error bit e0)
        index 3 -> data q1     (error bit e1)
        index 4 -> data q2     (error bit e2)

    Parameters
    ----------
    circuit : stim.Circuit
    num_shots : int
    seed : int

    Returns
    -------
    syndromes : np.ndarray, shape (num_shots, 2), dtype bool
    observables : np.ndarray, shape (num_shots, 1), dtype bool
        Observable = final measurement of the logical readout qubit.
    errors : np.ndarray, shape (num_shots, 3), dtype bool
        The actual X-error pattern on the data qubits.
    """
    if num_shots <= 0:
        raise ValueError(f"num_shots must be positive, got {num_shots}")

    sampler = circuit.compile_sampler(seed=int(seed))
    measurements = np.asarray(sampler.sample(shots=num_shots), dtype=bool)

    n_anc = config.NUM_ANCILLA_QUBITS
    syndromes = measurements[:, :n_anc]
    errors = measurements[:, n_anc:]
    observables = errors[:, config.LOGICAL_OBSERVABLE_QUBIT : config.LOGICAL_OBSERVABLE_QUBIT + 1]

    # Sanity: the syndrome must equal the parity of the sampled error bits.
    # If this ever fails, the circuit's gate order was changed incorrectly.
    expected_s0 = errors[:, 0] ^ errors[:, 1]
    expected_s1 = errors[:, 1] ^ errors[:, 2]
    assert np.array_equal(syndromes[:, 0], expected_s0), "syndrome bit 0 inconsistent"
    assert np.array_equal(syndromes[:, 1], expected_s1), "syndrome bit 1 inconsistent"

    return syndromes, observables, errors


def print_sample_examples(
    syndromes: np.ndarray,
    observables: np.ndarray,
    n: int = 10,
    errors: np.ndarray | None = None,
) -> None:
    """Pretty-print the first `n` shots so you can eyeball the data.

    Reading the table
    -----------------
    syndrome [0,0] with observable 0 is the boring, overwhelmingly common
    case: nothing happened. A shot like syndrome [1,0], observable 0 means
    "the q0/q1 parity check fired, and the logical readout qubit was fine" -
    i.e. qubit 0 most likely flipped.

    Parameters
    ----------
    syndromes : np.ndarray, shape (num_shots, 2)
    observables : np.ndarray, shape (num_shots, 1)
    n : int
        How many shots to print.
    errors : np.ndarray, shape (num_shots, 3), optional
        Ground-truth error bits, printed as an extra column when available.
        Debug aid only - decoders never receive this.

    Returns
    -------
    None
    """
    n = int(min(n, len(syndromes)))
    header = f"{'shot':>5} | {'syndrome':>10} | {'state':>5} | {'observable':>10}"
    if errors is not None:
        header += f" | {'true error':>10}"
    print(header)
    print("-" * len(header))

    for i in range(n):
        s0, s1 = int(syndromes[i, 0]), int(syndromes[i, 1])
        state = 2 * s0 + s1
        row = (
            f"{i:>5} | {f'[{s0},{s1}]':>10} | {state:>5} | "
            f"{int(observables[i, 0]):>10}"
        )
        if errors is not None:
            e = "".join(str(int(b)) for b in errors[i])
            row += f" | {e:>10}"
        print(row)

    frac = float(np.mean(np.any(syndromes, axis=1)))
    print("-" * len(header))
    print(f"non-trivial syndromes in this batch: {frac:.4%}")
    print(f"logical observable flipped:          {float(np.mean(observables)):.4%}")
