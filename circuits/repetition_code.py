"""
circuits/repetition_code.py
===========================
Builds the Stim circuit for a single round of the 3-qubit bit-flip
repetition code.

-----------------------------------------------------------------------
CRASH COURSE IN THE QUANTUM TERMS USED IN THIS FILE
-----------------------------------------------------------------------
qubit
    The quantum version of a bit. For this project you can safely picture it
    as a bit that is either |0> or |1>; the code we simulate only ever has to
    deal with bit-flip ("X") errors, which is why a purely classical mental
    model works here.

X error / bit flip
    The X gate maps |0> -> |1> and |1> -> |0>. `X_ERROR(p) 0` in Stim means
    "with probability p, apply an X gate to qubit 0". This is our noise model.

repetition code
    The simplest error-correcting code: store one logical bit in three
    physical qubits by repeating it, |0>_L = |000> and |1>_L = |111>. If one
    of the three qubits flips, the other two out-vote it. It cannot correct
    two simultaneous flips - that is precisely the failure mode this project
    measures.

stabilizer / parity check
    Instead of measuring a data qubit (which would destroy the encoded
    information), we measure *relationships* between qubits: "is qubit 0 the
    same as qubit 1?" and "is qubit 1 the same as qubit 2?". In stabilizer
    language these are the operators Z0*Z1 and Z1*Z2. Their outcomes tell you
    where a disagreement is without revealing whether the logical bit is 0
    or 1, so the encoded information survives the measurement.

CNOT (controlled-NOT)
    Two-qubit gate: `CX control target` flips the target if the control is
    |1>. WHY we use it here: applying CX from a data qubit onto a fresh
    ancilla *copies the data qubit's value into the ancilla's parity* without
    measuring the data qubit. Doing it twice, from qubits 0 and 1 onto the
    same ancilla, leaves the ancilla holding the XOR (the parity) of those two
    data qubits. Measuring the ancilla then reveals only that parity.

ancilla
    The scratch qubit that absorbs the parity information and is measured.
    Qubits 3 and 4 in this circuit.

syndrome
    The pair of ancilla measurement outcomes, e.g. [1,0]. Under our noise
    model, with error bits (e0,e1,e2) on the data qubits, the syndrome is
    exactly (e0 XOR e1, e1 XOR e2). Note that it is invariant under flipping
    ALL THREE qubits - that ambiguity is the reason a decoder can ever fail.

DETECTOR
    A Stim annotation marking a measurement (or parity of measurements) whose
    value is deterministic in a noiseless run. Any detector that fires (=1)
    signals that noise happened. Detectors are what decoders consume. Because
    all qubits here start in |0> and the measurements are perfect, each
    ancilla measurement is 0 in a noiseless run, so each one is a valid
    detector on its own.

OBSERVABLE_INCLUDE
    A Stim annotation marking the *logical* quantity we care about preserving.
    Here it is the final measurement of data qubit 2. In a noiseless run it
    reads 0; if the decoder cannot undo the noise, it reads 1 and we have
    suffered a logical error.

rec[-1], rec[-2]
    Stim's measurement record, indexed from the end. rec[-1] is the most
    recent measurement, rec[-2] the one before it.
-----------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import stim

import config


def build_repetition_code_circuit(
    p: float,
    per_qubit_rates: Sequence[float] | None = None,
) -> stim.Circuit:
    """Build one round of the 3-qubit bit-flip repetition code as a Stim circuit.

    What it does
    ------------
    Constructs the memory experiment:
        1. Reset all 5 qubits to |0> (so the encoded logical state is |0>_L,
           i.e. |000> on the data qubits).
        2. Apply an independent X error with probability p to each DATA qubit
           only. Ancillas are assumed noiseless: this is Phase 1, we study
           decoder behaviour, not measurement noise.
        3. Extract the two parity checks with CNOTs into the two ancillas.
        4. Measure the ancillas -> the syndrome; declare both as DETECTORs.
        5. Measure the three data qubits (destructive readout) and declare the
           last one as logical OBSERVABLE 0.

    Why the observable is a single data qubit
    -----------------------------------------
    For a repetition code, "did the logical bit flip?" can be read off any one
    data qubit *once the decoder's correction has been accounted for*. Stim's
    convention is therefore: observable = one raw data measurement, and the
    decoder must predict whether that measurement was flipped by noise. Both
    the MWPM baseline and the RL agent are scored on exactly this prediction,
    which makes the comparison apples-to-apples.

    Parameters
    ----------
    p : float
        Physical X-error probability per data qubit, in [0, 1].
    per_qubit_rates : sequence of 3 floats, optional
        If given, overrides the uniform rate: qubit i gets per_qubit_rates[i].
        Used by the biased-noise study in experiments/exp_drift.py, where the
        noise SHAPE (not just its magnitude) drifts away from training
        conditions. Each entry must lie in [0, 1].

    Returns
    -------
    stim.Circuit
        A circuit with 2 detectors and 1 logical observable. Sampling it with
        `compile_detector_sampler()` yields
            detectors   : shape (num_shots, 2), dtype bool
            observables : shape (num_shots, 1), dtype bool

    Raises
    ------
    ValueError
        If p (or any per-qubit rate) is not a probability in [0, 1].
    """
    if not isinstance(p, (int, float)):
        raise ValueError(f"p must be a number, got {type(p).__name__}")
    if not (0.0 <= float(p) <= 1.0):
        raise ValueError(f"p must be a probability in [0, 1], got {p}")

    if per_qubit_rates is None:
        rates = [float(p)] * config.NUM_DATA_QUBITS
    else:
        rates = [float(r) for r in per_qubit_rates]
        if len(rates) != config.NUM_DATA_QUBITS:
            raise ValueError(
                f"per_qubit_rates must have {config.NUM_DATA_QUBITS} entries, "
                f"got {len(rates)}"
            )
        for r in rates:
            if not (0.0 <= r <= 1.0):
                raise ValueError(f"per-qubit rate must be in [0, 1], got {r}")

    d0, d1, d2 = config.DATA_QUBITS
    a0, a1 = config.ANCILLA_QUBITS

    circuit = stim.Circuit()

    # -- 1. Initialise -----------------------------------------------------
    # R = reset to |0>. Resetting the ancillas too is essential: an ancilla
    # must start from a known state, otherwise its measurement would not be a
    # deterministic 0 in a noiseless run and could not serve as a detector.
    circuit.append("R", [d0, d1, d2, a0, a1])

    # -- 2. Noise ----------------------------------------------------------
    # Errors are applied to DATA qubits only, and independently per qubit.
    # We emit one X_ERROR instruction per qubit so that the asymmetric
    # (biased) case is expressible with the same code path.
    for qubit, rate in zip(config.DATA_QUBITS, rates):
        circuit.append("X_ERROR", [qubit], rate)

    # -- 3. Parity extraction ---------------------------------------------
    # WHY CNOTs: each CX writes the control's value into the target's parity.
    # After CX d0->a0 and CX d1->a0, ancilla a0 holds e0 XOR e1. The data
    # qubits are controls, so they are never disturbed - we learn a
    # relationship, not a value, and the encoded information survives.
    circuit.append("CX", [d0, a0])
    circuit.append("CX", [d1, a0])
    circuit.append("CX", [d1, a1])
    circuit.append("CX", [d2, a1])

    # -- 4. Syndrome measurement ------------------------------------------
    # M measures in the Z basis and appends outcomes to the measurement
    # record: rec[-2] is a0's outcome, rec[-1] is a1's outcome.
    circuit.append("M", [a0, a1])

    # Each ancilla outcome is deterministically 0 without noise, so each is a
    # detector on its own. Detector order defines the syndrome bit order:
    #   detector 0 = parity(q0, q1),  detector 1 = parity(q1, q2)
    circuit.append("DETECTOR", [stim.target_rec(-2)], [])
    circuit.append("DETECTOR", [stim.target_rec(-1)], [])

    # -- 5. Logical readout ------------------------------------------------
    # Destructive measurement of the data qubits. Because the data qubits were
    # only ever used as CNOT *controls*, these outcomes are literally the
    # error bits (e0, e1, e2) - a fact the environment exploits to compute
    # ground-truth rewards (the agent never sees them).
    circuit.append("M", [d0, d1, d2])

    # rec[-1] is the measurement of data qubit 2 -> our logical observable.
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-1)],
        config.LOGICAL_OBSERVABLE_INDEX,
    )

    return circuit


def syndrome_to_state(syndrome) -> int:
    """Convert a 2-bit syndrome vector into the integer RL state index.

    Quantum concept: the *syndrome* is the pair of parity-check outcomes
    (detector values). It is the ONLY thing the RL agent is ever allowed to
    observe (project rule 2).

    Parameters
    ----------
    syndrome : sequence of 2 ints/bools
        [s0, s1] with s0 = parity(q0,q1) and s1 = parity(q1,q2).

    Returns
    -------
    int
        state = 2 * s0 + s1, in {0, 1, 2, 3}; see config.STATE_LABELS.
    """
    s0 = int(bool(syndrome[0]))
    s1 = int(bool(syndrome[1]))
    return 2 * s0 + s1


def state_to_syndrome(state: int) -> tuple[int, int]:
    """Inverse of :func:`syndrome_to_state`.

    Parameters
    ----------
    state : int
        Integer state index in {0, 1, 2, 3}.

    Returns
    -------
    tuple of 2 ints
        The syndrome bits (s0, s1).
    """
    if state not in range(config.NUM_STATES):
        raise ValueError(f"state must be in 0..{config.NUM_STATES - 1}, got {state}")
    return (state >> 1) & 1, state & 1


def describe_circuit(p: float) -> str:
    """Return a human-readable dump of the circuit plus its error model.

    Useful for a report appendix or for convincing yourself the circuit is
    what you think it is. The *detector error model* (DEM) is Stim's
    compilation of the circuit into "which error mechanisms flip which
    detectors and observables" - it is exactly what MWPM decoders consume.

    Parameters
    ----------
    p : float
        Physical error rate to build the circuit at.

    Returns
    -------
    str
        Multi-line description: the circuit source and its DEM.
    """
    circuit = build_repetition_code_circuit(p)
    dem = circuit.detector_error_model(decompose_errors=True)
    lines = [
        f"Repetition-code circuit at p = {p}",
        "-" * 60,
        str(circuit),
        "-" * 60,
        "Detector error model (what MWPM sees):",
        str(dem),
        "-" * 60,
        f"num_detectors  = {circuit.num_detectors}",
        f"num_observables= {circuit.num_observables}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    print(describe_circuit(config.NOISE_RATE_TRAIN))
