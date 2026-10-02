"""
decoders/mwpm_decoder.py
========================
The classical baseline: Minimum-Weight Perfect Matching (MWPM) via PyMatching.

-----------------------------------------------------------------------
WHAT MWPM IS, IN PLAIN LANGUAGE
-----------------------------------------------------------------------
detector error model (DEM)
    Stim can analyse a circuit and list every elementary error mechanism
    together with (a) its probability and (b) which detectors and observables
    it flips. For our repetition code the DEM has three mechanisms:
        X on q0 -> flips detector 0                    (and not the observable)
        X on q1 -> flips detectors 0 and 1
        X on q2 -> flips detector 1 AND the observable
    That is a graph: detectors are nodes, error mechanisms are edges. An error
    that flips only one detector becomes an edge to a virtual "boundary" node.

matching
    Detectors that fired must be paired up (or paired to the boundary) by a
    set of edges = a hypothesis about which errors occurred. MWPM picks the
    pairing with the smallest total weight, where an edge's weight is
    log((1-p)/p): unlikely errors are expensive. So MWPM = "most likely
    explanation of the observed syndrome, assuming independent errors".

what MWPM outputs
    Whether the chosen set of error mechanisms flips the logical observable.
    That is a single bit per shot: the PREDICTED observable flip. The decoder
    is correct on a shot when this prediction equals the true observable.

why it is the right baseline
    For codes like this one, MWPM is the standard, near-optimal decoder used
    throughout the QEC literature. Beating it is not the goal of this study;
    measuring how each decoder degrades under noise drift is.

calibration / "oracle" vs "fixed"
    The edge weights come from the p that was used to build the circuit. A
    "fixed" MWPM is calibrated at the training rate p=0.03 and then used at
    every test rate; an "oracle" MWPM is rebuilt at each test rate, i.e. it
    always knows the true noise. The gap between them is the price of a stale
    noise model - exactly the drift effect we are measuring for the RL agent.
-----------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pymatching
import stim

import config


class MWPMDecoder:
    """Minimum-weight perfect matching decoder wrapping PyMatching.

    Attributes
    ----------
    circuit : stim.Circuit
        The circuit the decoder was calibrated on (its p sets the weights).
    dem : stim.DetectorErrorModel
        Compiled error model extracted from the circuit.
    matcher : pymatching.Matching
        The matching graph used for decoding.
    """

    def __init__(self, circuit: stim.Circuit, name: str = "MWPM") -> None:
        """Build a matching graph from a Stim circuit.

        Parameters
        ----------
        circuit : stim.Circuit
            Circuit including DETECTOR and OBSERVABLE_INCLUDE annotations. The
            physical error rate baked into this circuit is what calibrates the
            matching weights.
        name : str
            Human-readable label used in logs/plots, e.g. "MWPM (fixed p=0.03)".

        Notes
        -----
        `decompose_errors=True` asks Stim to split any error that flips more
        than two detectors into graph-like pieces. Our errors already flip at
        most two detectors, so it is a no-op here, but it is the habit to keep
        for bigger codes in Phase 2.
        """
        if circuit.num_observables != 1:
            raise ValueError(
                f"expected exactly 1 logical observable, got {circuit.num_observables}"
            )

        self.circuit = circuit
        self.name = name
        self.dem = circuit.detector_error_model(decompose_errors=True)
        self.matcher = pymatching.Matching.from_detector_error_model(self.dem)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch of syndromes into predicted logical flips.

        Parameters
        ----------
        syndromes : np.ndarray, shape (num_shots, 2), dtype bool/uint8
            Detector outcomes, as returned by simulation.sampler.

        Returns
        -------
        np.ndarray, shape (num_shots, 1), dtype uint8
            predictions[i, 0] == 1 means "I believe the logical observable was
            flipped on shot i". Compare against the sampled observables to get
            the logical error rate.
        """
        syndromes = np.asarray(syndromes)
        if syndromes.ndim != 2 or syndromes.shape[1] != config.NUM_ANCILLA_QUBITS:
            raise ValueError(
                f"syndromes must have shape (N, {config.NUM_ANCILLA_QUBITS}), "
                f"got {syndromes.shape}"
            )

        predictions = self.matcher.decode_batch(syndromes.astype(np.uint8))
        predictions = np.asarray(predictions, dtype=np.uint8).reshape(len(syndromes), -1)
        return predictions[:, :1]

    # Alias so MWPMDecoder and RLDecoder expose the same batch API.
    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Alias of :meth:`decode`, for interface parity with RLDecoder.

        Parameters
        ----------
        syndromes : np.ndarray, shape (num_shots, 2)

        Returns
        -------
        np.ndarray, shape (num_shots, 1), dtype uint8
        """
        return self.decode(syndromes)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def get_ler(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Logical error rate = fraction of shots where the prediction is wrong.

        Quantum meaning: a mismatch means the correction implied by the
        decoder leaves the encoded bit flipped - the code failed on that shot.

        Parameters
        ----------
        predictions : np.ndarray, shape (num_shots, 1)
        actuals : np.ndarray, shape (num_shots, 1)

        Returns
        -------
        float
            LER in [0, 1].
        """
        predictions = np.asarray(predictions).astype(bool).reshape(-1)
        actuals = np.asarray(actuals).astype(bool).reshape(-1)
        if predictions.shape != actuals.shape:
            raise ValueError(
                f"shape mismatch: predictions {predictions.shape} vs "
                f"actuals {actuals.shape}"
            )
        return float(np.mean(predictions != actuals))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def policy_table(self) -> dict[int, int]:
        """Return the decoder's decision for each of the 4 possible syndromes.

        Why this is useful: it converts the matching graph into the same
        4-entry lookup table the RL agent learns, so the two decoders can be
        compared symbolically and not just by their error rates.

        Returns
        -------
        dict
            {state index -> predicted observable flip (0 or 1)}
        """
        all_syndromes = np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8
        )
        preds = self.decode(all_syndromes)
        return {state: int(preds[state, 0]) for state in range(config.NUM_STATES)}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MWPMDecoder(name={self.name!r}, detectors={self.circuit.num_detectors})"
