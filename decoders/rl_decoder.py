"""
decoders/rl_decoder.py
======================
Wraps a trained Q-table so the RL agent can be evaluated exactly like the
MWPM baseline.

-----------------------------------------------------------------------
THE ONE CONVERSION THAT MATTERS
-----------------------------------------------------------------------
The two decoders speak different languages:

    MWPM says : "the logical observable was (not) flipped"   -> a bit
    RL says   : "flip data qubit 1"                          -> an action

To score them with the same metric we translate the RL action into the MWPM
language. The logical observable of our circuit is the final measurement of
data qubit 2 (config.LOGICAL_OBSERVABLE_QUBIT). If the agent's correction
includes a flip of that qubit, then applying the correction changes what the
logical readout would have been - which is precisely the statement "I predict
the observable was flipped".

    predicted_observable_flip = correction_bit_on_qubit_2

Concretely, with the minimum-weight policy:

    state [0,0] -> action 0 (no-op)   -> correction 000 -> predict 0
    state [0,1] -> action 3 (flip q2) -> correction 001 -> predict 1
    state [1,0] -> action 1 (flip q0) -> correction 100 -> predict 0
    state [1,1] -> action 2 (flip q1) -> correction 010 -> predict 0

Sanity check on the second line: syndrome [0,1] means the (q1,q2) check fired
while the (q0,q1) check did not, so q2 is the odd one out. The readout qubit
itself flipped, hence "predict the observable was flipped" = 1. Good.

This conversion is a *definition*, not an approximation: for a single round of
this code, applying the correction and reading out is mathematically the same
as XOR-ing the correction's readout-qubit bit into the raw measurement.
-----------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

import config
from circuits.repetition_code import syndrome_to_state


class RLDecoder:
    """Greedy decoder built from a trained tabular Q-table.

    Attributes
    ----------
    q_table : np.ndarray, shape (NUM_STATES, NUM_ACTIONS)
        Learned action values. Row = syndrome state, column = correction.
    name : str
        Label used in logs and plots.
    """

    def __init__(self, q_table: np.ndarray, name: str = "RL (tabular Q)") -> None:
        """Store the Q-table and validate its shape.

        Parameters
        ----------
        q_table : np.ndarray, shape (4, 4)
            Trained Q-values. Produced by training.qlearner.TabularQLearner.
        name : str
            Human-readable label.

        Raises
        ------
        ValueError
            If the table does not have the expected (NUM_STATES, NUM_ACTIONS)
            shape - a wrong shape almost always means a stale .npy file.
        """
        q_table = np.asarray(q_table, dtype=float)
        expected = (config.NUM_STATES, config.NUM_ACTIONS)
        if q_table.shape != expected:
            raise ValueError(f"q_table must have shape {expected}, got {q_table.shape}")

        self.q_table = q_table
        self.name = name

    # ------------------------------------------------------------------
    # Core policy
    # ------------------------------------------------------------------
    def decode(self, syndrome_state: int) -> int:
        """Greedy action for one syndrome state.

        Quantum meaning: the returned action is a *correction*, i.e. which
        data qubit the agent believes was hit by an X error and should be
        flipped back (action 0 = "I think nothing went wrong").

        No exploration happens here - evaluation always uses the greedy
        policy, because epsilon-greedy randomness at test time would measure
        the exploration schedule rather than what was learned.

        Parameters
        ----------
        syndrome_state : int
            State index in {0,1,2,3}; see config.STATE_LABELS.

        Returns
        -------
        int
            Action index in {0,1,2,3}; see config.ACTION_LABELS.
        """
        state = int(syndrome_state)
        if state not in range(config.NUM_STATES):
            raise ValueError(f"state must be in 0..{config.NUM_STATES - 1}, got {state}")
        return int(np.argmax(self.q_table[state]))

    def action_to_observable_prediction(self, action: int) -> int:
        """Translate a correction action into a predicted logical flip.

        See the module docstring: the prediction is simply whether the
        correction touches the readout qubit (data qubit 2).

        Parameters
        ----------
        action : int
            Action index in {0,1,2,3}.

        Returns
        -------
        int
            0 or 1 - the predicted value of the logical observable.
        """
        correction = config.ACTION_CORRECTIONS[int(action)]
        return int(correction[config.LOGICAL_OBSERVABLE_QUBIT])

    # ------------------------------------------------------------------
    # Batch API (mirrors MWPMDecoder)
    # ------------------------------------------------------------------
    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch of syndrome vectors into predicted logical flips.

        Steps: syndrome vector -> integer state -> greedy action ->
        predicted observable flip. The output format is identical to
        MWPMDecoder.decode, so evaluation code does not care which decoder it
        was handed.

        Parameters
        ----------
        syndromes : np.ndarray, shape (num_shots, 2), dtype bool/uint8

        Returns
        -------
        np.ndarray, shape (num_shots, 1), dtype uint8
        """
        syndromes = np.asarray(syndromes)
        if syndromes.ndim != 2 or syndromes.shape[1] != config.NUM_ANCILLA_QUBITS:
            raise ValueError(
                f"syndromes must have shape (N, {config.NUM_ANCILLA_QUBITS}), "
                f"got {syndromes.shape}"
            )

        # Vectorised: state = 2*s0 + s1, then two table lookups. Doing this
        # with numpy instead of a Python loop keeps 10k-shot evaluations
        # instantaneous, which matters when we sweep 7 noise rates x 5 seeds.
        states = 2 * syndromes[:, 0].astype(np.uint8) + syndromes[:, 1].astype(np.uint8)
        greedy_actions = np.argmax(self.q_table, axis=1)            # (4,)
        obs_of_action = np.array(
            [config.ACTION_CORRECTIONS[a][config.LOGICAL_OBSERVABLE_QUBIT]
             for a in range(config.NUM_ACTIONS)],
            dtype=np.uint8,
        )                                                            # (4,)
        predictions = obs_of_action[greedy_actions[states]]
        return predictions.reshape(-1, 1)

    def get_ler(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Logical error rate of this decoder on a batch of shots.

        Parameters
        ----------
        predictions : np.ndarray, shape (num_shots, 1)
        actuals : np.ndarray, shape (num_shots, 1)

        Returns
        -------
        float
            Fraction of shots where the predicted logical flip was wrong.
        """
        predictions = np.asarray(predictions).astype(bool).reshape(-1)
        actuals = np.asarray(actuals).astype(bool).reshape(-1)
        if predictions.shape != actuals.shape:
            raise ValueError(
                f"shape mismatch: {predictions.shape} vs {actuals.shape}"
            )
        return float(np.mean(predictions != actuals))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_policy(self) -> tuple[int, ...]:
        """Return the greedy policy as a tuple indexed by state.

        Returns
        -------
        tuple of 4 ints
            policy[state] = action, directly comparable to
            config.OPTIMAL_POLICY.
        """
        return tuple(int(np.argmax(self.q_table[s])) for s in range(config.NUM_STATES))

    def matches_optimal_policy(self) -> bool:
        """Whether the learned greedy policy equals the minimum-weight policy.

        Returns
        -------
        bool
        """
        return self.get_policy() == tuple(config.OPTIMAL_POLICY)

    def policy_table(self) -> dict[int, int]:
        """Predicted observable flip for each syndrome state.

        Mirrors MWPMDecoder.policy_table so the two lookup tables can be
        diffed entry by entry.

        Returns
        -------
        dict
            {state -> predicted observable flip}
        """
        return {
            state: self.action_to_observable_prediction(self.decode(state))
            for state in range(config.NUM_STATES)
        }

    @classmethod
    def from_file(cls, filepath: str, name: str = "RL (tabular Q)") -> "RLDecoder":
        """Load a Q-table from a .npy file and wrap it.

        Parameters
        ----------
        filepath : str
            Path to a numpy .npy file holding a (4, 4) array.
        name : str

        Returns
        -------
        RLDecoder
        """
        return cls(np.load(filepath), name=name)

    @classmethod
    def from_policy(cls, policy=config.OPTIMAL_POLICY, name: str = "RL (hardcoded)") -> "RLDecoder":
        """Build a decoder from an explicit policy (mainly for testing).

        Parameters
        ----------
        policy : sequence of 4 ints
            policy[state] = action.
        name : str

        Returns
        -------
        RLDecoder
            A decoder whose greedy actions reproduce `policy`.
        """
        q = np.zeros((config.NUM_STATES, config.NUM_ACTIONS))
        for state, action in enumerate(policy):
            q[state, action] = 1.0
        return cls(q, name=name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RLDecoder(name={self.name!r}, policy={self.get_policy()})"


def decode_syndrome_batch_to_actions(q_table: np.ndarray, syndromes: np.ndarray) -> np.ndarray:
    """Convenience: map a batch of syndromes to greedy correction actions.

    Parameters
    ----------
    q_table : np.ndarray, shape (4, 4)
    syndromes : np.ndarray, shape (num_shots, 2)

    Returns
    -------
    np.ndarray, shape (num_shots,), dtype int
        The correction action chosen for each shot.
    """
    states = np.array([syndrome_to_state(s) for s in syndromes], dtype=int)
    greedy = np.argmax(np.asarray(q_table), axis=1)
    return greedy[states]
