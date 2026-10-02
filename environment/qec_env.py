"""
environment/qec_env.py
======================
The Gymnasium environment that turns quantum error correction into a
one-step reinforcement learning problem.

-----------------------------------------------------------------------
THE MDP IN ONE PARAGRAPH
-----------------------------------------------------------------------
Each episode is one round of the memory experiment:

    reset()  -> Stim runs the circuit once. Noise hits the data qubits, the
                two parity checks are measured, and the resulting SYNDROME
                (4 possible values) is handed to the agent as its state.
                The true error pattern stays hidden inside the environment.

    step(a)  -> The agent names a correction: do nothing, or flip one data
                qubit. The environment applies it to the hidden error and
                checks whether the encoded ("logical") bit survived.
                reward = +1 survived, -1 lost. The episode ends immediately.

So this is really a contextual bandit (a one-step MDP): there is no sequence
of decisions to plan over, only "given this syndrome, what is the best
correction?". That is exactly the right shape for Phase 1 - and it is why a
4x4 lookup table is enough, no neural network required.

-----------------------------------------------------------------------
HOW THE REWARD IS DECIDED (read this before touching the code)
-----------------------------------------------------------------------
Let e = (e0,e1,e2) be the hidden X-error bits and c = the correction implied
by the action. The residual error after correction is r = e XOR c.

    r = 000  ->  the encoded state is exactly restored.      reward = +1
    r = 111  ->  all three qubits flipped: that is a LOGICAL X operation.
                 The syndrome is still 000, so no decoder could ever detect
                 it. The logical bit is lost.                reward = -1
    otherwise -> the residual still violates a parity check, i.e. the
                 correction pushed the state OUT of the code space instead of
                 back into it. The round failed.             reward = -1

Design note (worth a paragraph in your report): an alternative convention
scores a residual of weight 1 as a success, on the grounds that a majority
vote over the three final measurements would still return the right bit.
We deliberately do NOT do that, for two reasons.
  1. Under that convention "do nothing" and "flip q0" earn *identical*
     expected reward when the syndrome is [1,0], so several policies tie for
     optimal and the learned table becomes ambiguous - you could never tell a
     converged agent from a lazy one.
  2. The strict convention makes the environment's success rate numerically
     identical to the logical error rate that MWPM is scored with (both equal
     1 - [(1-p)^3 + 3p(1-p)^2] for the minimum-weight policy), so the RL
     reward and the published metric measure the same thing.

-----------------------------------------------------------------------
INVARIANTS ENFORCED HERE (project rules 1 and 2)
-----------------------------------------------------------------------
* The observation returned by reset() is the syndrome integer and nothing
  else. The error pattern is stored in a private attribute prefixed with an
  underscore and is only exposed through `info` AFTER the episode ends, for
  debugging and logging.
* The reward depends only on the logical outcome. There is no fidelity, no
  partial credit, no shaping term.
-----------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from circuits.repetition_code import build_repetition_code_circuit
from simulation.sampler import sample_syndromes_and_errors


class QECDecoderEnv(gym.Env):
    """One-step Gymnasium environment for decoding the 3-qubit repetition code.

    Observation space
        Discrete(4) - the syndrome, encoded as 2*s0 + s1.
    Action space
        Discrete(4) - 0: no correction, 1: flip q0, 2: flip q1, 3: flip q2.
    Reward
        +1 if the logical bit survived the round, -1 otherwise.
    Episode length
        Always 1 step.

    Attributes
    ----------
    noise_rate : float
        Physical X-error probability per data qubit used to build the circuit.
    circuit : stim.Circuit
        The Stim circuit being sampled.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        noise_rate: float = config.NOISE_RATE_TRAIN,
        seed: int = config.SEED,
        per_qubit_rates: Sequence[float] | None = None,
        batch_size: int = config.SAMPLE_BATCH_SIZE,
    ) -> None:
        """Create the environment and its Stim circuit.

        Parameters
        ----------
        noise_rate : float
            Physical error rate p in [0, 1]. Defaults to the training rate.
        seed : int
            Seed for the Stim sampler, for reproducible episode streams.
        per_qubit_rates : sequence of 3 floats, optional
            Asymmetric noise (used by the biased-noise drift study). When
            given it overrides `noise_rate` inside the circuit; `noise_rate`
            is kept only as a label.
        batch_size : int
            How many shots to pull from Stim at once. Sampling in blocks and
            serving them one at a time is ~100x faster than calling Stim per
            episode, and it changes nothing statistically because shots are
            independent and identically distributed.
        """
        super().__init__()

        self.noise_rate = float(noise_rate)
        self.per_qubit_rates = (
            None if per_qubit_rates is None else tuple(float(r) for r in per_qubit_rates)
        )
        self.batch_size = int(batch_size)
        self._seed = int(seed)

        # ---- spaces (project rules 2 and 3) -----------------------------
        # Discrete(4) on both sides: 4 syndromes in, 4 corrections out.
        self.observation_space = spaces.Discrete(config.NUM_STATES)
        self.action_space = spaces.Discrete(config.NUM_ACTIONS)

        # ---- the quantum simulation ------------------------------------
        self.circuit = build_repetition_code_circuit(
            self.noise_rate, per_qubit_rates=self.per_qubit_rates
        )

        # ---- shot buffer -------------------------------------------------
        self._buffer_index = 0
        self._buf_syndromes = np.empty((0, config.NUM_ANCILLA_QUBITS), dtype=bool)
        self._buf_errors = np.empty((0, config.NUM_DATA_QUBITS), dtype=bool)
        self._buf_observables = np.empty((0, 1), dtype=bool)
        self._refill_counter = 0
        self._refill_buffer()

        # ---- hidden ground truth for the current episode ------------------
        # Underscore prefix = "the agent must never read this".
        self._current_state: int | None = None
        self._current_error = np.zeros(config.NUM_DATA_QUBITS, dtype=bool)
        self._current_observable = False
        self._episode_open = False

        # ---- bookkeeping --------------------------------------------------
        self.episode_count = 0
        self.total_reward = 0.0

    # ------------------------------------------------------------------
    # Internal plumbing
    # ------------------------------------------------------------------
    def _refill_buffer(self) -> None:
        """Draw a fresh block of shots from Stim into the internal buffer.

        Each refill uses a different derived seed so the environment does not
        replay the same block over and over, while the whole stream stays a
        deterministic function of the seed given to __init__.
        """
        block_seed = (self._seed + 7919 * self._refill_counter) % (2**31 - 1)
        syndromes, observables, errors = sample_syndromes_and_errors(
            self.circuit, num_shots=self.batch_size, seed=block_seed
        )
        self._buf_syndromes = syndromes
        self._buf_observables = observables
        self._buf_errors = errors
        self._buffer_index = 0
        self._refill_counter += 1

    def _next_shot(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return the next (syndrome, error, observable) triple from the buffer.

        Returns
        -------
        syndrome : np.ndarray, shape (2,), dtype bool
        error : np.ndarray, shape (3,), dtype bool
        observable : bool
        """
        if self._buffer_index >= len(self._buf_syndromes):
            self._refill_buffer()
        i = self._buffer_index
        self._buffer_index += 1
        return (
            self._buf_syndromes[i],
            self._buf_errors[i],
            bool(self._buf_observables[i, 0]),
        )

    @staticmethod
    def _residual(error: np.ndarray, action: int) -> np.ndarray:
        """Apply a correction to an error pattern and return the residual.

        Quantum concept: X errors compose by XOR. Flipping a qubit that was
        already flipped restores it, so "error then correction" is just the
        bitwise XOR of the two patterns.

        Parameters
        ----------
        error : np.ndarray, shape (3,), dtype bool
            The hidden X-error pattern.
        action : int
            Action index; mapped through config.ACTION_CORRECTIONS.

        Returns
        -------
        np.ndarray, shape (3,), dtype bool
            Residual error r = e XOR c.
        """
        correction = np.array(config.ACTION_CORRECTIONS[int(action)], dtype=bool)
        return np.logical_xor(np.asarray(error, dtype=bool), correction)

    @classmethod
    def _logical_survived(cls, error: np.ndarray, action: int) -> bool:
        """Decide whether the encoded bit survived the round.

        See the module docstring for the full argument. Summary: the round is
        a success if and only if the residual error is the identity, because
        the only other residual with a trivial syndrome is the all-ones
        pattern, which is a logical X (undetectable and fatal), and any other
        residual leaves the state outside the code space.

        Parameters
        ----------
        error : np.ndarray, shape (3,)
        action : int

        Returns
        -------
        bool
            True = logical bit survived.
        """
        return not bool(np.any(cls._residual(error, action)))

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Start a new episode by running the quantum circuit once.

        Quantum concepts
        ----------------
        Sampling one shot means: reset all qubits to |0>, let each data qubit
        suffer an X error with probability p, entangle the ancillas with the
        data via CNOTs, and measure the ancillas. The resulting syndrome is
        the agent's entire view of the world.

        Parameters
        ----------
        seed : int, optional
            Re-seeds the shot stream (Gymnasium API). Passing a seed discards
            the current buffer so the new stream really starts fresh.
        options : dict, optional
            Unused; present for Gymnasium API compatibility.

        Returns
        -------
        state : int
            Syndrome index in {0,1,2,3}.
        info : dict
            Empty at reset - deliberately, so no ground truth can leak into
            the agent before it acts.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self._refill_counter = 0
            self._refill_buffer()

        syndrome, error, observable = self._next_shot()

        # state = 2*s0 + s1 (see config for the encoding table)
        self._current_state = int(2 * int(syndrome[0]) + int(syndrome[1]))
        self._current_error = np.asarray(error, dtype=bool).copy()
        self._current_observable = bool(observable)
        self._episode_open = True

        return self._current_state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Apply a correction and score the round.

        Parameters
        ----------
        action : int
            0 = no correction, 1 = flip q0, 2 = flip q1, 3 = flip q2.

        Returns
        -------
        next_state : int
            The same syndrome index. The episode is over, so the value is
            irrelevant to learning; we return the current state rather than a
            fake terminal index to keep the observation space honest.
        reward : float
            +1.0 if the logical bit survived, -1.0 if it was lost.
        terminated : bool
            Always True - one decision per episode.
        truncated : bool
            Always False - there is no time limit to hit.
        info : dict
            Post-hoc diagnostics: the true error, the residual, the optimal
            action for this syndrome and whether the agent matched it. This is
            returned AFTER the decision, so it cannot influence the agent; it
            exists so training curves and unit tests can be written.
        """
        if not self._episode_open:
            raise RuntimeError("step() called before reset() - one reset per episode")
        if not self.action_space.contains(int(action)):
            raise ValueError(
                f"invalid action {action}; expected 0..{config.NUM_ACTIONS - 1}"
            )

        action = int(action)
        residual = self._residual(self._current_error, action)
        survived = not bool(np.any(residual))

        # RULE 1: reward is a pure function of the logical outcome.
        reward = 1.0 if survived else -1.0

        state = int(self._current_state)
        optimal_action = int(config.OPTIMAL_POLICY[state])

        info = {
            "true_error": self._current_error.astype(np.uint8).tolist(),
            "residual": residual.astype(np.uint8).tolist(),
            "logical_survived": survived,
            "observable": int(self._current_observable),
            "optimal_action": optimal_action,
            "action_was_optimal": action == optimal_action,
            "uncorrectable": bool(
                np.count_nonzero(self._current_error) >= 2
            ),  # >=2 flips cannot be fixed by any single-qubit correction
        }

        self._episode_open = False
        self.episode_count += 1
        self.total_reward += reward

        return state, reward, True, False, info

    def render(self) -> str:
        """Return a one-line textual view of the current episode.

        Returns
        -------
        str
            Human-readable snapshot (syndrome + hidden truth, for debugging).
        """
        if self._current_state is None:
            return "<no episode started>"
        e = "".join(str(int(b)) for b in self._current_error)
        return (
            f"syndrome={config.STATE_LABELS[self._current_state]} "
            f"(state {self._current_state}) | hidden error={e} | "
            f"optimal action={config.ACTION_LABELS[config.OPTIMAL_POLICY[self._current_state]]}"
        )

    # ------------------------------------------------------------------
    # RULE 9: environment self-test
    # ------------------------------------------------------------------
    def verify_with_optimal_policy(
        self,
        num_episodes: int | None = None,
        verbose: bool = True,
    ) -> float:
        """Run the known-optimal decoder in this environment and report its success rate.

        Why this exists
        ---------------
        RL is finicky: when an agent fails to learn, the bug is in the
        environment at least as often as in the algorithm. Before training
        anything we drive the environment with the textbook minimum-weight
        policy, whose success rate we can compute analytically:

            P(success) = (1-p)^3 + 3p(1-p)^2

        At p = 0.03 that is 99.74%. If the measured rate is far from the
        prediction, the reward function, the syndrome encoding or the circuit
        is wrong - and no amount of hyperparameter tuning would fix it.

        Hardcoded policy (config.OPTIMAL_POLICY):
            syndrome [0,0] -> action 0 (no correction)
            syndrome [1,0] -> action 1 (flip q0)
            syndrome [1,1] -> action 2 (flip q1)
            syndrome [0,1] -> action 3 (flip q2)

        Parameters
        ----------
        num_episodes : int
            Episodes to run. Defaults to config.VERIFY_EPISODES.
        verbose : bool
            Print a short report.

        Returns
        -------
        float
            Empirical success rate in [0, 1]. Compare against
            config.VERIFY_MIN_SUCCESS_RATE; below that, the environment has a
            bug and training must not proceed.
        """
        # Resolved at call time so runtime config overrides are honoured.
        num_episodes = int(config.VERIFY_EPISODES if num_episodes is None else num_episodes)
        successes = 0
        per_state_visits = np.zeros(config.NUM_STATES, dtype=int)
        per_state_success = np.zeros(config.NUM_STATES, dtype=int)

        for _ in range(int(num_episodes)):
            state, _ = self.reset()
            action = int(config.OPTIMAL_POLICY[state])
            _, reward, _, _, _ = self.step(action)
            per_state_visits[state] += 1
            if reward > 0:
                successes += 1
                per_state_success[state] += 1

        success_rate = successes / float(num_episodes)

        if verbose:
            p = self.noise_rate
            theory = (1 - p) ** 3 + 3 * p * (1 - p) ** 2
            print("\n[ENV VERIFICATION] hardcoded minimum-weight policy")
            print(f"  noise rate p              : {p}")
            print(f"  episodes                  : {num_episodes}")
            print(f"  measured success rate     : {success_rate:.4%}")
            if self.per_qubit_rates is None:
                print(f"  analytic prediction       : {theory:.4%}")
                print(f"  |measured - analytic|     : {abs(success_rate - theory):.4%}")
            print(f"  pass threshold            : {config.VERIFY_MIN_SUCCESS_RATE:.2%}")
            print("  per-syndrome breakdown:")
            for s in range(config.NUM_STATES):
                visits = per_state_visits[s]
                rate = per_state_success[s] / visits if visits else float("nan")
                print(
                    f"    {config.STATE_LABELS[s]} (state {s}): "
                    f"{visits:>6} visits, success {rate:.4%}, "
                    f"action = {config.ACTION_LABELS[config.OPTIMAL_POLICY[s]]}"
                )
            verdict = "PASS" if success_rate >= config.VERIFY_MIN_SUCCESS_RATE else "FAIL"
            print(f"  verdict                   : {verdict}\n")

        return success_rate

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def analytic_optimal_success_rate(self) -> float:
        """Analytic success rate of the minimum-weight policy at this noise rate.

        Derivation: a single-qubit correction succeeds exactly when at most
        one data qubit was hit, i.e. P = (1-p)^3 + 3p(1-p)^2 for uniform
        noise. For asymmetric noise the same enumeration is done term by term,
        but a per-qubit correction only wins when the *most likely* error
        consistent with the syndrome is the one that actually happened, so the
        formula below applies to uniform noise only.

        Returns
        -------
        float
            Probability that the minimum-weight policy succeeds.
        """
        if self.per_qubit_rates is not None:
            rates = np.array(self.per_qubit_rates, dtype=float)
        else:
            rates = np.full(config.NUM_DATA_QUBITS, self.noise_rate, dtype=float)
        no_error = float(np.prod(1 - rates))
        single = 0.0
        for i in range(config.NUM_DATA_QUBITS):
            term = rates[i]
            for j in range(config.NUM_DATA_QUBITS):
                if j != i:
                    term *= 1 - rates[j]
            single += term
        return no_error + single

    def close(self) -> None:
        """Release references to the sampling buffers (Gymnasium API)."""
        self._buf_syndromes = np.empty((0, config.NUM_ANCILLA_QUBITS), dtype=bool)
        self._buf_errors = np.empty((0, config.NUM_DATA_QUBITS), dtype=bool)
        self._buf_observables = np.empty((0, 1), dtype=bool)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"QECDecoderEnv(p={self.noise_rate}, "
            f"per_qubit_rates={self.per_qubit_rates}, seed={self._seed})"
        )
