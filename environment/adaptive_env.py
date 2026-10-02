"""
environment/adaptive_env.py
===========================
Extended QEC Environment and Syndrome Feature Extractor.

Objectives Met:
- Constructs syndrome states using current syndrome + fixed-length history window H.
- Extracts compact statistical and drift features from sliding window buffers.
- Simulates continuous, dynamic time-varying noise drift environments over time.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from circuits.repetition_code import build_repetition_code_circuit
from simulation.sampler import sample_syndromes_and_errors


class SyndromeFeatureExtractor:
    """Maintains syndrome history window and extracts compact statistical drift features.

    Parameters
    ----------
    history_size : int
        Number of past syndrome states to retain in the history window H.
    stat_window_size : int
        Size of the long sliding window W for empirical distribution estimates.
    short_window_size : int
        Size of the short sliding window W_short for rapid drift detection.
    """

    def __init__(
        self,
        history_size: int = config.HISTORY_WINDOW_SIZE,
        stat_window_size: int = config.STATISTICAL_WINDOW_SIZE,
        short_window_size: int = config.SHORT_WINDOW_SIZE,
    ) -> None:
        self.history_size = int(history_size)
        self.stat_window_size = int(stat_window_size)
        self.short_window_size = int(short_window_size)

        self.history_buffer = deque(maxlen=self.history_size)
        self.stat_buffer = deque(maxlen=self.stat_window_size)

        self.reset()

    def reset(self) -> None:
        """Clear all buffers and reset history/statistics."""
        self.history_buffer.clear()
        self.stat_buffer.clear()
        # Pre-fill history buffer with zero state
        for _ in range(self.history_size):
            self.history_buffer.append(0)

    def update(self, state: int) -> None:
        """Push a new syndrome state into history and statistical buffers.

        Parameters
        ----------
        state : int
            Syndrome state index in {0, 1, 2, 3}.
        """
        self.history_buffer.append(int(state))
        self.stat_buffer.append(int(state))

    def get_history(self) -> tuple[int, ...]:
        """Return the current fixed-length history window H as a tuple.

        Returns
        -------
        tuple of int, length H
        """
        return tuple(self.history_buffer)

    def get_compact_features(self) -> np.ndarray:
        """Compute compact statistical and drift features.

        Features computed:
        1-4. Empirical syndrome probabilities p_hat(s) for s in {0, 1, 2, 3}.
        5. Empirical non-trivial syndrome rate (fraction of non-[0,0] states).
        6. Short-term error rate (over last short_window_size episodes).
        7. Drift magnitude indicator D = |p_hat_short - p_hat_long|.

        Returns
        -------
        np.ndarray, shape (7,), dtype float32
        """
        if len(self.stat_buffer) == 0:
            return np.zeros(7, dtype=np.float32)

        stats_arr = np.array(self.stat_buffer, dtype=int)
        n_total = len(stats_arr)

        # Empirical syndrome state distribution (4 values)
        counts = np.bincount(stats_arr, minlength=config.NUM_STATES)
        p_syndromes = counts.astype(np.float32) / float(n_total)

        # Non-trivial syndrome rate (fraction where state != 0)
        p_nontrivial = float(np.mean(stats_arr != 0))

        # Short-term error rate vs long-term error rate
        n_short = min(len(stats_arr), self.short_window_size)
        short_arr = stats_arr[-n_short:]
        p_short_nontrivial = float(np.mean(short_arr != 0))

        # Drift indicator D: absolute discrepancy between short and long window
        drift_indicator = abs(p_short_nontrivial - p_nontrivial)

        features = np.array(
            [
                p_syndromes[0],
                p_syndromes[1],
                p_syndromes[2],
                p_syndromes[3],
                p_nontrivial,
                p_short_nontrivial,
                drift_indicator,
            ],
            dtype=np.float32,
        )
        return features

    def estimate_per_qubit_noise(self) -> tuple[float, float, float]:
        """Estimate per-qubit error rates from syndrome statistics.

        Returns
        -------
        tuple of 3 floats: (p0_est, p1_est, p2_est)
        """
        feats = self.get_compact_features()
        p0, p1, p2, p3 = feats[0], feats[1], feats[2], feats[3]

        # s0 = p(state 2) + p(state 3) = parity(q0, q1)
        # s1 = p(state 1) + p(state 3) = parity(q1, q2)
        ps0 = p2 + p3
        ps1 = p1 + p3

        # Heuristic estimates of physical error rates under sparse error model
        p1_est = float(np.clip(p3, 0.001, 0.5))
        p0_est = float(np.clip(ps0 - p1_est, 0.001, 0.5))
        p2_est = float(np.clip(ps1 - p1_est, 0.001, 0.5))
        return p0_est, p1_est, p2_est


class DynamicDriftQECEnv(gym.Env):
    """QEC Environment supporting dynamic, time-varying noise drift.

    Unlike standard static environments, this environment allows the noise
    rate or noise profile to drift continuously over episode steps t.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        noise_schedule: Sequence[float] | None = None,
        per_qubit_schedule: Sequence[Sequence[float]] | None = None,
        seed: int = config.SEED,
        batch_size: int = config.SAMPLE_BATCH_SIZE,
    ) -> None:
        super().__init__()

        self.noise_schedule = (
            list(noise_schedule)
            if noise_schedule is not None
            else [config.NOISE_RATE_TRAIN]
        )
        self.per_qubit_schedule = (
            [tuple(r) if r is not None else None for r in per_qubit_schedule]
            if per_qubit_schedule is not None
            else None
        )
        self.seed_val = int(seed)
        self.batch_size = int(batch_size)

        self.step_counter = 0
        self.feature_extractor = SyndromeFeatureExtractor()

        # Observation space is a Dict holding syndrome state, history, and drift features
        self.observation_space = spaces.Dict(
            {
                "current_state": spaces.Discrete(config.NUM_STATES),
                "history": spaces.Box(
                    low=0,
                    high=config.NUM_STATES - 1,
                    shape=(config.HISTORY_WINDOW_SIZE,),
                    dtype=np.int32,
                ),
                "drift_features": spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(config.NUM_ACTIONS)

        self._active_circuit = None
        self._current_noise = self.noise_schedule[0]
        self._current_per_qubit = (
            self.per_qubit_schedule[0] if self.per_qubit_schedule is not None else None
        )

        self._update_circuit()

        self._buffer_index = 0
        self._buf_syndromes = np.empty((0, config.NUM_ANCILLA_QUBITS), dtype=bool)
        self._buf_errors = np.empty((0, config.NUM_DATA_QUBITS), dtype=bool)
        self._buf_observables = np.empty((0, 1), dtype=bool)
        self._refill_counter = 0
        self._refill_buffer()

        self._current_state = 0
        self._current_error = np.zeros(config.NUM_DATA_QUBITS, dtype=bool)
        self._current_observable = False
        self._episode_open = False

    def set_noise_state(
        self,
        noise_rate: float,
        per_qubit_rates: Sequence[float] | None = None,
    ) -> None:
        """Dynamically update the active noise state.

        Parameters
        ----------
        noise_rate : float
        per_qubit_rates : sequence of 3 floats, optional
        """
        self._current_noise = float(noise_rate)
        self._current_per_qubit = (
            tuple(float(r) for r in per_qubit_rates)
            if per_qubit_rates is not None
            else None
        )
        self._update_circuit()
        self._refill_buffer()

    def _update_circuit(self) -> None:
        """Rebuild Stim circuit with current noise configuration."""
        self._active_circuit = build_repetition_code_circuit(
            self._current_noise, per_qubit_rates=self._current_per_qubit
        )

    def _refill_buffer(self) -> None:
        """Draw fresh batch of shots from current circuit."""
        block_seed = (self.seed_val + 7919 * self._refill_counter + self.step_counter) % (
            2**31 - 1
        )
        syndromes, observables, errors = sample_syndromes_and_errors(
            self._active_circuit, num_shots=self.batch_size, seed=block_seed
        )
        self._buf_syndromes = syndromes
        self._buf_observables = observables
        self._buf_errors = errors
        self._buffer_index = 0
        self._refill_counter += 1

    def _next_shot(self) -> tuple[np.ndarray, np.ndarray, bool]:
        if self._buffer_index >= len(self._buf_syndromes):
            self._refill_buffer()
        idx = self._buffer_index
        self._buffer_index += 1
        return (
            self._buf_syndromes[idx],
            self._buf_errors[idx],
            bool(self._buf_observables[idx, 0]),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode and update feature extractor."""
        super().reset(seed=seed)
        if seed is not None:
            self.seed_val = int(seed)
            self._refill_counter = 0
            self._refill_buffer()
            self.feature_extractor.reset()

        # Check if noise schedule changes at this step
        if len(self.noise_schedule) > 1:
            idx = min(self.step_counter, len(self.noise_schedule) - 1)
            new_p = self.noise_schedule[idx]
            new_per_q = (
                self.per_qubit_schedule[idx]
                if self.per_qubit_schedule is not None
                else None
            )
            if new_p != self._current_noise or new_per_q != self._current_per_qubit:
                self.set_noise_state(new_p, new_per_q)

        syndrome, error, observable = self._next_shot()
        self._current_state = int(2 * int(syndrome[0]) + int(syndrome[1]))
        self._current_error = np.asarray(error, dtype=bool).copy()
        self._current_observable = bool(observable)
        self._episode_open = True

        self.feature_extractor.update(self._current_state)

        obs = {
            "current_state": self._current_state,
            "history": np.array(self.feature_extractor.get_history(), dtype=np.int32),
            "drift_features": self.feature_extractor.get_compact_features(),
        }
        return obs, {}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute action, evaluate logical outcome, and update step counter."""
        if not self._episode_open:
            raise RuntimeError("step() called before reset()")

        action = int(action)
        correction = np.array(config.ACTION_CORRECTIONS[action], dtype=bool)
        residual = np.logical_xor(self._current_error, correction)
        survived = not bool(np.any(residual))
        reward = 1.0 if survived else -1.0

        self._episode_open = False
        self.step_counter += 1

        obs = {
            "current_state": self._current_state,
            "history": np.array(self.feature_extractor.get_history(), dtype=np.int32),
            "drift_features": self.feature_extractor.get_compact_features(),
        }

        info = {
            "logical_survived": survived,
            "true_error": self._current_error.astype(np.uint8).tolist(),
            "residual": residual.astype(np.uint8).tolist(),
            "current_p": self._current_noise,
            "current_per_qubit": self._current_per_qubit,
        }

        return obs, reward, True, False, info
