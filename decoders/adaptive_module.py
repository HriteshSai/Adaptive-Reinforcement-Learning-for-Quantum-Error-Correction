"""
decoders/adaptive_module.py
===========================
Adaptive RL Decision Module and Meta-Controller.

Objectives Met:
- Designs an adaptive decision module selecting:
  (1) Direct decoding
  (2) MWPM recalibration
  (3) RL policy fine-tuning
- Evaluates selective adaptation vs continuous fine-tuning under operational cost constraints.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config
from circuits.repetition_code import build_repetition_code_circuit
from decoders.mwpm_decoder import MWPMDecoder
from decoders.rl_decoder import RLDecoder


class AdaptiveRLController:
    """Meta-Controller evaluating syndrome drift features to select adaptation strategy.

    Action choices:
        0: Direct Decoding (no update cost)
        1: MWPM Recalibration (rebuild matching graph from statistical features)
        2: RL Policy Fine-Tuning (perform online Q-table update)

    Parameters
    ----------
    base_q_table : np.ndarray, shape (4, 4)
        Initial Q-table trained on baseline noise.
    drift_threshold : float
        Threshold on drift indicator feature D to trigger adaptation.
    learning_rate : float
    """

    def __init__(
        self,
        base_q_table: np.ndarray,
        drift_threshold: float = 0.03,
        learning_rate: float = config.LEARNING_RATE,
        recalibration_cost: float = config.RECALIBRATION_COST,
        fine_tuning_cost: float = config.FINE_TUNING_COST,
    ) -> None:
        self.q_table = np.array(base_q_table, dtype=float).copy()
        self.drift_threshold = float(drift_threshold)
        self.alpha = float(learning_rate)
        self.recal_cost = float(recalibration_cost)
        self.fine_cost = float(fine_tuning_cost)

        # Base decoders
        self.rl_decoder = RLDecoder(self.q_table, name="Adaptive RL Base")
        self._current_mwpm_decoder = None

        # Counters for computational overhead analysis
        self.count_direct_decode = 0
        self.count_recalibrate = 0
        self.count_fine_tune = 0
        self.total_overhead_cost = 0.0

        self.action_history: list[int] = []

    def reset_stats(self) -> None:
        """Reset overhead counters."""
        self.count_direct_decode = 0
        self.count_recalibrate = 0
        self.count_fine_tune = 0
        self.total_overhead_cost = 0.0
        self.action_history.clear()

    def select_meta_action(
        self,
        compact_features: np.ndarray,
        strategy: str = "selective",
    ) -> int:
        """Select meta-action based on strategy and drift features.

        Strategies:
            - 'selective': Trigger adaptation when drift indicator D exceeds threshold.
            - 'continuous': Always fine-tune (action 2).
            - 'fixed': Always direct decode (action 0).
            - 'mwpm_only': Always recalibrate (action 1).

        Parameters
        ----------
        compact_features : np.ndarray, shape (7,)
            Features from SyndromeFeatureExtractor.
        strategy : str

        Returns
        -------
        meta_action : int (0, 1, or 2)
        """
        if strategy == "fixed":
            return 0
        elif strategy == "continuous":
            return 2
        elif strategy == "mwpm_only":
            return 1
        elif strategy == "selective":
            drift_indicator = compact_features[6]
            short_p = compact_features[5]

            # If drift indicator exceeds threshold, trigger recalibration or fine-tuning
            if drift_indicator > self.drift_threshold:
                # If short-term error rate is high/asymmetric, MWPM recalibration is preferred
                if short_p > 0.05:
                    return 1  # MWPM Recalibration
                else:
                    return 2  # RL Fine-tuning
            return 0  # Direct Decoding
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def execute_step(
        self,
        env_obs: dict[str, Any],
        env: Any,
        strategy: str = "selective",
    ) -> tuple[int, bool, float, int]:
        """Execute one decision step using selected meta-action.

        Parameters
        ----------
        env_obs : dict
            Observation dict with 'current_state', 'history', 'drift_features'.
        env : DynamicDriftQECEnv
        strategy : str

        Returns
        -------
        meta_action : int
        logical_survived : bool
        net_reward : float (logical reward minus update cost penalty)
        decoder_action : int
        """
        curr_state = int(env_obs["current_state"])
        features = env_obs["drift_features"]

        meta_action = self.select_meta_action(features, strategy=strategy)
        self.action_history.append(meta_action)

        cost_penalty = 0.0

        if meta_action == 0:
            # Action 0: Direct Decoding
            self.count_direct_decode += 1
            decoder_action = self.rl_decoder.decode(curr_state)

        elif meta_action == 1:
            # Action 1: MWPM Recalibration
            self.count_recalibrate += 1
            cost_penalty = self.recal_cost

            # Estimate noise parameters from statistical features
            p0_est, p1_est, p2_est = env.feature_extractor.estimate_per_qubit_noise()
            mean_p = (p0_est + p1_est + p2_est) / 3.0

            recal_circuit = build_repetition_code_circuit(
                mean_p, per_qubit_rates=(p0_est, p1_est, p2_est)
            )
            self._current_mwpm_decoder = MWPMDecoder(
                recal_circuit, name="Recalibrated MWPM"
            )

            # MWPM prediction translated to correction action
            synd_vec = np.array(
                [[(curr_state >> 1) & 1, curr_state & 1]], dtype=np.uint8
            )
            obs_pred = int(self._current_mwpm_decoder.decode(synd_vec)[0, 0])
            # Map MWPM observable prediction back to action (flip q2 if obs_pred=1, else no-op/q0/q1)
            decoder_action = 3 if obs_pred == 1 else (1 if curr_state == 2 else 0)

        elif meta_action == 2:
            # Action 2: RL Policy Fine-tuning
            self.count_fine_tune += 1
            cost_penalty = self.fine_cost

            # Greedy choice with small epsilon for exploration during fine-tuning
            decoder_action = self.rl_decoder.decode(curr_state)

        else:
            raise ValueError(f"Invalid meta_action: {meta_action}")

        # Step environment
        _, raw_reward, _, _, info = env.step(decoder_action)
        logical_survived = bool(info["logical_survived"])

        # Update Q-table online if fine-tuning action was selected
        if meta_action == 2:
            td_error = raw_reward - self.q_table[curr_state, decoder_action]
            self.q_table[curr_state, decoder_action] += self.alpha * td_error
            self.rl_decoder.q_table = self.q_table.copy()

        self.total_overhead_cost += cost_penalty
        net_reward = raw_reward - cost_penalty

        return meta_action, logical_survived, net_reward, decoder_action
