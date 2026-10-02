"""
config.py
=========
SINGLE SOURCE OF TRUTH for every constant, hyperparameter and seed in the
project. No other module is allowed to hardcode a number that lives here.

Why a config module at all?
    An empirical study is only meaningful if every experiment ran under the
    same conditions. Keeping the numbers in one file means "change once,
    affects everything", and it makes the results reproducible by a reader who
    only has to look at a single file to know exactly what was run.

Quantum vocabulary used below (defined once, referenced everywhere):
    data qubit   : a physical qubit that actually stores the encoded
                   information. Our 3-qubit repetition code uses qubits 0,1,2.
    ancilla qubit: a helper ("scratch") qubit that is entangled with the data
                   qubits and then measured. Measuring an ancilla instead of a
                   data qubit is what lets us learn *something* about the error
                   without collapsing the encoded information. Qubits 3,4.
    syndrome     : the classical bit string produced by measuring the
                   ancillas. It tells you which parity checks were violated,
                   i.e. *where* something looks wrong - never *what* the
                   quantum state is.
    logical error: the code failed. The decoder's correction, combined with
                   the physical error, left the encoded bit flipped.
    LER          : Logical Error Rate = fraction of shots ending in a logical
                   error. Lower is better. This is THE metric of the project.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42
# Several experiments are repeated with different seeds so we can report
# mean +/- spread instead of a single lucky/unlucky run.
NUM_SEEDS: int = 5

# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------
# p = probability that an individual data qubit suffers an X ("bit flip")
# error during one round of the memory experiment. An X error on a qubit in
# state |0> turns it into |1>: it is the quantum analogue of a classical bit
# flip, and it is exactly the error a repetition code is designed to catch.
NOISE_RATES_TRAIN: list[float] = [0.03]
NOISE_RATES_TEST: list[float] = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]

# Convenience alias: the single noise rate the RL agent is trained at.
NOISE_RATE_TRAIN: float = NOISE_RATES_TRAIN[0]

# ---------------------------------------------------------------------------
# Monte-Carlo sampling
# ---------------------------------------------------------------------------
# One "shot" = one full run of the quantum circuit (prepare -> noise ->
# measure ancillas -> measure data). LER is estimated as a binomial fraction
# over NUM_SHOTS shots, so its statistical uncertainty is ~sqrt(LER/NUM_SHOTS).
NUM_SHOTS: int = 10_000

# The environment pulls shots from Stim in blocks instead of one at a time,
# because one-shot-at-a-time sampling is dominated by Python call overhead.
SAMPLE_BATCH_SIZE: int = 4_096

# ---------------------------------------------------------------------------
# Reinforcement learning hyperparameters
# ---------------------------------------------------------------------------
NUM_EPISODES: int = 50_000
LEARNING_RATE: float = 0.1        # alpha in the Q-update
DISCOUNT_FACTOR: float = 0.99     # gamma; see note below
EPSILON_START: float = 1.0        # fully random at the start
EPSILON_END: float = 0.05         # never stop exploring completely
EPSILON_DECAY: float = 0.9999     # multiplicative decay, applied per episode

# NOTE on DISCOUNT_FACTOR: every episode in this environment is exactly one
# step long (see a syndrome -> apply a correction -> episode over). The next
# state is terminal, so the bootstrapped term gamma * max_a' Q(s', a') is
# multiplied by (1 - done) = 0 and vanishes. Gamma is kept in the config
# because the update rule in training/qlearner.py is written in the general
# form, and Phase 2 (multi-round decoding) will have episodes longer than one
# step, where gamma starts to matter.

LOG_EVERY: int = 1_000            # print a training line every N episodes

# ---------------------------------------------------------------------------
# Code / circuit geometry
# ---------------------------------------------------------------------------
NUM_DATA_QUBITS: int = 3
NUM_ANCILLA_QUBITS: int = 2

# Qubit index labels used by circuits/repetition_code.py.
DATA_QUBITS: tuple[int, ...] = (0, 1, 2)
ANCILLA_QUBITS: tuple[int, ...] = (3, 4)

# The logical observable of the circuit is the measurement of ONE data qubit
# (the last one measured, referenced as rec[-1] in the Stim circuit). A
# decoder's job is to predict whether that measurement came out flipped.
LOGICAL_OBSERVABLE_QUBIT: int = 2
# Stim allows several observables per circuit; we declare exactly one, index 0.
LOGICAL_OBSERVABLE_INDEX: int = 0

# ---------------------------------------------------------------------------
# RL state / action spaces
# ---------------------------------------------------------------------------
# STATE = syndrome only. Two ancillas -> two syndrome bits -> 4 states.
# Encoding: state = 2 * s0 + s1, where s0 is the first detector (parity of
# data qubits 0 and 1) and s1 is the second detector (parity of qubits 1,2).
#   state 0 = [0,0]   state 1 = [0,1]   state 2 = [1,0]   state 3 = [1,1]
NUM_STATES: int = 4

# ACTION = which data qubits to flip back (or do nothing).
# 3 data qubits -> 2^3 = 8 possible correction patterns.
NUM_ACTIONS: int = 8

# action index -> correction bit-string applied to data qubits (0,1,2)
ACTION_CORRECTIONS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),   # "no correction"
    1: (1, 0, 0),   # flip data qubit 0
    2: (0, 1, 0),   # flip data qubit 1
    3: (0, 0, 1),   # flip data qubit 2
    4: (1, 1, 0),   # flip qubits 0 and 1
    5: (1, 0, 1),   # flip qubits 0 and 2
    6: (0, 1, 1),   # flip qubits 1 and 2 (two-qubit flip, optimal under biased noise)
    7: (1, 1, 1),   # flip qubits 0, 1 and 2
}

ACTION_LABELS: tuple[str, ...] = (
    "no-op",
    "flip q0",
    "flip q1",
    "flip q2",
    "flip q0,q1",
    "flip q0,q2",
    "flip q1,q2",
    "flip q0,q1,q2",
)
STATE_LABELS: tuple[str, ...] = ("[0,0]", "[0,1]", "[1,0]", "[1,1]")

# The textbook minimum-weight policy for the 3-qubit repetition code under uniform noise.
# Index = state, value = action.
OPTIMAL_POLICY: tuple[int, ...] = (0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Environment self-test (rule 9: this MUST pass before any RL training)
# ---------------------------------------------------------------------------
VERIFY_EPISODES: int = 10_000
VERIFY_MIN_SUCCESS_RATE: float = 0.95   # below this -> the env has a bug

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
NUM_BOOTSTRAP: int = 100          # bootstrap resamples for LER error bars
CONFIDENCE_LEVEL: float = 0.95    # for the bootstrap percentile interval

# ---------------------------------------------------------------------------
# Bonus analysis: biased (asymmetric) noise
# ---------------------------------------------------------------------------
# See experiments/exp_drift.py. Under *uniform* noise the optimal decoding
# rule for this code is mathematically independent of p, so "drift" in p alone
# cannot hurt any decoder. Changing the SHAPE of the noise can. BIAS_PROFILE
# gives the per-data-qubit multipliers used for that extra study.
# The multipliers average to 1.0, so sweeping the bias keeps the MEAN physical
# error rate fixed and changes only its SHAPE: qubit 0 becomes nearly perfect
# while qubit 2 (the logical readout qubit) becomes the noisiest. That is the
# regime where "one flip on q0" stops being the most likely explanation of
# syndrome [1,0], so the optimal decoding rule genuinely changes.
BIAS_PROFILE: tuple[float, float, float] = (0.01, 1.24, 1.75)
BIAS_STRENGTHS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
# Bias-drift LERs are small, so this study uses more shots to resolve them.
BIAS_NUM_SHOTS: int = 100_000

# ---------------------------------------------------------------------------
# Adaptive RL & Feature Space (Phase 2 / Complete Objectives)
# ---------------------------------------------------------------------------
# Fixed-length syndrome history window size (number of past rounds stored)
HISTORY_WINDOW_SIZE: int = 4

# Statistical sliding window size for compact drift features
STATISTICAL_WINDOW_SIZE: int = 200
SHORT_WINDOW_SIZE: int = 40

# Adaptive Controller Action Space:
#   0: Direct Decoding (no update cost)
#   1: MWPM Recalibration (rebuild matching graph from statistical features)
#   2: RL Policy Fine-tuning (update Q-table entry online)
NUM_META_ACTIONS: int = 3
META_ACTION_LABELS: tuple[str, ...] = (
    "Direct Decode",
    "MWPM Recalibrate",
    "RL Fine-Tune",
)

# Operational overhead cost penalties subtracted from reward during meta-learning
RECALIBRATION_COST: float = 0.05
FINE_TUNING_COST: float = 0.02

# Dynamic drift timeline parameters for continuous adaptation experiments
DYNAMIC_DRIFT_TIMESTEPS: int = 15_000
DRIFT_CHANGE_INTERVAL: int = 3_000

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
RESULTS_DIR: str = "results"

# Canonical file names, so every module refers to the same artefact.
BASELINE_CSV: str = "baseline_results.csv"
BASELINE_PLOT: str = "baseline_mwpm.png"
QTABLE_FILE: str = "q_table.npy"
TRAINING_CURVE_PLOT: str = "training_curve.png"
RL_VS_MWPM_PLOT: str = "rl_vs_mwpm.png"
RL_VS_MWPM_CSV: str = "rl_vs_mwpm.csv"
DRIFT_CSV: str = "drift_results.csv"
DRIFT_PLOT: str = "drift_comparison.png"
BIAS_CSV: str = "drift_bias_results.csv"
BIAS_PLOT: str = "drift_bias_comparison.png"
ADAPTIVE_CSV: str = "adaptive_drift_results.csv"
ADAPTIVE_PLOT: str = "adaptive_drift_comparison.png"
ADAPTIVE_METADATA: str = "adaptive_metadata.json"
ADAPTIVE_MULTISEED_CSV: str = "adaptive_drift_multiseed.csv"
ADAPTIVE_MULTISEED_METADATA: str = "adaptive_drift_multiseed.json"
LOG_FILE: str = "run.log"


