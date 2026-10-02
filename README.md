# Quantum RL Decoder — Phase 1

**Can a reinforcement learning agent learn to decode quantum errors, and how badly does it break when the hardware noise drifts away from what it trained on?**

This project builds a 3-qubit bit-flip repetition code in [Stim](https://github.com/quantumlib/Stim), trains a tabular Q-learning agent to correct errors using nothing but the syndrome measurement and a ±1 "did the logical bit survive?" reward, and then benchmarks it against the standard classical decoder (minimum-weight perfect matching, via PyMatching) as the noise model drifts. It is an **empirical study**, not a new algorithm: the deliverable is a measurement of *where* and *how much* performance degrades under noise-model drift, with the analytically-known optimum as a reference at every step.

Everything runs in under 10 seconds on a laptop.

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py

# Launch the Interactive Web Dashboard & QEC Arena
python run_dashboard.py
```

Outputs (tables, plots, the trained Q-table, a log) land in `results/`.

```bash
python main.py --experiment baseline   # just the MWPM sweep
python main.py --experiment rl         # just the RL training run
python main.py --experiment drift      # just the drift comparison
python main.py --experiment bias       # just the bonus noise-shape study
python main.py --quick                 # tiny run, checks the pipeline works
```

---

## The physics, in the smallest number of words

You have zero quantum background, so here is the whole thing.

* A **qubit** can suffer an **X error** (a bit flip): `|0>` becomes `|1>`. That is the only error type in this project, which is why a classical mental model works throughout.
* The **3-qubit repetition code** stores one logical bit in three physical qubits: `|0>_L = |000>`, `|1>_L = |111>`. If one qubit flips, the other two out-vote it. Two simultaneous flips defeat it — that failure mode is what we measure.
* You cannot just look at the qubits; measuring them destroys the encoded state. Instead you measure **parity checks**: "is qubit 0 the same as qubit 1?" and "is qubit 1 the same as qubit 2?". These are done with **CNOT** gates onto two spare **ancilla** qubits, which are then measured. A CNOT copies parity information without revealing any individual qubit's value.
* The two parity outcomes are the **syndrome**, e.g. `[1,0]`. With hidden error bits `(e0,e1,e2)`, the syndrome is exactly `(e0 XOR e1, e1 XOR e2)`. **This is the decoder's entire input.**
* A **decoder** maps syndrome → correction. It fails when the encoded bit ends up wrong anyway; the frequency of that is the **logical error rate (LER)**, the metric of this project.
* The syndrome cannot distinguish `000` from `111` — both give `[0,0]`. That ambiguity is the irreducible reason decoding is a guessing game, and it is why the agent gets a reward signal rather than a label.

The whole decision problem is 4 syndromes × 4 corrections. The known-optimal (minimum-weight) rule is:

| syndrome | most likely cause | optimal action |
|---|---|---|
| `[0,0]` | nothing happened | no correction |
| `[1,0]` | q0 flipped | flip q0 |
| `[1,1]` | q1 flipped (it is in both checks) | flip q1 |
| `[0,1]` | q2 flipped | flip q2 |

The agent is never told this table. The point of Experiment 2 is that it finds it anyway.

---

## The RL formulation

| | |
|---|---|
| **State** | The syndrome, encoded as `2*s0 + s1` → `Discrete(4)`. Nothing else. The agent never sees the error pattern or the qubit state. |
| **Action** | `Discrete(4)`: no correction, flip q0, flip q1, flip q2. |
| **Reward** | `+1` if the logical bit survived the round, `-1` if it was lost. No fidelity, no shaping, no partial credit. |
| **Episode** | Exactly one step. This is a contextual bandit, which is why a 4×4 table is sufficient and no neural network is needed. |

**How survival is decided.** With hidden error `e` and correction `c`, the residual is `r = e XOR c`. `r = 000` means the state was restored (+1). `r = 111` is a **logical X** — undetectable, the encoded bit is lost (−1). Anything else means the correction pushed the state out of the code space, also a failed round (−1). So in practice: reward is +1 exactly when the correction matched the error.

There is a deliberate design choice here, and it is worth a paragraph in your write-up. A looser convention would score a weight-1 residual as a success (a majority vote over the final measurements would still return the right bit). We reject it because under that rule "do nothing" and "flip q0" earn *identical* expected reward on syndrome `[1,0]`, so several policies tie for optimal and you can never distinguish a converged agent from a lazy one. The strict rule also makes the environment's success rate numerically equal to `1 − LER` of the MWPM metric, so the reward and the published metric measure the same thing. See the long docstring at the top of `environment/qec_env.py`.

---

## The experiments

### Experiment 1 — MWPM baseline (`experiments/exp_baseline.py`)
Sweeps `p = 0.01 … 0.15`, samples 10,000 shots at each rate, decodes with PyMatching, and reports LER with bootstrap 95% intervals.

**This is also the correctness check for the whole simulation stack.** The measured curve must track the analytic `3p² − 2p³`, which is the exact LER of an optimal decoder on a distance-3 code. If it does not, the circuit or the sampler is broken and nothing downstream is trustworthy.

→ `results/baseline_results.csv`, `results/baseline_mwpm.png`

### Experiment 2 — Train the agent (`experiments/exp_rl_train.py`)
1. **Environment self-test runs first and training aborts if it fails.** The hardcoded minimum-weight policy is driven through the environment for 10,000 episodes; its success rate must match the analytic `(1−p)³ + 3p(1−p)² = 99.74%` at `p = 0.03`. If a known-good decoder scores badly, the bug is in the environment, not the agent — and you have just saved yourself a week of tuning hyperparameters that were never the problem.
2. Trains `NUM_SEEDS = 5` independent agents for 50,000 episodes each at `p = 0.03`.
3. Prints the Q-table, the visit counts and the learned-vs-optimal policy comparison.
4. Evaluates the trained agent against MWPM at every test noise rate, on identical shots.

→ `results/q_table.npy`, `results/training_curve.png`, `results/rl_vs_mwpm.png`, `results/rl_vs_mwpm.csv`, `results/rl_training_metadata.json`

### Experiment 3 — Noise drift (`experiments/exp_drift.py`)
Three decoders on identical shots:

* **Fixed RL** — Q-table trained once at `p = 0.03`, never updated.
* **Fixed MWPM** — matching weights calibrated once at `p = 0.03`, never updated.
* **Oracle MWPM** — weights rebuilt at the true test noise. The unattainable upper bound.

**Part A** drifts the *magnitude* of uniform noise (`p = 0.01 … 0.15`).
**Part B** (bonus) drifts the *shape* of the noise: per-qubit rates become asymmetric while the **average rate stays at 0.03**, so magnitude is held fixed and only the structure changes. A fourth decoder, an agent retrained under the drifted noise, is added here.

→ `results/drift_results.csv`, `results/drift_comparison.png`, `results/drift_bias_results.csv`, `results/drift_bias_comparison.png`

---

## What you should see (actual results from `python main.py`)

**The agent learns the optimal decoder, on every seed.**

```
  state (syndrome) |        no-op      flip q0      flip q1      flip q2
          0  [0,0] |      1.0000*     -1.0000      -1.0000      -1.0000
          1  [0,1] |     -0.9985      -0.9996      -0.9999       0.9938*
          2  [1,0] |     -0.9997       0.9969*     -0.9997      -0.9978
          3  [1,1] |     -0.9998      -0.9985       0.8975*     -0.9999

  policies identical: YES        5/5 seeds recovered the exact minimum-weight policy
```

Those numbers are not arbitrary: the fixed point of the update is `Q*(s,a) = 2·P(success | s,a) − 1`, so the winning entries should sit near `2(1−p) − 1 = 0.94` and the losers at `−1`. `convergence_report()` checks exactly that, and `print_q_table()` prints the deviation.

**RL and MWPM produce identical LER at every noise rate** — a difference of exactly `0.000000` across the sweep. The agent, given no physics whatsoever, reconstructed the same lookup table that matching derives from the error model. On a code this small, matching a near-optimal baseline *is* the ceiling.

**Part A is a null result, and that is the interesting part.** All three curves coincide exactly. This is not a bug and not noise — it is provable. Syndrome `[1,0]` is explained either by one flip on q0 (probability `p(1−p)²`) or two flips on q1,q2 (`p²(1−p)`); the ratio is `(1−p)/p > 1` for **every** `p < 0.5`. Changing `p` rescales all matching weights monotonically and never reorders the hypotheses, so a decoder calibrated at `p = 0.03` is still exactly optimal at `p = 0.15`. **Magnitude drift is free.** Report this as a finding with its proof, not as a failed experiment.

**Part B is where drift actually bites.** Hold the mean error rate at 0.03 and make the noise asymmetric (q0 nearly perfect, q2 the noisiest). At full bias, `p1·p2` overtakes `p0`, so syndrome `[1,0]` is now better explained by "q1 and q2 both flipped". The oracle switches its answer; the frozen decoders do not:

| decoder | LER at full bias | penalty |
|---|---|---|
| Oracle MWPM (bias-aware) | 0.00038 | — |
| Fixed MWPM (stale) | 0.00181 | 4.8× |
| Fixed RL (stale) | 0.00181 | 4.8× |
| **Retrained RL (bias-aware)** | **0.00181** | **4.8×** |

Two findings. First, the stale RL agent and the stale MWPM degrade by *exactly* the same amount — they encode the same stale lookup table, so robustness is a property of the calibration being out of date, not of how the decoder was obtained. Learning from reward buys you no extra robustness for free.

Second, and more interesting: **retraining does not help.** The optimal response to `[1,0]` under biased noise is the weight-2 correction "flip q1 and q2", which is not in the 4-action set. The agent converges correctly to the best action it *can* take, and that action is not good enough. MWPM has no such limit because it predicts the logical flip directly instead of naming a single-qubit correction. **The RL decoder's ceiling under shape drift is set by its action-space design, not by its learning algorithm** — which is a clean, well-isolated result and an obvious Phase 2 experiment (widen the action space to all 8 correction patterns and re-run).

---

## File structure

```
quantum_rl_decoder/
├── README.md
├── requirements.txt
├── config.py                  # ALL constants, hyperparameters, seeds
├── circuits/
│   └── repetition_code.py     # Stim circuit builder (+ glossary of every quantum term)
├── simulation/
│   └── sampler.py             # Monte-Carlo shots: syndromes, observables, (hidden) errors
├── decoders/
│   ├── mwpm_decoder.py        # PyMatching baseline
│   └── rl_decoder.py          # Q-table → greedy policy → predicted logical flip
├── environment/
│   └── qec_env.py             # Gymnasium env + the mandatory self-test
├── training/
│   └── qlearner.py            # Tabular Q-learning, epsilon-greedy, diagnostics
├── experiments/
│   ├── exp_baseline.py        # EXP 1
│   ├── exp_rl_train.py        # EXP 2
│   └── exp_drift.py           # EXP 3 (parts A and B)
├── evaluation/
│   └── metrics.py             # LER math, bootstrap CIs, all plotting
├── utils/
│   └── helpers.py             # seeds, logging, CSV/JSON I/O
├── main.py                    # entry point
└── results/                   # auto-created: plots, tables, q_table.npy, run.log
```

`config.py` is the single source of truth. Every other module imports from it; no experiment hardcodes a number.

---

## Reproducibility

`SEED = 42` throughout. `utils.helpers.set_seed()` seeds `random` and `numpy`. Stim deliberately has **no** global seed (it would break multi-threaded sampling), so every sampler in the project is constructed with an explicit seed derived from `config.SEED`; each noise rate gets its own derived seed so the points on a curve are statistically independent rather than correlated.

---

## Debugging notes (RL-specific, from the failure modes that actually occur here)

* **The environment self-test is your first move, always.** A known-good policy scoring badly means the environment is wrong. The failure message in `exp_rl_train.py` prints an ordered checklist: reward rule → state encoding → detector order → noise rate.
* **The training curve plateaus slightly below the optimum. That is correct.** `EPSILON_END = 0.05` keeps 5% of actions random forever, so training reward is capped below the greedy optimum. Evaluation always uses the greedy policy (`decode()` never explores). If you want the curve to touch the dashed line, lower `EPSILON_END` — but then you are measuring a different thing.
* **Check visit counts before blaming the learning rate.** At `p = 0.03` the syndrome is `[0,0]` about 91% of the time, so each non-trivial syndrome is only seen ~1,450 times in 50,000 episodes. `print_q_table()` prints per-(state, action) visit counts; a row with tiny counts is a data problem, not a hyperparameter problem.
* **Beware silent argmax ties.** An untouched all-zero row makes `np.argmax` return action 0, which looks like a decision but is not one. The visit counts expose this.
* **`convergence_report()` gives you a number, not a vibe:** the max deviation between learned Q-values and their analytic fixed point.

---

## Extending this (Phase 2 candidates, in order of value)

1. **Widen the action space to all 8 correction patterns.** Part B shows this is the binding constraint. Highest-value next experiment, and the code changes are confined to `config.ACTION_CORRECTIONS` plus `NUM_ACTIONS`.
2. **Multiple measurement rounds with noisy ancillas.** Episodes stop being one step, `DISCOUNT_FACTOR` starts to matter, and the state space grows past what a table can hold — the honest motivation for function approximation.
3. **Larger codes** (5-qubit repetition, then the surface code). This is where MWPM's advantage over a lookup table becomes real.
4. **Online adaptation:** let the agent keep learning while the noise drifts continuously, and measure recovery time.

---

## Requirements

```
stim, pymatching, gymnasium, numpy, matplotlib, pandas
```

Python 3.10+ (the code uses `X | None` type-hint syntax). Tested on Python 3.13.
