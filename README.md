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
python main.py --experiment adaptive   # dynamic drift & selective adaptation
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

The decision space consists of 4 syndrome states × 8 physical correction actions ($2^3 = 8$ possible bit-flip patterns: `000` to `111`). Under standard uniform noise, the textbook minimum-weight rule chooses the single-qubit corrections:

| syndrome | most likely cause (uniform noise) | optimal action | correction applied |
|---|---|---|---|
| `[0,0]` | nothing happened | action 0 (`no-op`) | `000` |
| `[1,0]` | q0 flipped | action 1 (`flip q0`) | `100` |
| `[1,1]` | q1 flipped (in both checks) | action 2 (`flip q1`) | `010` |
| `[0,1]` | q2 flipped | action 3 (`flip q2`) | `001` |

Under asymmetric/biased noise, higher-weight patterns (like action 6 = `011`, flipping `q1` and `q2`) can become more probable than single-qubit flips. Providing all 8 actions allows the RL agent to discover both standard minimum-weight and non-standard biased-noise corrections. The agent is never told this table — it discovers the optimal policy purely from syndrome and reward feedback.

---

## The RL formulation

| | |
|---|---|
| **State** | The syndrome, encoded as `2*s0 + s1` → `Discrete(4)`. Nothing else. The agent never sees the error pattern or the qubit state. |
| **Action** | `Discrete(8)`: all $2^3 = 8$ correction patterns (`000` no-op, `100` q0, `010` q1, `001` q2, `110` q0+q1, `101` q0+q2, `011` q1+q2, `111` q0+q1+q2). |
| **Reward** | `+1` if the logical bit survived the round, `-1` if it was lost. No fidelity, no shaping, no partial credit. |
| **Episode** | Exactly one step. This is a contextual bandit; the entire value function is a 4×8 table (32 numbers) where tabular Q-learning is exact, fully inspectable, and immune to neural-network approximation error. |

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

### Experiment 4 — Adaptive RL & Selective Adaptation (`experiments/exp_adaptive.py`)
Evaluates continuous dynamic noise drift across 15,000 timesteps spanning 5 operational drift regimes (baseline uniform $p = 0.03$, abrupt magnitude jump $p = 0.08$, biased shape drift, severe drift $p = 0.12$, baseline recovery):

* **Fixed MWPM** — frozen matching graph (zero online updates; suffers high LER under bias).
* **Continuous RL Fine-Tune** — updates Q-table on every step (100% online update overhead).
* **Selective Adaptation (RL)** — syndrome-statistics-based adaptive controller that tracks short vs. long sliding window features and selectively triggers:
  * *Meta-Action 0 (Direct Decode)*: Zero-cost inference when noise is stationary.
  * *Meta-Action 1 (MWPM Recalibrate)*: Estimates per-qubit noise rates $\hat{p}_0, \hat{p}_1, \hat{p}_2$, rebuilds a PyMatching matching graph, and updates the active policy.
  * *Meta-Action 2 (RL Fine-Tune)*: Performs targeted online Q-learning updates.
* **Oracle MWPM** — dynamic upper-bound baseline rebuilt at each timestep.

**Unified Evaluation Methodology:** All four strategies — including Oracle MWPM — are scored via the identical `env.step` → `info["logical_survived"]` interface. This eliminates measurement discrepancies and guarantees that all reported LER values are strictly apples-to-apples.

→ `results/adaptive_drift_results.csv`, `results/adaptive_drift_comparison.png`, `results/adaptive_metadata.json`, `results/adaptive_drift_multiseed.csv`

---

## What you should see (actual results from `python main.py`)

**The agent learns the optimal decoder across all 8 actions, on every seed.**

```
  state (syndrome) |        no-op      flip q0      flip q1      flip q2  flip q0,q1  flip q0,q2  flip q1,q2 flip q0,q1,q2
          0  [0,0] |      1.0000*     -1.0000      -1.0000      -1.0000     -1.0000     -1.0000     -1.0000      -1.0000
          1  [0,1] |     -0.9985      -0.9996      -0.9999       0.9938*    -0.9998     -0.9999     -0.9997      -1.0000
          2  [1,0] |     -0.9997       0.9969*     -0.9997      -0.9978     -0.9998     -0.9999     -0.9999      -1.0000
          3  [1,1] |     -0.9998      -0.9985       0.8975*     -0.9999     -0.9998     -0.9997     -0.9998      -1.0000

  policies identical: YES        5/5 seeds recovered the exact minimum-weight policy
```

Those numbers are not arbitrary: the fixed point of the update is `Q*(s,a) = 2·P(success | s,a) − 1`, so the winning entries sit near `2(1−p) − 1 = 0.94` and non-optimal entries sit near `−1`. `convergence_report()` checks exactly that, and `print_q_table()` prints the deviation.

**RL and MWPM produce identical LER at every noise rate** — a difference of exactly `0.000000` across the sweep. The agent, given no physics whatsoever, reconstructed the same lookup table that matching derives from the error model. On a code this small, matching a near-optimal baseline *is* the ceiling.

**Part A is a null result, and that is the interesting part.** All three curves coincide exactly. This is not a bug and not noise — it is provable. Syndrome `[1,0]` is explained either by one flip on q0 (probability `p(1−p)²`) or two flips on q1,q2 (`p²(1−p)`); the ratio is `(1−p)/p > 1` for **every** `p < 0.5`. Changing `p` rescales all matching weights monotonically and never reorders the hypotheses, so a decoder calibrated at `p = 0.03` is still exactly optimal at `p = 0.15`. **Magnitude drift is free.** Report this as a finding with its proof, not as a failed experiment.

**Part B is where drift actually bites.** Hold the mean error rate at 0.03 and make the noise asymmetric (q0 nearly perfect, q2 the noisiest). At full bias, `p1·p2` overtakes `p0`, so syndrome `[1,0]` is now better explained by "q1 and q2 both flipped". The oracle switches its answer; the frozen decoders do not:

| decoder | LER at full bias | penalty |
|---|---|---|
| Oracle MWPM (bias-aware) | 0.00029 | — |
| Fixed MWPM (stale) | 0.00210 | 7.2× |
| Fixed RL (stale) | 0.00210 | 7.2× |
| **Retrained RL (bias-aware)** | **0.00029** | **1.0× (Fully Recovered)** |

Two findings. First, the stale RL agent and the stale MWPM degrade by *exactly* the same amount — they encode the same stale lookup table, so robustness is a property of the calibration being out of date, not of how the decoder was obtained. Learning from reward buys you no extra robustness for free.

Second: **Under noise-shape drift, retrained RL recovered the Oracle MWPM logical-error performance using only logical reward feedback.** By expanding the action space to all 8 correction patterns ($2^3 = 8$), the retrained agent learned action 6 (`011`: flip `q1` and `q2`) for syndrome `[1,0]` under high $q2$ bias, matching Oracle MWPM LER without requiring a pre-characterized error model.

**Adaptive RL & Selective Adaptation (Experiment 4):**
Selective adaptation maintained the observed decoding performance of continuous RL while reducing online RL updates by ~50%–55% across multi-seed runs. By coupling sliding-window drift metrics with PyMatching recalibration and selective RL fine-tuning, the controller achieves rapid post-drift recovery while eliminating redundant updates in stationary regimes.

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
│   ├── rl_decoder.py          # Q-table → greedy policy → predicted logical flip
│   └── adaptive_module.py     # Syndrome-statistics-based adaptive controller
├── environment/
│   ├── qec_env.py             # Gymnasium env + the mandatory self-test
│   └── adaptive_env.py        # Dynamic continuous drift env & feature extractor
├── training/
│   └── qlearner.py            # Tabular Q-learning, epsilon-greedy, diagnostics
├── experiments/
│   ├── exp_baseline.py        # EXP 1 (MWPM baseline sweep)
│   ├── exp_rl_train.py        # EXP 2 (Q-learning validation & multi-seed)
│   ├── exp_drift.py           # EXP 3 (Magnitude & shape drift)
│   └── exp_adaptive.py        # EXP 4 (Dynamic drift & selective adaptation)
├── evaluation/
│   └── metrics.py             # LER math, bootstrap CIs, multi-seed aggregation, plotting
├── utils/
│   └── helpers.py             # seeds, logging, CSV/JSON I/O
├── main.py                    # entry point
├── api.py                     # FastAPI backend for web GUI
└── results/                   # auto-created: plots, tables, q_table.npy, run.log
```

`config.py` is the single source of truth. Every other module imports from it; no experiment hardcodes a number.

---

## Reproducibility

`SEED = 42` throughout. `utils.helpers.set_seed()` seeds `random` and `numpy`. Stim deliberately has **no** global seed (it would break multi-threaded sampling), so every sampler in the project is constructed with an explicit seed derived from `config.SEED`; each noise rate gets its own derived seed so the points on a curve are statistically independent rather than correlated. Multi-seed evaluations in Experiments 2 and 4 evaluate across 5 seeds (`[seed + 1000 * i]`) and report mean ± standard deviation.

---

## Debugging notes (RL-specific, from the failure modes that actually occur here)

* **The environment self-test is your first move, always.** A known-good policy scoring badly means the environment is wrong. The failure message in `exp_rl_train.py` prints an ordered checklist: reward rule → state encoding → detector order → noise rate.
* **The training curve plateaus slightly below the optimum. That is correct.** `EPSILON_END = 0.05` keeps 5% of actions random forever, so training reward is capped below the greedy optimum. Evaluation always uses the greedy policy (`decode()` never explores). If you want the curve to touch the dashed line, lower `EPSILON_END` — but then you are measuring a different thing.
* **Check visit counts before blaming the learning rate.** At `p = 0.03` the syndrome is `[0,0]` about 91% of the time, so each non-trivial syndrome is only seen ~1,450 times in 50,000 episodes. `print_q_table()` prints per-(state, action) visit counts; a row with tiny counts is a data problem, not a hyperparameter problem.
* **Beware silent argmax ties.** An untouched all-zero row makes `np.argmax` return action 0, which looks like a decision but is not one. The visit counts expose this.
* **`convergence_report()` gives you a number, not a vibe:** the max deviation between learned Q-values and their analytic fixed point.

---

## Future directions

1. **Multiple measurement rounds with noisy ancillas.** Episodes stop being one step, `DISCOUNT_FACTOR` starts to matter, and the state space grows past what a table can hold — the honest motivation for function approximation.
2. **Larger codes** (5-qubit repetition, then the surface code). This is where MWPM's advantage over a lookup table becomes real.

---

## Requirements

```
stim, pymatching, gymnasium, numpy, matplotlib, pandas, fastapi, uvicorn
```

Python 3.10+ (the code uses `X | None` type-hint syntax). Tested on Python 3.13.

