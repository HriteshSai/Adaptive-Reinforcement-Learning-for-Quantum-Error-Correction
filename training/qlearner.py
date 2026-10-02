"""
training/qlearner.py
====================
Tabular Q-learning for the 4-state / 8-action QEC decoding problem.

-----------------------------------------------------------------------
WHY TABULAR IS THE RIGHT CHOICE HERE (project rule 3)
-----------------------------------------------------------------------
The state space has 4 elements and the action space has 8, so the entire
value function is 32 numbers. A neural network would add approximation error,
optimiser noise and 500 lines of debugging for exactly zero representational
benefit. A table is also *inspectable*: you can print it, compare it against
the analytically optimal policy, and know with certainty whether the agent
learned the right thing. For an empirical study, that auditability is the
whole point.

-----------------------------------------------------------------------
THE UPDATE RULE
-----------------------------------------------------------------------
The general Q-learning update is

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * (1 - done) * max_a' Q(s',a')
                                 - Q(s,a) ]

Every episode here terminates after one step, so done = 1 and the
bootstrapped term disappears:

    Q(s,a) <- Q(s,a) + alpha * [ r - Q(s,a) ]

which is an exponentially-weighted running estimate of the expected reward of
taking action a in syndrome s. The fixed point is therefore

    Q*(s,a) = E[reward | s, a] = P(success | s, a) - P(failure | s, a)
            = 2 * P(success | s, a) - 1

At p = 0.03 that predicts, for syndrome [1,0]:
    Q(s, flip q0) = 2 * 0.97 - 1 = 0.94   (the correct correction)
    Q(s, anything else)           = -1.00 (guaranteed failure)
A trained table that does not look like this has not converged - the
`convergence_report()` method checks exactly this against the environment.

-----------------------------------------------------------------------
DEBUGGING NOTES (things that actually bite in tabular RL)
-----------------------------------------------------------------------
* Rare states. At p = 0.03 the syndrome is [0,0] about 91% of the time, so
  each non-trivial syndrome is visited only ~3% of 50,000 = ~1,450 times.
  That is plenty for a bandit, but the learner still logs per-state visit
  counts so you can prove it rather than assume it.
* Epsilon floor. EPSILON_END = 0.05 keeps 5% exploration forever. That caps
  the *training* reward curve below the optimum (you keep making deliberate
  mistakes), which is expected and is why evaluation always uses the greedy
  policy. If your training curve plateaus slightly below the analytic
  optimum, that is the epsilon floor, not a bug.
* Ties. Numpy's argmax breaks ties by choosing the lowest index, so an
  all-zeros row would silently look like "action 0 is best". We initialise
  the table at zero and log visit counts so an untouched row is visible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config
from environment.qec_env import QECDecoderEnv


def moving_average(values, window: int = 500) -> np.ndarray:
    """Smooth a noisy reward curve with a simple moving average.

    Per-episode rewards are +1/-1, so a raw plot is an unreadable band. The
    moving average is what makes the learning trend visible.

    Parameters
    ----------
    values : sequence of float
        Per-episode rewards.
    window : int
        Window length in episodes.

    Returns
    -------
    np.ndarray, shape (len(values) - window + 1,)
        Smoothed curve; empty array if the input is shorter than the window.
    """
    values = np.asarray(values, dtype=float)
    window = int(max(1, min(window, len(values))))
    if len(values) < window:
        return np.array([])
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


class TabularQLearner:
    """Epsilon-greedy tabular Q-learning agent for QECDecoderEnv.

    Attributes
    ----------
    env : QECDecoderEnv
        The environment to learn in.
    q_table : np.ndarray, shape (NUM_STATES, NUM_ACTIONS)
        The learned action-value table.
    epsilon : float
        Current exploration probability (decays during training).
    state_visits : np.ndarray, shape (NUM_STATES, NUM_ACTIONS)
        How often each (state, action) pair was updated - the first thing to
        look at when an agent "won't converge".
    """

    def __init__(self, env: QECDecoderEnv, cfg: Any = config, seed: int = config.SEED) -> None:
        """Initialise the Q-table and copy hyperparameters out of the config.

        Parameters
        ----------
        env : QECDecoderEnv
            Environment providing (syndrome -> reward) samples.
        cfg : module or object
            Anything exposing LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_START,
            EPSILON_END, EPSILON_DECAY, NUM_STATES, NUM_ACTIONS, LOG_EVERY.
            Defaults to the project config module. Passing an object instead
            lets an ablation study override one number without editing files.
        seed : int
            Seed for the agent's own RNG (exploration coin flips). Kept
            separate from the environment's seed so that "same noise stream,
            different exploration" ablations are possible.
        """
        self.env = env
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))

        self.num_states = int(cfg.NUM_STATES)
        self.num_actions = int(cfg.NUM_ACTIONS)

        # Optimistic vs zero initialisation: zeros are fine here because the
        # reward range is [-1, +1], so any successful action immediately looks
        # better than an untried one (0), and any failing action looks worse.
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)

        self.alpha = float(cfg.LEARNING_RATE)
        self.gamma = float(cfg.DISCOUNT_FACTOR)
        self.epsilon = float(cfg.EPSILON_START)
        self.epsilon_end = float(cfg.EPSILON_END)
        self.epsilon_decay = float(cfg.EPSILON_DECAY)
        self.log_every = int(cfg.LOG_EVERY)

        self.state_visits = np.zeros((self.num_states, self.num_actions), dtype=int)
        self.reward_history: list[float] = []
        self.epsilon_history: list[float] = []
        self.diagnostics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def select_action(self, state: int, greedy: bool = False) -> int:
        """Choose an action with an epsilon-greedy rule.

        Exploration matters even in a 4×8 problem: the agent has to try
        "flip q1" on syndrome [1,1] at least a few times before it can know
        that it beats the alternatives. Epsilon starts at 1.0 (pure
        exploration) and decays towards EPSILON_END.

        Parameters
        ----------
        state : int
            Syndrome index in {0,1,2,3}.
        greedy : bool
            If True, ignore epsilon and always take argmax (used for
            evaluation).

        Returns
        -------
        int
            The chosen action index.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Apply one Q-learning update and return the TD error.

        Parameters
        ----------
        state : int
        action : int
        reward : float
            +1 or -1 from the environment.
        next_state : int
            Ignored in practice because `done` is always True here; kept so
            the update is written in its general form for Phase 2.
        done : bool
            Whether the episode terminated (always True in Phase 1).

        Returns
        -------
        float
            The temporal-difference error. Its magnitude shrinking towards a
            steady value is the practical signal that learning converged.
        """
        bootstrap = 0.0 if done else self.gamma * float(np.max(self.q_table[next_state]))
        td_error = (reward + bootstrap) - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        self.state_visits[state, action] += 1
        return float(td_error)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, num_episodes: int | None = None, verbose: bool = True):
        """Run the epsilon-greedy tabular Q-learning loop.

        Loop body: reset the environment (which runs the quantum circuit once
        and hands back a syndrome), pick a correction, receive +1/-1 depending
        on whether the encoded bit survived, update one table entry, decay
        epsilon.

        Parameters
        ----------
        num_episodes : int
            Number of one-step episodes. Defaults to config.NUM_EPISODES.
        verbose : bool
            Print a progress line every config.LOG_EVERY episodes.

        Returns
        -------
        q_table : np.ndarray, shape (NUM_STATES, NUM_ACTIONS)
            The trained table (also available as `self.q_table`).
        history : list of float
            Per-episode rewards, in order. Extra diagnostics (epsilon trace,
            visit counts, TD errors) are stored in `self.diagnostics`.
        """
        num_episodes = int(config.NUM_EPISODES if num_episodes is None else num_episodes)
        rewards: list[float] = []
        epsilons: list[float] = []
        td_errors: list[float] = []

        if verbose:
            print(f"\n[TRAINING] {num_episodes} episodes at p = {self.env.noise_rate}")
            print(
                f"  alpha={self.alpha}  gamma={self.gamma}  "
                f"epsilon {self.epsilon:.2f} -> {self.epsilon_end:.2f} "
                f"(decay {self.epsilon_decay})"
            )
            header = (
                f"{'episode':>9} | {'epsilon':>8} | {'mean R (last 1k)':>17} | "
                f"{'greedy policy':>14}"
            )
            print(header)
            print("-" * len(header))

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _info = self.env.step(action)
            done = terminated or truncated

            td_errors.append(self.update(state, action, reward, next_state, done))
            rewards.append(float(reward))
            epsilons.append(self.epsilon)

            # Epsilon decays once per episode (not per step - identical here,
            # but the distinction matters as soon as episodes get longer).
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if verbose and (episode + 1) % self.log_every == 0:
                window = rewards[-self.log_every:]
                policy = "".join(str(a) for a in self.get_policy())
                print(
                    f"{episode + 1:>9} | {self.epsilon:>8.4f} | "
                    f"{np.mean(window):>17.4f} | {policy:>14}"
                )

        self.reward_history = rewards
        self.epsilon_history = epsilons
        self.diagnostics = {
            "final_epsilon": self.epsilon,
            "state_action_visits": self.state_visits.copy(),
            "mean_abs_td_error_last_1k": float(np.mean(np.abs(td_errors[-1000:]))),
            "mean_reward_last_1k": float(np.mean(rewards[-1000:])),
            "num_episodes": num_episodes,
        }

        if verbose:
            print(f"[TRAINING] done. final epsilon = {self.epsilon:.4f}")

        return self.q_table, rewards

    # ------------------------------------------------------------------
    # Evaluation / inspection
    # ------------------------------------------------------------------
    def get_policy(self) -> tuple[int, ...]:
        """Greedy policy implied by the current table.

        Returns
        -------
        tuple of 4 ints
            policy[state] = action.
        """
        return tuple(int(np.argmax(self.q_table[s])) for s in range(self.num_states))

    def evaluate(self, num_episodes: int | None = None) -> float:
        """Run the greedy policy in the environment and return its success rate.

        No exploration, no learning - this is the number to quote as "how good
        is the trained agent" in the environment's own reward terms.

        Parameters
        ----------
        num_episodes : int

        Returns
        -------
        float
            Fraction of episodes with reward +1.
        """
        num_episodes = int(config.VERIFY_EPISODES if num_episodes is None else num_episodes)
        successes = 0
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            action = self.select_action(state, greedy=True)
            _, reward, _, _, _ = self.env.step(action)
            successes += int(reward > 0)
        return successes / float(num_episodes)

    def convergence_report(self) -> dict[str, Any]:
        """Compare the learned table against the analytic fixed point.

        For each state, the analytically expected Q-value of the optimal
        action is 2 * P(success | s, optimal action) - 1. This method returns
        the learned values, the reference values and the largest deviation, so
        "did it converge?" becomes a number instead of a vibe.

        Returns
        -------
        dict
            Keys: 'policy', 'optimal_policy', 'policy_matches',
            'max_abs_error_optimal_actions', 'per_state', 'visits'.
        """
        p = self.env.noise_rate
        # P(the minimum-weight guess is exactly right | syndrome):
        #   [0,0]: the alternative is all-three-flipped   -> (1-p)^3 vs p^3
        #   others: one flip vs the complementary two     -> (1-p) vs p
        cond_success = {
            0: (1 - p) ** 3 / ((1 - p) ** 3 + p**3),
            1: 1 - p,
            2: 1 - p,
            3: 1 - p,
        }

        per_state = []
        max_err = 0.0
        for s in range(self.num_states):
            a_opt = int(config.OPTIMAL_POLICY[s])
            expected_q = 2 * cond_success[s] - 1
            learned_q = float(self.q_table[s, a_opt])
            err = abs(learned_q - expected_q)
            max_err = max(max_err, err)
            per_state.append(
                {
                    "state": s,
                    "syndrome": config.STATE_LABELS[s],
                    "optimal_action": a_opt,
                    "learned_q": learned_q,
                    "expected_q": expected_q,
                    "abs_error": err,
                    "visits": int(self.state_visits[s].sum()),
                }
            )

        return {
            "policy": self.get_policy(),
            "optimal_policy": tuple(config.OPTIMAL_POLICY),
            "policy_matches": self.get_policy() == tuple(config.OPTIMAL_POLICY),
            "max_abs_error_optimal_actions": max_err,
            "per_state": per_state,
            "visits": self.state_visits.copy(),
        }

    def print_q_table(self) -> None:
        """Pretty-print the Q-table, the learned policy and the optimal policy.

        The printed table is the single most useful debugging artefact in the
        whole project: each row is a syndrome, each column a correction, and
        the starred entry is what the agent would do. Rows should show one
        clearly positive value and three values near -1.

        Returns
        -------
        None
        """
        col_width = 16
        total_width = 18 + 2 + self.num_actions * col_width
        print("\n" + "=" * total_width)
        print("Q-TABLE".center(total_width))
        print("=" * total_width)

        header = f"{'state (syndrome)':>18} |" + "".join(
            f"{label:>{col_width}}" for label in config.ACTION_LABELS
        )
        print(header)
        print("-" * len(header))

        policy = self.get_policy()
        for s in range(self.num_states):
            row = f"{f'{s}  {config.STATE_LABELS[s]}':>18} |"
            for a in range(self.num_actions):
                star = "*" if a == policy[s] else " "
                row += f"{self.q_table[s, a]:>{col_width - 1}.4f}{star}"
            print(row)

        print("-" * len(header))
        print("visits per (state, action):")
        for s in range(self.num_states):
            counts = "".join(f"{self.state_visits[s, a]:>{col_width}}" for a in range(self.num_actions))
            print(f"{f'{s}  {config.STATE_LABELS[s]}':>18} |{counts}")

        print("\nlearned policy vs known-optimal minimum-weight policy")
        print(f"  {'state':>6} {'syndrome':>10} {'learned':>14} {'optimal':>14}   match")
        all_match = True
        for s in range(self.num_states):
            learned = config.ACTION_LABELS[policy[s]]
            optimal = config.ACTION_LABELS[config.OPTIMAL_POLICY[s]]
            match = policy[s] == config.OPTIMAL_POLICY[s]
            all_match &= match
            print(
                f"  {s:>6} {config.STATE_LABELS[s]:>10} {learned:>14} {optimal:>14}"
                f"   {'YES' if match else 'NO'}"
            )
        print(f"\n  policies identical: {'YES' if all_match else 'NO'}")

        report = self.convergence_report()
        print(
            f"  max |Q_learned - Q_expected| on optimal actions: "
            f"{report['max_abs_error_optimal_actions']:.4f}"
        )
        print("=" * total_width + "\n")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_q_table(self, filepath: str) -> str:
        """Save the Q-table to a .npy file.

        Parameters
        ----------
        filepath : str
            Destination path, e.g. "results/q_table.npy".

        Returns
        -------
        str
            The path written.
        """
        np.save(filepath, self.q_table)
        return filepath

    def load_q_table(self, filepath: str) -> np.ndarray:
        """Load a Q-table from a .npy file into this learner.

        Parameters
        ----------
        filepath : str

        Returns
        -------
        np.ndarray, shape (NUM_STATES, NUM_ACTIONS)
            The loaded table (also assigned to `self.q_table`).

        Raises
        ------
        ValueError
            If the stored array has the wrong shape.
        """
        table = np.load(filepath)
        expected = (self.num_states, self.num_actions)
        if table.shape != expected:
            raise ValueError(f"expected Q-table of shape {expected}, got {table.shape}")
        self.q_table = table.astype(float)
        return self.q_table
