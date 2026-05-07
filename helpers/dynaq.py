from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from helpers.env import ACTIONS, SlipperyGridWorld
from helpers.rl import epsilon_greedy


def dyna_q(
    env: SlipperyGridWorld,
    n_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    n_planning_steps: int = 10,
    seed: int = 0,
    env_changes: Optional[dict[int, Callable]] = None,
) -> Tuple[np.ndarray, list[float]]:
    """Dyna-Q: Q-learning with model-based planning.

    After each real environment step, performs n_planning_steps additional
    Q-learning updates using a learned model of past (s, a) -> (r, s') transitions.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.num_states, len(ACTIONS)))

    # model[(s, a)] = (r, s') — deterministic, stores last observed transition
    model: dict[tuple[int, int], tuple[float, int]] = {}
    # ordered list of visited (s, a) pairs for uniform sampling
    visited: list[tuple[int, int]] = []

    episode_rewards: list[float] = []

    for ep in range(n_episodes):
        if env_changes and ep in env_changes:
            env_changes[ep](env)

        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)

            # --- direct RL update ---
            Q[state, action] += alpha * (
                reward + gamma * (0 if done else np.max(Q[next_state])) - Q[state, action]
            )

            # --- model update ---
            if (state, action) not in model:
                visited.append((state, action))
            model[(state, action)] = (reward, next_state)

            # --- planning ---
            if visited:
                indices = rng.integers(len(visited), size=n_planning_steps)
                for idx in indices:
                    ps, pa = visited[idx]
                    pr, pns = model[(ps, pa)]
                    pns_terminal = env.is_terminal_state(pns)
                    Q[ps, pa] += alpha * (
                        pr + gamma * (0 if pns_terminal else np.max(Q[pns])) - Q[ps, pa]
                    )

            state = next_state
            total_reward += float(reward)

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, episode_rewards


def dyna_q_plus(
    env: SlipperyGridWorld,
    n_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    n_planning_steps: int = 10,
    kappa: float = 0.001,
    seed: int = 0,
    env_changes: Optional[dict[int, Callable]] = None,
) -> Tuple[np.ndarray, list[float]]:
    """Dyna-Q+: Dyna-Q with an exploration bonus κ√τ(s,a) added to planning rewards.

    τ(s,a) counts the number of real steps since (s,a) was last tried.
    The bonus encourages re-visiting stale state-action pairs, which is especially
    useful in non-stationary / changing environments where the model may be outdated.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.num_states, len(ACTIONS)))

    model: dict[tuple[int, int], tuple[float, int]] = {}
    visited: list[tuple[int, int]] = []

    # τ[s, a] = steps since (s, a) was last tried in the real environment
    tau = np.zeros((env.num_states, len(ACTIONS)))

    episode_rewards: list[float] = []

    for ep in range(n_episodes):
        if env_changes and ep in env_changes:
            env_changes[ep](env)

        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)

            # --- direct RL update ---
            Q[state, action] += alpha * (
                reward + gamma * (0 if done else np.max(Q[next_state])) - Q[state, action]
            )

            # --- model update ---
            if (state, action) not in model:
                visited.append((state, action))
            model[(state, action)] = (reward, next_state)

            # age all pairs, reset the one just tried
            tau += 1
            tau[state, action] = 0

            # --- planning with exploration bonus ---
            if visited:
                indices = rng.integers(len(visited), size=n_planning_steps)
                for idx in indices:
                    ps, pa = visited[idx]
                    pr, pns = model[(ps, pa)]
                    bonus = kappa * np.sqrt(tau[ps, pa])
                    pns_terminal = env.is_terminal_state(pns)
                    Q[ps, pa] += alpha * (
                        pr + bonus + gamma * (0 if pns_terminal else np.max(Q[pns])) - Q[ps, pa]
                    )

            state = next_state
            total_reward += float(reward)

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, episode_rewards
