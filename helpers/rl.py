from __future__ import annotations

from typing import Tuple

import numpy as np

from helpers.env import ACTIONS, SlipperyGridWorld


def epsilon_greedy(
    Q: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Select an action using epsilon-greedy exploration."""
    if rng.random() < epsilon:
        return int(rng.integers(len(ACTIONS)))
    return int(np.argmax(Q[state]))


def greedy_policy_from_Q(Q: np.ndarray) -> np.ndarray:
    """Return deterministic policy pi(s)=argmax_a Q(s,a)."""
    return np.argmax(Q, axis=1).astype(float)


def q_learning(
    env: SlipperyGridWorld,
    n_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    seed: int = 0,
) -> Tuple[np.ndarray, list[float]]:
    """Q-learning with per-episode reward tracking."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.num_states, len(ACTIONS)))
    episode_rewards: list[float] = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            Q[state,
              action] += alpha * (reward + gamma *
                                  (0 if done else np.max(Q[next_state])) -
                                  Q[state, action])
            state = next_state
            total_reward += float(reward)

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, episode_rewards


def value_iteration(
    env: SlipperyGridWorld,
    max_iterations: int,
    gamma: float,
    threshold: float,
) -> Tuple[np.ndarray, int]:
    """Value iteration. Returns V and the number of iterations until convergence."""
    V = np.zeros(env.num_states)

    for i in range(max_iterations):
        delta = 0
        V_new = np.zeros(env.num_states)
        for state in range(env.num_states):
            if env.is_terminal_state(state):
                continue
            v_a = []
            for a in ACTIONS:
                val = sum(
                    prob * (env.reward(state, a, next_state) + gamma * V[next_state])
                    for prob, next_state in env.get_transition_distribution(state, a)
                )
                v_a.append(val)
            V_new[state] = max(v_a)
            delta = max(delta, abs(V_new[state] - V[state]))
        V = V_new
        if delta < threshold:
            return V, i + 1

    return V, max_iterations


def sarsa(
    env: SlipperyGridWorld,
    n_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    seed: int = 0,
) -> Tuple[np.ndarray, list[float]]:
    """SARSA with per-episode reward tracking."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.num_states, len(ACTIONS)))
    episode_rewards: list[float] = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        action = epsilon_greedy(Q, state, epsilon, rng)
        total_reward = 0.0

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon, rng)
            Q[state,
              action] += alpha * (reward + gamma *
                                  (0 if done else Q[next_state, next_action]) -
                                  Q[state, action])
            state, action = next_state, next_action
            total_reward += float(reward)

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, episode_rewards
