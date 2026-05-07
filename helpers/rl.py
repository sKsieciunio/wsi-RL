from __future__ import annotations

from typing import Tuple
from collections import defaultdict

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


def dyna_q(
    env: SlipperyGridWorld,
    n_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    n_planning: int = 10,
    seed: int = 0,
) -> Tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.num_states, len(ACTIONS)))
    
    # Model: maps (state, action) -> (next_state, reward)
    model: dict[tuple[int, int], Tuple[int, float]] = defaultdict()
    
    episode_rewards: list[float] = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # 1. Real experience - perform action in environment
            action = epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning update
            Q[state,
              action] += alpha * (reward + gamma *
                                  (0 if done else np.max(Q[next_state])) -
                                  Q[state, action])
            
            # Update model
            model[(state, action)] = (next_state, reward)
            
            # 2. Planning: simulate n_planning steps using the model
            for _ in range(n_planning):
                # Sample a random (state, action) pair from the model
                if len(model) > 0:
                    model_keys = list(model.keys())
                    s, a = model_keys[int(rng.integers(len(model_keys)))]
                    s_prime, r = model[(s, a)]
                    
                    # Q-learning update using simulated experience
                    Q[s, a] += alpha * (r + gamma * np.max(Q[s_prime]) - Q[s, a])
            
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
    n_planning: int = 10,
    kappa: float = 0.01,
    seed: int = 0,
) -> Tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.num_states, len(ACTIONS)))
    
    # Model: maps (state, action) -> (next_state, reward)
    model: dict[tuple[int, int], Tuple[int, float]] = defaultdict()
    
    # Time tracking
    time_step = 0
    last_visit: dict[tuple[int, int], int] = defaultdict(lambda: 0)
    
    episode_rewards: list[float] = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # 1. Real experience - perform action in environment
            action = epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            time_step += 1
            
            # Q-learning update with real reward
            Q[state,
              action] += alpha * (reward + gamma *
                                  (0 if done else np.max(Q[next_state])) -
                                  Q[state, action])
            
            # Update model and last visit time
            model[(state, action)] = (next_state, reward)
            last_visit[(state, action)] = time_step
            
            # 2. Planning: simulate n_planning steps using the model
            for _ in range(n_planning):
                # Sample a random (state, action) pair from the model
                if len(model) > 0:
                    model_keys = list(model.keys())
                    s, a = model_keys[int(rng.integers(len(model_keys)))]
                    s_prime, r = model[(s, a)]
                    
                    # Calculate exploration bonus based on time since last visit
                    tau = time_step - last_visit[(s, a)]
                    exploration_bonus = kappa * np.sqrt(float(tau))
                    
                    # Q-learning update with exploration bonus
                    Q[s, a] += alpha * (r + exploration_bonus + gamma * np.max(Q[s_prime]) - Q[s, a])
            
            state = next_state
            total_reward += float(reward)

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, episode_rewards
