from __future__ import annotations
import random
from typing import Optional, Tuple, Dict, List
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from helpers.env import ACTIONS, SlipperyGridWorld

ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}
def _base_grid_figure(env, title: str = ""):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)

    ax.set_xticks(np.arange(-0.5, env.cols, 1))
    ax.set_yticks(np.arange(-0.5, env.rows, 1))
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    return fig, ax

def plot_policy(
    env: SlipperyGridWorld,
    policy: np.ndarray,
    filename: Optional[str] = None,
    title: str = "Policy",
) -> None:
    """Visualize policy for each state

    Args:
        env (SlipperyGridWorld): Initialized environment.
        policy (np.ndarray): Policy (deterministic action per each state).
        filename (Optional[str], optional): Where to save the plot. Defaults to None.
        title (str, optional): Defaults to "Policy".
        
    """
    fig, ax = _base_grid_figure(env, title=title)

    sr, sc = env.start_row_column
    gr, gc = env.goal_row_column
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=14, fontweight="bold")

    for s in range(env.num_states):
        r, c = env.state_to_row_column(s)
        if (r, c) in [env.start_row_column, env.goal_row_column]:
            continue
        a = int(policy[s])
        ax.text(c, r, ARROWS[a], ha="center", va="center", fontsize=14)

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()

def plot_value_heatmap(
    env,
    V: np.ndarray,
    filename: Optional[str] = None,
    title: str = "State Value",
) -> None:
    """Produces a heatmap image for V(s).

    Args:
        env (SlipperyGridWorld): Initialized environment.
        V (np.ndarray): V(s)
        filename (Optional[str], optional): Where to save the plot. Defaults to None.
        title (str, optional): Defaults to "State Value".

    """
    V_grid = V.reshape(env.rows, env.cols)
    fig, ax = plt.subplots()
    im = ax.imshow(V_grid)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    sr, sc = env.start_row_column
    gr, gc = env.goal_row_column
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=12, fontweight="bold")

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()

def render_episode_frames(
    env:SlipperyGridWorld,
    trajectory: List[Tuple[int,int,int,float,int,bool]],
    out_dir: str = "frames",
    prefix: str = "frame",
    show_executed_action: bool = True,
) -> List[str]:
    """
    Saves one PNG per step with agent position and (optionally) executed action.
    Returns list of filepaths.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    for t, (s, a_intended, a_exec, r, s_next, done) in enumerate(trajectory):
        r_next, c_next = env.state_to_row_column(s_next)

        fig, ax = _base_grid_figure(env, title=f"t={t}, r={r:.2f}, done={done}")
        sr, sc = env.start_row_column
        gr, gc = env.goal_row_column
        ax.text(sc, sr, "S", ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(gc, gr, "G", ha="center", va="center", fontsize=14, fontweight="bold")

        ax.text(c_next, r_next, "A", ha="center", va="center", fontsize=16, fontweight="bold")

        if show_executed_action:
            ax.set_title(f"t={t}  intended={ARROWS[a_intended]}  executed={ARROWS[a_exec]}  r={r:.2f}  done={done}")

        path = os.path.join(out_dir, f"{prefix}_{t:04d}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved

def run_to_gif(env: SlipperyGridWorld, Q: Optional[np.ndarray]=None, policy: Optional[np.ndarray]=None, gif_path: str = "episode.gif", fps: int = 6) -> None:
    """Creates a gif for a single run of the agent in the environment.

    Args:
        env (SlipperyGridWorld): Initialized environment.
        Q (Optional[np.ndarray], Optional): Q(s,a).
        policy (Optional[np.ndarray], optional): pi(s). Defaults to None.
        gif_path (str, optional): Where to save the path. Defaults to "episode.gif".
        fps (int, optional): Frames per second for the gif. Defaults to 6.
    """
    roll = run_episode(env, Q=Q, policy = policy)
    frames = render_episode_frames(env, roll["trajectory"], out_dir="frames", prefix="ep")
    imgs = [imageio.imread(p) for p in frames]
    imageio.mimsave(gif_path, imgs, duration=1.0 / fps)

def greedy_policy_from_V(V: np.ndarray, env: SlipperyGridWorld, gamma: float):
    """Returns greedy policy rom the value function V(s)

    Args:
        V (np.ndarray): Array of values for each state.
        env (SlipperyGridWorld): Initialzed environment.
        gamma (float): Discount factor (0 < gamma < 1).
    
    Returns:
        pi(s)
    """
    policy = np.zeros(len(V))
    for state in range(len(V)):
        q_a = [-np.inf]*len(ACTIONS)
        for a in ACTIONS:
            q = 0.0
            for p, s_next in env.get_transition_distribution(state, a):
                r = env.reward(state, a, s_next)
                if env.is_terminal_state(s_next):
                    q += p * (r)
                else:
                    q += p * (r + gamma * V[s_next])
            q_a[a] = q
        policy[state] = int(np.argmax(q_a))
    return policy

def run_episode(
    env: SlipperyGridWorld,
    Q: Optional[np.ndarray] = None,
    policy: Optional[np.ndarray] = None,
    seed: int = None
) -> Dict:
    """Roll out a single episode.

    Args:
        env (SlipperyGridWorld): environment
        Q (Optional[np.ndarray], optional): Q(s,a). Defaults to None.
        policy (Optional[np.ndarray], optional): pi(s). Defaults to None.

    Returns:
        Dict: episode run stats
    """
    assert (Q is not None) or (policy is not None), "Provide Q or policy"

    s = env.reset()
    if seed is not None:
        env.rng = random.Random(seed)
    done = False
    total_return = 0.0
    steps = 0

    traj = []

    while not done:

        if policy is not None:
            a = int(policy[s])
        else:
            a = int(np.argmax(Q[s]))

        s_next, r, done, info = env.step(a)

        traj.append((s, a, info.get("executed_action", a), r, s_next, done))
        total_return += float(r)
        s = s_next
        steps += 1

    success = (env.state_to_row_column(s) == env.goal_row_column)
    return {
        "return": total_return,
        "steps": steps,
        "success": bool(success),
        "trajectory": traj,
    }

def evaluate(
    env: SlipperyGridWorld,
    Q: Optional[np.ndarray] = None,
    policy: Optional[np.ndarray] = None,
    n_episodes: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Evaluate resulting Q(s,a) or deterministic policy.

    Args:
        env (SlipperyGridWorld): Initialized environment.
        Q (Optional[np.ndarray], optional): Q(s,a). Defaults to None.
        policy (Optional[np.ndarray], optional): pi(s). Defaults to None.
        n_episodes (int, optional): Number of episodes to run in evaluation. Defaults to 200.
        seed (int, optional): random seed. Defaults to 0.

    Returns:
        Dict[str, float]: Evaluation stats.
    """
    rng = np.random.default_rng(seed)
    returns, steps, success = [], [], []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 1_000_000))
        res = run_episode(env, Q=Q, policy=policy, seed=ep_seed)
        returns.append(res["return"])
        steps.append(res["steps"])
        success.append(1.0 if res["success"] else 0.0)

    return {
        "avg_return": float(np.round(np.mean(returns), decimals=4)),
        "std_return": float(np.round(np.std(returns), decimals=4)),
        "success_rate": float(np.round(np.mean(success), decimals=4)),
        "avg_steps": float(np.round(np.mean(steps), decimals=4)),
    }
