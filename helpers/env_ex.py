from __future__ import annotations
from collections import defaultdict
import random
from typing import Optional, Tuple, List

from helpers.env import SlipperyGridWorld, ACTIONS


class SlipperyGridWorldExtended(SlipperyGridWorld):
    """Extended GridWorld with optional dynamic features inferred from parameters:

    - moving_goal: enabled when goal_move_frequency is not None
    - moving_obstacles: enabled when obstacle_move_frequency is not None and num_obstacles > 0
    - multiple_goals: enabled when goals_list is not None

    The implementation reuses parent helpers (`_apply_action`,
    `_sample_action_with_slip`, `row_column_to_state`, etc.) via `super()`.
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (3, 3),
        slip_prob: float = 0.2,
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        # moving goal params
        goal_move_frequency: Optional[int] = None,
        # obstacles params
        num_obstacles: int = 0,
        obstacle_penalty: float = -5.0,
        obstacle_move_frequency: Optional[int] = None,
        # multiple goals params
        goals_list: Optional[List[Tuple[int, int]]] = None,
        goal_rewards: Optional[List[float]] = None,
    ):
        # Infer multiple_goals mode from goals_list
        use_multiple_goals = goals_list is not None
        primary_goal = goals_list[0] if use_multiple_goals else goal
        
        super().__init__(
            rows=rows,
            cols=cols,
            start=start,
            goal=primary_goal,
            slip_prob=slip_prob,
            step_reward=step_reward,
            goal_reward=goal_reward,
            max_steps=max_steps,
            seed=seed,
        )

        # Moving goal settings (inferred from goal_move_frequency)
        self.goal_move_frequency = goal_move_frequency
        self.steps_since_goal_move = 0

        # Multiple goals (inferred from goals_list)
        if use_multiple_goals:
            self.goals = goals_list
            self.goal_rewards = goal_rewards or [goal_reward] * len(self.goals)
            assert len(self.goals) == len(self.goal_rewards), "Number of goals must match rewards"
        else:
            self.goals = [self.goal_row_column]
            self.goal_rewards = [goal_reward]

        # Obstacles (inferred from obstacle_move_frequency and num_obstacles)
        self.num_obstacles = num_obstacles
        self.obstacle_penalty = obstacle_penalty
        self.obstacle_move_frequency = obstacle_move_frequency
        self.steps_since_obstacle_move = 0
        self.obstacles: List[Tuple[int, int]] = []
        
        # Initialize obstacles only if moving and count > 0
        if obstacle_move_frequency is not None and num_obstacles > 0:
            self._initialize_obstacles()

    # --- obstacles helpers ---
    def _initialize_obstacles(self):
        """Place obstacles randomly avoiding start and goals."""
        self.obstacles.clear()
        excluded = {self.start_row_column, *self.goals}
        while len(self.obstacles) < self.num_obstacles:
            r = self.rng.randint(0, self.rows - 1)
            c = self.rng.randint(0, self.cols - 1)
            pos = (r, c)
            if pos not in excluded and pos not in self.obstacles:
                self.obstacles.append(pos)

    def _move_obstacles_randomly(self):
        """Move obstacles randomly to adjacent cells (including staying)."""
        if not self.obstacles:
            return
        new_obstacles: List[Tuple[int, int]] = []
        excluded = {self.start_row_column, *self.goals}
        for obs_r, obs_c in self.obstacles:
            # choice: 0..3 = move, 4 = stay
            action = self.rng.choice([0, 1, 2, 3, 4])
            if action < 4:
                nr, nc = self._apply_action(obs_r, obs_c, action)
            else:
                nr, nc = obs_r, obs_c
            if (nr, nc) not in excluded:
                new_obstacles.append((nr, nc))
            else:
                new_obstacles.append((obs_r, obs_c))
        self.obstacles = new_obstacles

    # --- moving goal helper ---
    def _move_goal_randomly(self):
        new_r = self.rng.randint(0, self.rows - 1)
        new_c = self.rng.randint(0, self.cols - 1)
        self.goal_row_column = (new_r, new_c)
        # if not using multiple goals, update primary list entry
        if len(self.goals) == 1:
            self.goals = [self.goal_row_column]

    # --- public API overrides ---
    def reset(self, start: Optional[Tuple[int, int]] = None) -> int:
        """Reset agent and reinitialize dynamic elements if enabled."""
        if start is not None:
            self.start_row_column = start
        self._agent_row_column = self.start_row_column
        self._steps = 0
        self.steps_since_goal_move = 0
        self.steps_since_obstacle_move = 0
        if self.obstacle_move_frequency is not None and self.num_obstacles > 0:
            self._initialize_obstacles()
        return self.row_column_to_state(*self._agent_row_column)

    def is_terminal_state(self, state: int) -> bool:
        state_pos = self.state_to_row_column(state)
        if self.goals is not None and len(self.goals) > 1:
            return state_pos in self.goals
        return state_pos == self.goal_row_column

    def step(self, action: int):
        """Perform one step applying enabled features while reusing parent helpers."""
        assert action in ACTIONS, f"Invalid action {action}. Use 0=U,1=R,2=D,3=L."
        self._steps += 1

        intended = action
        executed = self._sample_action_with_slip(intended)

        r, c = self._agent_row_column
        nr, nc = self._apply_action(r, c, executed)
        self._agent_row_column = (nr, nc)

        current_pos = self._agent_row_column
        reached_goal_idx = -1
        if len(self.goals) > 1:
            for idx, g in enumerate(self.goals):
                if current_pos == g:
                    reached_goal_idx = idx
                    break
        else:
            if current_pos == self.goal_row_column:
                reached_goal_idx = 0

        done = reached_goal_idx >= 0
        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True

        reward = self.step_reward
        info = {"intended_action": intended, "executed_action": executed, "steps": self._steps}

        # obstacle collision
        if self.obstacle_move_frequency is not None and (current_pos in self.obstacles):
            reward += self.obstacle_penalty
            info["hit_obstacle"] = True
        else:
            info["hit_obstacle"] = False

        # goal reward
        if reached_goal_idx >= 0:
            reward = self.goal_rewards[reached_goal_idx]
            info["reached_goal_idx"] = reached_goal_idx

        # move obstacles periodically
        if self.obstacle_move_frequency is not None and self.obstacles:
            self.steps_since_obstacle_move += 1
            if self.steps_since_obstacle_move >= self.obstacle_move_frequency:
                self._move_obstacles_randomly()
                self.steps_since_obstacle_move = 0

        # move goal periodically
        if self.goal_move_frequency is not None:
            self.steps_since_goal_move += 1
            if self.steps_since_goal_move >= self.goal_move_frequency:
                self._move_goal_randomly()
                self.steps_since_goal_move = 0
                info["goal_moved"] = True

        return self.row_column_to_state(*self._agent_row_column), reward, done, info

