from __future__ import annotations
from collections import defaultdict
import random
from typing import Optional, Tuple

# Action mapping: 0=Up, 1=Right, 2=Down, 3=Left
ACTIONS = (0, 1, 2, 3)
ACTION_TO_DELTA = {
    0: (-1, 0),
    1: (0, +1),
    2: (+1, 0),
    3: (0, -1),
}


def _perpendicular_actions(a: int) -> Tuple[int, int]:
    if a in (0, 2):  # Up or Down
        return (3, 1)  # Left, Right
    else:  # Left or Right
        return (0, 2)  # Up, Down


class SlipperyGridWorld:
    """
    Simple tabular GridWorld with slippery actions.

    State space: integers 0..(rows*cols-1), row-major.
    Actions: 0=Up, 1=Right, 2=Down, 3=Left.

    Slippery dynamics:
      - with prob (1 - slip_prob): take intended action
      - with prob slip_prob: take one of the two perpendicular actions (uniformly)
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
        penalty_fields: Optional[set] = None,
        penalty_reward: float = -5.0,
        cliff_fields: Optional[set] = None,
        cliff_reward: float = -10.0,
        wall_fields: Optional[set] = None,
    ):
        assert rows > 0 and cols > 0
        assert 0.0 <= slip_prob <= 1.0

        self.rows = rows
        self.cols = cols
        self.start_row_column = start
        self.goal_row_column = goal
        self.slip_prob = slip_prob
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        self.penalty_fields: set = set(penalty_fields) if penalty_fields else set()
        self.penalty_reward = penalty_reward
        self.cliff_fields: set = set(cliff_fields) if cliff_fields else set()
        self.cliff_reward = cliff_reward
        self.wall_fields: set = set(wall_fields) if wall_fields else set()

        self.rng = random.Random(seed)
        self._steps = 0
        self._agent_row_column = start

        self.num_states = rows * cols
        self.nA = len(ACTIONS)

    # --- helpers ---
    def row_column_to_state(self, r: int, c: int) -> int:
        return r * self.cols + c

    def state_to_row_column(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.cols)

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_passable(self, r: int, c: int) -> bool:
        return self._in_bounds(r, c) and (r, c) not in self.wall_fields

    def _apply_action(self, r: int, c: int, a: int) -> Tuple[int, int]:
        dr, dc = ACTION_TO_DELTA[a]
        nr, nc = r + dr, c + dc
        if self._is_passable(nr, nc):
            return nr, nc
        return r, c

    def _sample_action_with_slip(self, intended: int) -> int:
        if self.rng.random() >= self.slip_prob:
            return intended
        left, right = _perpendicular_actions(intended)
        return left if self.rng.random() < 0.5 else right

    def _apply_action_stateless(self, state: int, action: int):
        r, c = self.state_to_row_column(state)
        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = r + dr, c + dc
        if self._is_passable(nr, nc):
            return self.row_column_to_state(nr, nc)
        return self.row_column_to_state(r, c)

    # --- public API ---
    def reset(self, start: Optional[Tuple[int, int]] = None) -> int:
        """Reset environment to start state specified (optional).

        Args:
            start (Optional[Tuple[int, int]], optional): If not specified, 
            takes start state from environment initialization. 
            Defaults to None.

        Returns:
            int: Reset agent's state.
        """
        if start is not None:
            self.start_row_column = start
        self._agent_row_column = self.start_row_column
        self._steps = 0
        return self.row_column_to_state(*self._agent_row_column)

    def get_transition_distribution(self, state: int,
                                    action: int) -> list[tuple[float, int]]:
        """Returns env transition probability distribution for given (s,a) and respective next states (s').
            Because the environment is slippery, attempting one action may lead to several possible next states.

        Args:
            state (int): Current state (s).
            action (int): Action to attempt (a).

        Returns:
            List of (probability, next_state) pairs.
        """
        assert action in ACTIONS, f"Invalid action {action}. Use 0=U,1=R,2=D,3=L."
        probs = [0] * len(ACTION_TO_DELTA)
        probs[action] = 1 - self.slip_prob
        act_s_1, act_s_2 = _perpendicular_actions(action)
        probs[act_s_1] = self.slip_prob / 2
        probs[act_s_2] = self.slip_prob / 2
        state_probs = defaultdict(float)

        for a_real, prob in enumerate(probs):
            if prob == 0:
                continue

            next_state = self._apply_action_stateless(state, a_real)
            state_probs[next_state] += prob

        return [(p, s_next) for s_next, p in state_probs.items()]

    def is_terminal_state(self, state: int) -> bool:
        """Check if the current state is terminal in the environment."""
        rc = self.state_to_row_column(state)
        return rc == self.goal_row_column or rc in self.cliff_fields

    def step(self, action: int):
        """Perform one step in the environment.

        Args:
            action (int): Action to perform [0, 1, 2, 3].

        Returns:
            Next state, reward, flag done, info dictionary
        """
        assert action in ACTIONS, f"Invalid action {action}. Use 0=U,1=R,2=D,3=L."
        self._steps += 1

        intended = action
        executed = self._sample_action_with_slip(intended)

        r, c = self._agent_row_column
        nr, nc = self._apply_action(r, c, executed)
        self._agent_row_column = (nr, nc)

        at_goal = self._agent_row_column == self.goal_row_column
        at_cliff = self._agent_row_column in self.cliff_fields
        done = at_goal or at_cliff
        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True

        if at_goal:
            reward = self.goal_reward
        elif at_cliff:
            reward = self.cliff_reward
        elif self._agent_row_column in self.penalty_fields:
            reward = self.penalty_reward
        else:
            reward = self.step_reward

        info = {
            "intended_action": intended,
            "executed_action": executed,
            "steps": self._steps
        }
        return self.row_column_to_state(
            *self._agent_row_column), reward, done, info

    def add_wall(self, rc: Tuple[int, int]) -> None:
        """Add a wall cell (row, col). Agent cannot enter it."""
        self.wall_fields.add(rc)

    def remove_wall(self, rc: Tuple[int, int]) -> None:
        """Remove a wall cell, opening the passage."""
        self.wall_fields.discard(rc)

    def set_goal(self, goal: Tuple[int, int]):
        """Specify the goal state

        Args:
            goal (Tuple[int, int]): (row, column)
        """
        self.goal_row_column = goal

    def reward(self, state: int, action: int, next_state: int) -> float:
        """Return the reward R(s, a, s') for a transition.
        
        In this simplified GridWorld, reward depends only on the next state

        Args:
            state (int): State for which the reward should be retrieved.
            action (int): Attempted action.
            next_state (int): Next state after action in state.

        Returns:
            float: reward from the environment.
        """
        src_rc = self.state_to_row_column(state)
        if src_rc == self.goal_row_column or src_rc in self.cliff_fields:
            return 0.0
        dst_rc = self.state_to_row_column(next_state)
        if dst_rc == self.goal_row_column:
            return self.goal_reward
        if dst_rc in self.cliff_fields:
            return self.cliff_reward
        if dst_rc in self.penalty_fields:
            return self.penalty_reward
        return self.step_reward

    def set_size(self,
                 rows: int,
                 cols: int,
                 start: Optional[Tuple[int, int]] = None,
                 goal: Optional[Tuple[int, int]] = None) -> None:
        """Set enviroment grid size.

        Args:
            rows (int): Number of rows.
            cols (int): Number of columns.
            start (Optional[Tuple[int, int]], optional): Start state (row, column). Defaults to None.
            goal (Optional[Tuple[int, int]], optional): Goal state (row, column). Defaults to None.
        """
        assert rows > 0 and cols > 0
        self.rows, self.cols = rows, cols
        self.num_states = rows * cols
        if start is not None:
            self.start_row_column = start
        if goal is not None:
            self.goal_row_column = goal
        r, c = self._agent_row_column
        self._agent_row_column = (min(max(r, 0),
                                      rows - 1), min(max(c, 0), cols - 1))
