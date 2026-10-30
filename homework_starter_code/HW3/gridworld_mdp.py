from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize

Action = int      # 0..3
State = int       # 0..(n_states-1)
Coord = Tuple[int, int]  # (row, col)

# Actions: Up, Right, Down, Left
ACTIONS: Dict[Action, Tuple[int, int]] = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}
ACTION_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

@dataclass(frozen=True)
class Transition:
    """One transition outcome for (s, a)."""
    prob: float
    next_state: State
    reward: float
    done: bool

class GridworldMDP:
    """
    4-action gridworld
    """

    def __init__(
        self,
        height: int,
        width: int,
        init: Coord,
        goal: Coord,
        sink: Coord | None = None,
        wall: Coord | List[Coord] | None = None,
        reward_goal: float = +1.0,
        reward_sink: float = -1.0,
        step_cost: float = -0.1,
        slip_p: float = 0.3,
        discount : float = 0.99,
        reward_depends_on_next : bool = False,
    ):
        assert height >= 1 and width >= 1
        assert 0.0 <= slip_p <= 1.0
        self.height, self.width = height, width

        if isinstance(wall, list): # multiple walls
            self.walls = list(wall)
        elif isinstance(wall, tuple): # single wall
            if len(wall) != 2 or not all(isinstance(x, int) for x in wall):
                raise TypeError("wall must be a (row, col) tuple of ints")
            self.walls = [wall]
        elif wall is None:
            self.walls = []
        else:
            raise TypeError("wall must be None, a (row, col) tuple, or a list of such tuples")

        self.reward_goal = reward_goal
        self.reward_sink = reward_sink
        self.step_cost = step_cost
        self.slip_p = slip_p
        self.discount = discount
        self.reward_depends_on_next = bool(reward_depends_on_next)

        # Build state space as all cells except the wall (if any)
        coords: List[Coord] = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r, c) not in self.walls
        ]
        self._idx_of: Dict[Coord, State] = {xy: i for i, xy in enumerate(coords)}
        self._coord_of: Dict[State, Coord] = {i: xy for xy, i in self._idx_of.items()}

        # Validate special cells
        for name, xy in [("init", init), ("goal", goal), ("sink", sink)]:
            if xy is not None and xy not in self._idx_of:
                raise ValueError(f"{name} cell {xy} is invalid (outside grid or is the wall).")

        self.init: State = self._idx_of[init]
        self.goal: State = self._idx_of[goal]
        if sink:
            self.sink: State = self._idx_of[sink]
        else:
            self.sink = None

    # ------------- Basic API -------------

    @property
    def nS(self) -> int:
        return len(self._coord_of)

    @property
    def nA(self) -> int:
        return 4

    def state_index(self, coord: Coord) -> State:
        return self._idx_of[coord]

    def state_coord(self, s: State) -> Coord:
        return self._coord_of[s]

    # Absorbing (but not terminal)
    def is_absorbing(self, s: State) -> bool:
        return s == self.goal or s == self.sink
    
    def get_R_pi(self, pi) -> np.ndarray:
        # Construct a vector of size mdp.nS
        R = np.zeros(self.nS)
        for s in range(self.nS):
            for a in range(self.nA):
                R[s] += pi[s, a] * self._reward_sa(s, a)
        return R
    
    def get_P_pi(self, pi) -> np.ndarray:
        # Construct a transition probability matrix of size (mdp.nS, mdp.nS) under policy pi
        P = np.zeros((self.nS, self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                for s_next in range(self.nS):
                    P[s, s_next] += pi[s, a] * self.transition_probabilities(s, a).get(s_next, 0.0)
        return P

    # ------------- Helpers -------------

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    def _is_wall(self, r: int, c: int) -> bool:
        return len(self.walls) > 0 and (r, c) in self.walls

    def _bounce_next(self, coord: Coord, a: Action) -> Coord:
        """Deterministic move with wall/edge bounce. Absorbing states self-loop."""
        if self.is_absorbing(self._idx_of[coord]):
            return coord
        dr, dc = ACTIONS[a]
        nr, nc = coord[0] + dr, coord[1] + dc
        if not self._in_bounds(nr, nc) or self._is_wall(nr, nc):
            return coord  # bounce
        return (nr, nc)

    def _adjacent_valid_coords(self, coord: Coord) -> List[Coord]:
        """Neighbors that are within bounds and not the wall (no bounce)."""
        out: List[Coord] = []
        for dr, dc in ACTIONS.values():
            nr, nc = coord[0] + dr, coord[1] + dc
            if self._in_bounds(nr, nc) and not self._is_wall(nr, nc):
                out.append((nr, nc))
        return out

    # ------------- Dynamics -------------

    def reward(self, s: State, a: Action, s_next: State) -> float:
        """Reward on entry: goal -> +R, sink -> -R; else step_cost.
        Once in absorbing states, further actions yield 0."""
        if self.is_absorbing(s):
            return 0.0
        if s_next == self.goal:
            return self.reward_goal
        if s_next == self.sink:
            return self.reward_sink
        return self.step_cost
    
    def _reward_sa(self, s: State, a: Action) -> float:
        """Expected one-step reward for (s,a) under current slip/bounce dynamics."""
        if self.is_absorbing(s):
            return 0.0
        dist = self.transition_probabilities(s, a)
        p_goal = dist.get(self.goal, 0.0)
        p_sink = dist.get(self.sink, 0.0)
        p_other = 1.0 - p_goal - p_sink
        return (
            self.reward_goal * p_goal
            + self.reward_sink * p_sink
            + self.step_cost * p_other
        )

    def transition_probabilities(self, s: State, a: Action) -> Dict[State, float]:
        """
        For non-absorbing s:
        - intended = bounce move for action a, gets prob (1 - slip_p)
        - let N = valid adjacent coords (no bounce); give slip_p/|N'| to each coord in N' = N \\ {intended}
            (if intended == current due to bounce, it's not in N so slip spreads over all valid neighbors)
        - if N' is empty, add slip_p to intended (i.e., total prob still 1)
        For absorbing s: self-loop with prob 1.
        """
        if self.is_absorbing(s):
            return {s: 1.0}

        coord = self._coord_of[s]
        intended_coord = self._bounce_next(coord, a)
        intended_state = self._idx_of[intended_coord]

        # Start with intended move
        dist: Dict[State, float] = {intended_state: 1.0 - self.slip_p}

        # Compute slip targets
        neighbors = self._adjacent_valid_coords(coord)
        # Exclude intended destination if it is among valid neighbors
        slip_targets = [xy for xy in neighbors if xy != intended_coord]
        if len(slip_targets) == 0:
            # Nowhere else to slip -> fold slip mass into intended
            dist[intended_state] = dist.get(intended_state, 0.0) + self.slip_p
        else:
            p_each = self.slip_p / len(slip_targets)
            for xy in slip_targets:
                dist[self._idx_of[xy]] = dist.get(self._idx_of[xy], 0.0) + p_each

        total = sum(dist.values())
        if abs(total - 1.0) > 1e-12:
            # Normalize
            for k in list(dist.keys()):
                dist[k] /= total
        return dist

    def P(self, s: State, a: Action) -> List[Transition]:
        """List of (prob, next_state, reward, done) for (s, a)."""
        dist = self.transition_probabilities(s, a)
        use_expected = not self.reward_depends_on_next
        r_sa = self._reward_sa(s, a) if use_expected else None

        out: List[Transition] = []
        for s_next, p in dist.items():
            r = r_sa if use_expected else self.reward(s, a, s_next)
            out.append(Transition(prob=p, next_state=s_next, reward=r, done=False))
        return out

    # ------------- Utility -------------

    def render_ascii(self) -> str:
        """ASCII map showing I, G, X(wall), S(sink), and dots for free cells."""
        symbols = [["." for _ in range(self.width)] for _ in range(self.height)]
        for wall in self.walls:
            wr, wc = wall
            symbols[wr][wc] = "X"
        gr, gc = self.state_coord(self.goal)
        symbols[gr][gc] = "G"
        if self.sink is not None:
            sr, sc = self.state_coord(self.sink)
            symbols[sr][sc] = "S"
        ir, ic = self.state_coord(self.init)
        symbols[ir][ic] = "I"
        lines = [" ".join(row) for row in symbols]
        return "\n".join(lines)
    
    # ---------------- Visualization ----------------

    def _cell_corners(self, r: int, c: int):
        """Return corners of cell (r,c) as (x,y) in matplotlib coords (origin at top-left)."""
        # We'll let x be column index, y be row index; imshow-style with y increasing down.
        x0, y0 = c, r
        x1, y1 = c + 1, r + 1
        return (x0, y0), (x1, y1)

    def plot_grid(self, ax: plt.Axes | None = None, show: bool = True, draw_init: bool = True):
        """
        Draws the grid with:
        - walls = black
        - goal  = green
        - sink  = red
        - other cells white with gray gridlines
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.width, self.height))

        # Base white background
        ax.set_aspect('equal')
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)  # invert y to have (0,0) at top-left
        ax.set_xticks(np.arange(0, self.width + 1, 1))
        ax.set_yticks(np.arange(0, self.height + 1, 1))
        ax.grid(True, which='both', linestyle='-', linewidth=0.8, alpha=0.4, color='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.set_frame_on(False)

        # Fill all cells white first
        for r in range(self.height):
            for c in range(self.width):
                (x0, y0), (x1, y1) = self._cell_corners(r, c)
                rect = patches.Rectangle((x0, y0), 1, 1, facecolor='white', edgecolor='none')
                ax.add_patch(rect)

        # Walls
        for wall in self.walls:
            wr, wc = wall
            (x0, y0), _ = self._cell_corners(wr, wc)
            ax.add_patch(patches.Rectangle((x0, y0), 1, 1, facecolor='black'))

        # Goal (green) and Sink (red)
        gr, gc = self.state_coord(self.goal)
        ax.add_patch(patches.Rectangle((gc, gr), 1, 1, facecolor='green', alpha=0.85))
        if self.sink is not None:
            sr, sc = self.state_coord(self.sink)
            ax.add_patch(patches.Rectangle((sc, sr), 1, 1, facecolor='red',   alpha=0.85))

        # Init (optional marker)
        if draw_init:
            ir, ic = self.state_coord(self.init)
            ax.text(ic + 0.5, ir + 0.5, "I", ha='center', va='center', fontsize=12, weight='bold', color='black')

        if show:
            plt.show()
        return ax

    def plot_policy(
        self,
        policy: np.ndarray | Dict[State, np.ndarray],
        ax: plt.Axes | None = None,
        cmap: str = 'Blues',
        show: bool = True,
        vmin: float = 0.0,
        vmax: float = 1.0,
        draw_grid_first: bool = True,
    ):
        """
        Visualize a stochastic policy π(a|s) by splitting each cell into 4 triangles:
            top (↑=0), right (→=1), bottom (↓=2), left (←=3)
        Colored by probability using a colormap. Adds a colorbar.

        Args:
            policy: shape (nS, 4) array of probs OR dict {state: np.array(4)}.
            vmin, vmax: color scale bounds for probabilities (default 0..1).
        """
        # Normalize policy input
        if isinstance(policy, dict):
            pi = np.zeros((self.nS, self.nA), dtype=float)
            for s in range(self.nS):
                if s in policy:
                    pi[s] = np.asarray(policy[s], dtype=float)
            # rows not in dict remain zeros
        else:
            pi = np.asarray(policy, dtype=float)
            assert pi.shape == (self.nS, self.nA), f"policy must have shape (nS, nA)=({self.nS}, {self.nA})"
        # Optional renorm to guard small numeric issues
        row_sums = pi.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        pi = np.clip(pi / row_sums, 0.0, 1.0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(self.width, self.height))

        if draw_grid_first:
            self.plot_grid(ax=ax, show=False, draw_init=False)

        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        # Draw triangles per cell
        for s in range(self.nS):
            r, c = self.state_coord(s)
            # Skip the wall (not in state space by construction) and draw absorbing cells normally
            x0, y0 = c, r
            x1, y1 = c + 1, r + 1
            xc, yc = (x0 + x1) / 2.0, (y0 + y1) / 2.0

            # Triangle corners for actions: 0 up,1 right,2 down,3 left
            tris = {
                0: [(x0, y0), (x1, y0), (xc, yc)],       # top
                1: [(x1, y0), (x1, y1), (xc, yc)],       # right
                2: [(x0, y1), (x1, y1), (xc, yc)],       # bottom
                3: [(x0, y0), (x0, y1), (xc, yc)],       # left
            }

            # For absorbing states, still show the triangles (policy may exist); optionally outline
            for a in range(4):
                color = sm.to_rgba(pi[s, a])
                poly = patches.Polygon(tris[a], closed=True, facecolor=color, edgecolor='none')
                ax.add_patch(poly)

            # Thin outline so cell borders remain readable
            ax.add_patch(patches.Rectangle((x0, y0), 1, 1, fill=False, edgecolor='gray', linewidth=0.6, alpha=0.6))

        # Draw any walls as black squares
        for wall in self.walls:
            wr, wc = wall
            ax.add_patch(patches.Rectangle((wc, wr), 1, 1, facecolor='black'))

        # Colorbar
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("π(a|s)")

        # Re-mark goal/sink overlays so they are visible on top
        gr, gc = self.state_coord(self.goal)
        ax.add_patch(patches.Rectangle((gc, gr), 1, 1, fill=False, edgecolor='green', linewidth=2.0))

        if self.sink is not None:
            sr, sc = self.state_coord(self.sink)
            ax.add_patch(patches.Rectangle((sc, sr), 1, 1, fill=False, edgecolor='red', linewidth=2.0))

        # ax.set_title("Policy visualization (triangles per action)")
        if show:
            plt.show()
        return ax

    def plot_values(
        self,
        V: np.ndarray | Dict[State, float],
        ax: plt.Axes | None = None,
        cmap: str = 'viridis',
        show: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        annotate: bool = False,
        fmt: str = ".2f",
    ):
        """
        Heatmap of state values V(s). Walls are masked.
        Args:
            V: array of shape (nS,) or dict {state: value}
            annotate: write numeric values in cells
        """
        if isinstance(V, dict):
            V_arr = np.zeros((self.nS,), dtype=float)
            for s, v in V.items():
                V_arr[s] = float(v)
        else:
            V_arr = np.asarray(V, dtype=float).reshape(self.nS)

        # Build a (height x width) array with NaN for the wall (not in state set)
        grid = np.full((self.height, self.width), np.nan, dtype=float)
        for s in range(self.nS):
            r, c = self.state_coord(s)
            grid[r, c] = V_arr[s]

        if ax is None:
            fig, ax = plt.subplots(figsize=(self.width, self.height))

        im = ax.imshow(grid, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)

        # Overlay styling and special cells
        ax.set_xticks(np.arange(self.width))
        ax.set_yticks(np.arange(self.height))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        # Grid lines
        for r in range(self.height + 1):
            ax.axhline(r - 0.5, color='white', linewidth=0.8, alpha=0.7)
        for c in range(self.width + 1):
            ax.axvline(c - 0.5, color='white', linewidth=0.8, alpha=0.7)

        # Walls black
        for wall in self.walls:
            wr, wc = wall
            ax.add_patch(patches.Rectangle((wc - 0.5, wr - 0.5), 1, 1, facecolor='black', edgecolor='black'))

        # Outline goal/sink
        gr, gc = self.state_coord(self.goal)
        ax.add_patch(patches.Rectangle((gc - 0.5, gr - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2))

        if self.sink is not None:
            sr, sc = self.state_coord(self.sink)
            ax.add_patch(patches.Rectangle((sc - 0.5, sr - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2))

        # Optional annotation
        if annotate:
            for s in range(self.nS):
                r, c = self.state_coord(s)
                if len(self.walls) > 0 and (r, c) in self.walls:
                    continue
                if np.isfinite(grid[r, c]):
                    ax.text(c, r, format(grid[r, c], fmt), ha='center', va='center', color='black')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("V(s)")
        # ax.set_title("State Value Heatmap")
        if show:
            plt.show()
        return ax