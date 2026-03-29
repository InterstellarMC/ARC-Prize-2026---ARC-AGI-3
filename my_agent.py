import hashlib
import logging
import os
import random
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def hash_frame(frame_2d: np.ndarray) -> str:
    """Fast 16-char hex hash of a 2D integer frame."""
    return hashlib.md5(frame_2d.tobytes()).hexdigest()[:16]


def setup_experiment_directory(base_output_dir='runs'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(base_dir, exist_ok=True)
    log_file = os.path.join(base_dir, 'logs.log')
    return base_dir, log_file


# ---------------------------------------------------------------------------
# State Graph — directed graph of game states and transitions
# ---------------------------------------------------------------------------

class StateGraph:
    """
    Directed graph tracking the explored state space.
    Nodes  : unique frame hashes
    Edges  : (src_hash, action_id) -> dst_hash
    """

    def __init__(self):
        self.transitions: Dict[Tuple[str, int], str] = {}
        self.tried_actions: Dict[str, Set[int]] = defaultdict(set)
        self.all_states: Set[str] = set()
        self.visit_count: Dict[str, int] = defaultdict(int)
        self.self_loops: Dict[str, Set[int]] = defaultdict(set)
        # Cross-level stats
        self.action_change_count: Dict[int, int] = defaultdict(int)
        self.action_try_count: Dict[int, int] = defaultdict(int)

    def add_state(self, h: str):
        self.all_states.add(h)
        self.visit_count[h] += 1

    def record(self, src: str, action_id: int, dst: str):
        self.transitions[(src, action_id)] = dst
        self.tried_actions[src].add(action_id)
        self.action_try_count[action_id] += 1
        self.all_states.add(src)
        self.all_states.add(dst)
        if src != dst:
            self.action_change_count[action_id] += 1
        else:
            self.self_loops[src].add(action_id)

    def untried(self, state: str, available: List[int]) -> List[int]:
        tried = self.tried_actions.get(state, set())
        return [a for a in available if a not in tried]

    def has_untried(self, state: str, available: List[int]) -> bool:
        tried = self.tried_actions.get(state, set())
        return any(a not in tried for a in available)

    def effectiveness(self, action_id: int) -> float:
        t = self.action_try_count.get(action_id, 0)
        if t == 0:
            return 0.5
        return self.action_change_count.get(action_id, 0) / t

    def bfs_to_frontier(self, start: str, available: List[int]) -> Optional[List[int]]:
        """
        BFS from *start* through known transitions to find untried actions.
        """
        if self.has_untried(start, available):
            return []
        queue: deque = deque()
        visited: Set[str] = {start}
        for aid in available:
            key = (start, aid)
            if key in self.transitions:
                nxt = self.transitions[key]
                if nxt != start and nxt not in visited:
                    visited.add(nxt)
                    path = [aid]
                    if self.has_untried(nxt, available):
                        return path
                    queue.append((nxt, path))
        while queue:
            state, path = queue.popleft()
            if len(path) > 80:
                continue
            for aid in available:
                key = (state, aid)
                if key in self.transitions:
                    nxt = self.transitions[key]
                    if nxt != state and nxt not in visited:
                        visited.add(nxt)
                        new_path = path + [aid]
                        if self.has_untried(nxt, available):
                            return new_path
                        queue.append((nxt, new_path))
        return None

    def reset(self):
        self.transitions.clear()
        self.tried_actions.clear()
        self.all_states.clear()
        self.visit_count.clear()
        self.self_loops.clear()

    def full_reset(self):
        self.reset()
        self.action_change_count.clear()
        self.action_try_count.clear()


# ---------------------------------------------------------------------------
# Frame Analyzer
# ---------------------------------------------------------------------------

class FrameAnalyzer:
    @staticmethod
    def background_color(frame: np.ndarray) -> int:
        unique, counts = np.unique(frame, return_counts=True)
        return int(unique[np.argmax(counts)])

    @staticmethod
    def color_clusters(frame: np.ndarray, bg: int) -> List[Tuple[int, int, int]]:
        out = []
        for c in range(16):
            if c == bg:
                continue
            ys, xs = np.where(frame == c)
            if len(ys) == 0:
                continue
            out.append((int(np.mean(ys)), int(np.mean(xs)), c))
        return out


# ---------------------------------------------------------------------------
# Click Solver — coordinate action handler
# ---------------------------------------------------------------------------

class ClickSolver:
    def __init__(self):
        self.tried: Set[Tuple[int, int, str]] = set()
        self.queue: deque = deque()
        self.effective: List[Tuple[int, int]] = []

    def prepare(self, frame: np.ndarray, state_hash: str, bg: int):
        self.queue.clear()
        seen: Set[Tuple[int, int]] = set()
        def _enq(y, x):
            if (y, x) not in seen and (y, x, state_hash) not in self.tried:
                seen.add((y, x))
                self.queue.append((y, x))
        for cy, cx, _ in FrameAnalyzer.color_clusters(frame, bg):
            _enq(cy, cx)
        for y, x in self.effective:
            _enq(y, x)
        ys, xs = np.where(frame != bg)
        if len(ys) > 0:
            idxs = np.arange(len(ys))
            np.random.shuffle(idxs)
            for i in idxs[:40]:
                _enq(int(ys[i]), int(xs[i]))

    def next_click(self, state_hash: str) -> Optional[Tuple[int, int]]:
        while self.queue:
            y, x = self.queue.popleft()
            if (y, x, state_hash) not in self.tried:
                self.tried.add((y, x, state_hash))
                return (y, x)
        return None

    def record(self, y: int, x: int, changed: bool):
        if changed:
            self.effective.append((y, x))

    def reset(self):
        self.tried.clear()
        self.queue.clear()


# ---------------------------------------------------------------------------
# MyAgent — Multi-Strategy Graph Explorer
# ---------------------------------------------------------------------------

class MyAgent(Agent):
    """
    Graph-based BFS exploration agent for ARC-AGI-3.
    """

    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        self.start_time = time.time()
        self.logger = logging.getLogger(f'GE_{self.game_id}')

        self.graph = StateGraph()
        self.click_solver = ClickSolver()

        self.prev_frame: Optional[np.ndarray] = None
        self.prev_hash: Optional[str] = None
        self.prev_aid: Optional[int] = None
        self.prev_click_coords: Optional[Tuple[int, int]] = None
        self.bg_color: int = 0

        self.cur_level = -1
        self.level_actions = 0
        self.level_budget = 600
        self.reset_count = 0
        self.max_resets = 3

        self.nav_path: List[int] = []
        self.simple_aids: List[int] = []
        self.coord_aids: List[int] = []

        self.probing = True
        self.probed: List[int] = []
        self.probe_results: Dict[int, np.ndarray] = {}
        self.initial_frame: Optional[np.ndarray] = None

        self.actions_no_new = 0
        self.stuck_limit = 40
        self.eff_cache: Dict[int, float] = {}

    def _extract(self, fd: FrameData) -> Optional[np.ndarray]:
        try:
            f = np.array(fd.frame, dtype=np.int64)
            if f.ndim == 3:
                f = f[-1]
            return f if f.ndim == 2 else None
        except Exception:
            return None

    def _parse_actions(self, fd: FrameData):
        simple, coord = [], []
        avail = getattr(fd, 'available_actions', None)
        if avail:
            for a in avail:
                v = a.value if hasattr(a, 'value') else int(a)
                if 1 <= v <= 5:
                    simple.append(v)
                elif v in (6, 7):
                    coord.append(v)
        if not simple:
            simple = [1, 2, 3, 4, 5]
        return simple, coord

    def _make_action(self, aid: int, coords=None) -> GameAction:
        m = {
            1: GameAction.ACTION1, 2: GameAction.ACTION2,
            3: GameAction.ACTION3, 4: GameAction.ACTION4,
            5: GameAction.ACTION5, 6: GameAction.ACTION6,
            7: GameAction.ACTION7,
        }
        act = m.get(aid, GameAction.ACTION1)
        if aid in (6, 7) and coords is not None:
            y, x = coords
            act.set_data({'x': int(x), 'y': int(y)})
            act.reasoning = f'Click({x},{y})'
        else:
            act.reasoning = f'A{aid}'
        return act

    def _on_level_change(self, new_level: int):
        for aid in self.simple_aids:
            self.eff_cache[aid] = self.graph.effectiveness(aid)
        self.graph.reset()
        self.click_solver.reset()
        self.cur_level = new_level
        self.level_actions = 0
        self.reset_count = 0
        self.prev_frame = None
        self.prev_hash = None
        self.prev_aid = None
        self.prev_click_coords = None
        self.nav_path = []
        self.probing = True
        self.probed = []
        self.probe_results = {}
        self.initial_frame = None
        self.actions_no_new = 0

    def _probe(self, frame: np.ndarray, sh: str) -> GameAction:
        if self.initial_frame is None:
            self.initial_frame = frame.copy()
            self.bg_color = FrameAnalyzer.background_color(frame)
        for aid in self.simple_aids:
            if aid not in self.probed:
                self.probed.append(aid)
                return self._make_action(aid)
        self.probing = False
        return self._explore(frame, sh)

    def _explore(self, frame: np.ndarray, sh: str) -> GameAction:
        if self.nav_path:
            return self._make_action(self.nav_path.pop(0))

        untried = self.graph.untried(sh, self.simple_aids)
        if untried:
            untried.sort(key=lambda a: self.eff_cache.get(a, 0.5), reverse=True)
            self.actions_no_new = 0
            return self._make_action(untried[0])

        path = self.graph.bfs_to_frontier(sh, self.simple_aids)
        if path and len(path) > 0:
            self.nav_path = path[1:]
            return self._make_action(path[0])

        if self.coord_aids:
            return self._click_action(frame, sh)

        if self.actions_no_new > self.stuck_limit:
            return self._do_reset('stuck')

        if self.simple_aids:
            return self._make_action(random.choice(self.simple_aids))
        return self._do_reset('no_actions')

    def _click_action(self, frame: np.ndarray, sh: str) -> GameAction:
        if not self.click_solver.queue:
            self.click_solver.prepare(frame, sh, self.bg_color)
        coords = self.click_solver.next_click(sh)
        if coords:
            self.prev_click_coords = coords
            return self._make_action(self.coord_aids[0], coords)
        if self.simple_aids:
            return self._make_action(random.choice(self.simple_aids))
        return self._do_reset('no_clicks')

    def _do_reset(self, reason: str) -> GameAction:
        self.reset_count += 1
        self.actions_no_new = 0
        self.prev_frame = None
        self.prev_hash = None
        self.prev_aid = None
        self.nav_path = []
        act = GameAction.RESET
        act.reasoning = reason
        return act

    def is_done(self, frames, latest_frame) -> bool:
        try:
            return latest_frame.state is GameState.WIN or (time.time() - self.start_time) >= 8 * 3600 - 300
        except Exception:
            return True

    def choose_action(self, frames, latest_frame):
        try:
            lv = latest_frame.levels_completed
            if lv != self.cur_level:
                self._on_level_change(lv)
            self.level_actions += 1

            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                return self._do_reset('terminal')

            frame = self._extract(latest_frame)
            if frame is None:
                return self._do_reset('bad_frame')

            s, c = self._parse_actions(latest_frame)
            if s: self.simple_aids = s
            if c: self.coord_aids = c

            sh = hash_frame(frame)
            is_new = sh not in self.graph.all_states
            self.graph.add_state(sh)
            self.actions_no_new = 0 if is_new else self.actions_no_new + 1

            if self.prev_hash is not None and self.prev_aid is not None:
                self.graph.record(self.prev_hash, self.prev_aid, sh)
                if self.prev_aid in (6, 7) and self.prev_click_coords:
                    self.click_solver.record(self.prev_click_coords[0], self.prev_click_coords[1], self.prev_hash != sh)
                    self.prev_click_coords = None

            if self.level_actions > self.level_budget and self.reset_count < self.max_resets:
                return self._do_reset('budget')

            action = self._probe(frame, sh) if self.probing else self._explore(frame, sh)

            self.prev_frame = frame
            self.prev_hash = sh
            self.prev_aid = action.value if hasattr(action, 'value') else None

            return action

        except Exception as e:
            traceback.print_exc()
            act = GameAction.RESET
            act.reasoning = f'crash:{e}'
            return act
