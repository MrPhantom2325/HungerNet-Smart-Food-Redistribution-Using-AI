"""
FoodRescueEnv: Gymnasium-compatible environment for food rescue RL.

Episode flow
------------
1. reset() loads a fresh scenario, spawns vehicles, resets all state.
2. Each step():
   - Decode the action: head_to_donor_i / head_to_shelter_j / idle
   - Apply to the current vehicle (round-robin selection)
   - Advance the world: vehicle moves, donors generate batches, batches age,
     shelters grow demand, deliveries happen on arrival
   - Compute reward, build observation, check termination

Observation space
-----------------
A flat Box of floats representing:
  [vehicle_x, vehicle_y, vehicle_load_pct, vehicle_idle_flag,
   for each donor: (qty_pending, min_shelf_life, distance_from_vehicle),
   for each shelter: (current_demand_pct, distance_from_vehicle),
   normalized_time, current_vehicle_idx]

Action space
------------
Discrete: head_to_donor_0 ... head_to_donor_{N-1},
          head_to_shelter_0 ... head_to_shelter_{M-1},
          idle/wait
Total: N + M + 1 actions.

Reward
------
Dense per-step reward combining four objectives:
  + alpha * food_delivered_this_step      (delivery)
  - beta  * food_spoiled_this_step        (anti-spoilage)
  - gamma * distance_traveled_this_step   (transport cost / emissions)
  - delta * unmet_demand_this_step        (equity / shelter coverage)

Reward weights (alpha, beta, gamma, delta) come from the env config and become
hyperparameters that we sweep in Sprint 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.city import ScenarioLoader, make_vehicles
from sim.entities import BatchStatus, FoodBatch


# -----------------------------
# Config
# -----------------------------

@dataclass
class RewardWeights:
    """Per-event reward magnitudes. Tunable, swept in Sprint 6."""
    delivery: float = 10.0
    spoilage: float = 5.0
    distance: float = 0.1
    unmet_demand: float = 1.0
    priority_bonus: float = 0.5  # extra reward for delivering to priority-1 shelters
    oversupply_penalty: float = 0.3  # small penalty for delivering more than needed


@dataclass
class EnvConfig:
    """Top-level env configuration."""
    scenario_name: str = "weekday"
    vehicle_start_strategy: str = "center"
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    max_episode_steps: Optional[int] = None  # None means use scenario's episode_length
    seed: Optional[int] = None  # None means use scenario's random_seed


# -----------------------------
# Environment
# -----------------------------

class FoodRescueEnv(gym.Env):
    """
    Gymnasium env for food rescue dispatch.

    A single agent controls a fleet of vehicles in round-robin fashion: at each
    step, exactly one vehicle (selected by current_vehicle_idx) receives the
    action. The env then advances world state by one timestep.

    Use case:
        env = FoodRescueEnv()  # uses defaults (weekday scenario)
        obs, info = env.reset(seed=42)
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.config = config if config is not None else EnvConfig()
        self.render_mode = render_mode

        # Load scenario template — this gives us shapes for the spaces.
        # Each reset() reloads to get fresh entity instances.
        self._loader = ScenarioLoader()
        scenario_template = self._loader.load(self.config.scenario_name)
        self.num_donors = scenario_template.num_donors
        self.num_shelters = scenario_template.num_shelters
        self.num_vehicles = scenario_template.num_vehicles
        self.grid_size = scenario_template.city.grid_size

        self.max_episode_steps = (
            self.config.max_episode_steps
            if self.config.max_episode_steps is not None
            else scenario_template.city.episode_length
        )

        # Define spaces
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

        # Per-episode state. Real values populated in reset().
        self.scenario = None
        self.vehicles: list = []
        self.batches: list[FoodBatch] = []
        self.current_step: int = 0
        self.current_vehicle_idx: int = 0
        self.next_batch_id: int = 0
        self.rng: Optional[np.random.Generator] = None
        self._episode_metrics: dict = {}
        self._last_step_info: dict = {}  # raw per-step counters used for reward + info

    # -----------------------------
    # Space builders
    # -----------------------------

    def _build_action_space(self) -> spaces.Discrete:
        """N donors + M shelters + 1 idle = N+M+1 discrete actions."""
        return spaces.Discrete(self.num_donors + self.num_shelters + 1)

    def _build_observation_space(self) -> spaces.Box:
        """
        Observation vector layout (all floats):

          [0] vehicle_x                       in [0, 1]   (normalized by grid_size)
          [1] vehicle_y                       in [0, 1]
          [2] vehicle_load_pct                in [0, 1]
          [3] vehicle_idle_flag               in {0, 1}
          For each donor i:
            [_] qty_pending_normalized        in [0, ~5]  (capped soft, can spike)
            [_] min_shelf_life_normalized     in [0, 1]   (1 = fresh, 0 = expiring)
            [_] distance_from_vehicle_norm    in [0, 1]   (normalized by 2*grid_size)
          For each shelter j:
            [_] current_demand_pct            in [0, 1]
            [_] distance_from_vehicle_norm    in [0, 1]
          [_] normalized_time                 in [0, 1]
          [_] current_vehicle_idx_normalized  in [0, 1]
        """
        n_features = 4 + 3 * self.num_donors + 2 * self.num_shelters + 2
        return spaces.Box(
            low=0.0,
            high=10.0,  # generous upper bound; most features are [0, 1]
            shape=(n_features,),
            dtype=np.float32,
        )

    # -----------------------------
    # Action decoding
    # -----------------------------

    def _decode_action(self, action: int) -> tuple[str, Optional[int]]:
        """
        Decode the integer action into a (kind, index) tuple.

        Returns
        -------
        kind : str
            "donor", "shelter", or "idle"
        index : int | None
            Index into self.scenario.donors or self.scenario.shelters, or None for idle.
        """
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Action {action} out of range [0, {self.action_space.n})")

        if action < self.num_donors:
            return "donor", action
        if action < self.num_donors + self.num_shelters:
            return "shelter", action - self.num_donors
        return "idle", None

    # -----------------------------
    # Observation
    # -----------------------------

    def _get_observation(self) -> np.ndarray:
        """Build the observation vector for the *current* vehicle."""
        v = self.vehicles[self.current_vehicle_idx]
        scn = self.scenario
        gs = self.grid_size

        obs = []

        # Vehicle features
        obs.append(v.location[0] / gs)
        obs.append(v.location[1] / gs)
        obs.append(v.current_load() / v.capacity if v.capacity > 0 else 0.0)
        obs.append(1.0 if v.is_idle() else 0.0)

        # Per-donor features
        for d in scn.donors:
            qty = d.total_pending_quantity()
            qty_norm = qty / max(d.avg_quantity, 1.0)  # 1.0 ~= one batch's worth
            min_sl = d.min_pending_shelf_life()
            sl_norm = min(min_sl / d.shelf_life_max, 1.0) if d.shelf_life_max > 0 else 1.0
            dist = self._manhattan(v.location, d.location)
            dist_norm = dist / (2 * gs)
            obs.extend([qty_norm, sl_norm, dist_norm])

        # Per-shelter features
        for s in scn.shelters:
            obs.append(s.utilization())  # already in [0, 1]
            dist = self._manhattan(v.location, s.location)
            obs.append(dist / (2 * gs))

        # Time + which vehicle
        obs.append(self.current_step / self.max_episode_steps)
        obs.append(
            self.current_vehicle_idx / max(self.num_vehicles - 1, 1)
            if self.num_vehicles > 1 else 0.0
        )

        return np.array(obs, dtype=np.float32)

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # -----------------------------
    # Reset
    # -----------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Start a new episode.

        Loads a fresh scenario (new Donor/Shelter/Vehicle instances), seeds the
        RNG, and zeroes all metrics. Returns initial observation.
        """
        super().reset(seed=seed)

        # Choose seed: explicit > config > scenario default
        if seed is None:
            seed = self.config.seed
        if seed is None:
            scenario_template = self._loader.load(self.config.scenario_name)
            seed = scenario_template.random_seed

        self.rng = np.random.default_rng(seed)

        # Fresh scenario, fresh vehicles
        self.scenario = self._loader.load(self.config.scenario_name)
        self.vehicles = make_vehicles(
            self.scenario, start_strategy=self.config.vehicle_start_strategy
        )

        self.batches = []
        self.next_batch_id = 0
        self.current_step = 0
        self.current_vehicle_idx = 0

        self._episode_metrics = {
            "total_generated": 0,
            "total_delivered_units": 0.0,
            "total_spoiled_units": 0.0,
            "total_wasted_units": 0.0,
            "total_distance": 0,
            "total_unmet_demand_steps": 0.0,
            "deliveries_count": 0,
            "priority_deliveries_count": 0,
        }
        self._last_step_info = {}

        obs = self._get_observation()
        info = {
            "scenario": self.scenario.name,
            "current_vehicle_idx": self.current_vehicle_idx,
            "step": self.current_step,
        }
        return obs, info

    # -----------------------------
    # Step (placeholder — full implementation in Step 8)
    # -----------------------------

    def step(self, action: int):
        """Apply action, advance world. Implemented in Step 8."""
        raise NotImplementedError(
            "step() will be implemented in Step 8. "
            "This stub ensures the env structure is in place for Step 7's tests."
        )

    # -----------------------------
    # Render (placeholder — full implementation in Sprint 3)
    # -----------------------------

    def render(self):
        """Render the env. Implemented in Sprint 3."""
        if self.render_mode is None:
            return None
        raise NotImplementedError("Rendering is implemented in Sprint 3.")

    def close(self):
        pass
