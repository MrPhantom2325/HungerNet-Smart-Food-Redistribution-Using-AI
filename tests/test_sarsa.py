
"""Tests for SARSA agent (subclass of Q-learning)."""

import os
import tempfile

import numpy as np
import pytest

from agents.q_learning import QLearningConfig, discretize_state
from agents.sarsa import SARSAAgent
from sim.environment import FoodRescueEnv


class TestSARSAInheritance:
    def test_inherits_from_qlearning(self):
        from agents.q_learning import QLearningAgent
        agent = SARSAAgent(num_actions=11)
        assert isinstance(agent, QLearningAgent)

    def test_name_is_sarsa(self):
        agent = SARSAAgent(num_actions=11)
        assert agent.name == "sarsa"

    def test_action_selection_works(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        agent = SARSAAgent(num_actions=env.action_space.n, seed=0)
        for _ in range(20):
            a = agent.select_action(env, obs)
            assert 0 <= a < env.action_space.n


class TestSARSAUpdate:
    def test_update_uses_next_action_not_max(self):
        """SARSA should bootstrap from Q(s', next_action), not max."""
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = SARSAAgent(
            num_actions=env.action_space.n,
            config=QLearningConfig(learning_rate=1.0, discount=1.0, optimistic_init=0.0),
            seed=0,
        )
        agent.set_training(True)

        # Manually populate Q-row for the state we'll arrive at, with action 0
        # having a HIGH value and action 5 having a LOW value
        s_before = discretize_state(env)
        action = 0
        obs, reward, term, trunc, _ = env.step(action)
        s_after = discretize_state(env)

        # Plant Q-values for s_after
        agent._ensure_q_row(s_after)[0] = 100.0
        agent._ensure_q_row(s_after)[5] = -100.0

        # Update with next_action=5 (which has Q=-100)
        # SARSA: target = reward + 1.0 * Q(s', 5) = reward + (-100)
        agent.update_from_transition(
            env_before=env, action=action, reward=reward,
            env_after=env, done=False, next_action=5,
        )

        q_before_row = agent._q_table[s_before]
        # Update is: 0 + 1.0 * (reward + (-100) - 0) = reward - 100
        # Specifically the result should be NEGATIVE (much less than reward+100, which
        # is what Q-learning would have produced via max).
        # We just assert it's much less than +50 (which would be the case if max_action was used)
        assert q_before_row[action] < 50.0, (
            f"SARSA update should use next_action=5 (Q=-100), got {q_before_row[action]}"
        )

    def test_no_update_in_eval_mode(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = SARSAAgent(num_actions=env.action_space.n, seed=0)
        agent.set_training(False)
        agent.update_from_transition(env_before=env, action=0, reward=10.0,
                                     env_after=env, done=False, next_action=0)
        assert agent.table_size() == 0


class TestSARSASaveLoad:
    def test_save_load_roundtrip(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = SARSAAgent(num_actions=env.action_space.n, seed=0)

        # Populate a couple entries
        s = discretize_state(env)
        agent._ensure_q_row(s)[2] = 7.5

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sarsa.pkl")
            agent.save(path)
            loaded = SARSAAgent.load(path)
            assert loaded.table_size() == agent.table_size()
            np.testing.assert_array_almost_equal(loaded._q_table[s], agent._q_table[s])
