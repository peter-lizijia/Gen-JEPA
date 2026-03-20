import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import gym_pusht  # noqa: F401


@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        ("PushT-v0", "state"),
        ("PushT-v0", "pixels"),
        ("PushT-v0", "pixels_agent_pos"),
        ("PushT-v0", "environment_state_agent_pos"),
    ],
)
def test_env(env_task, obs_type):
    env = gym.make(f"gym_pusht/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)


def test_pixels_default_shape_and_light_info():
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    rendered = env.render()

    assert obs.shape == (224, 224, 3)
    assert rendered.shape == (224, 224, 3)
    assert info["light"] in {env.unwrapped.LIGHT_GREEN, env.unwrapped.LIGHT_RED}
    assert info["remaining_steps"] >= 1

    env.close()


def test_red_light_road_entry_is_failure():
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    env.reset(seed=0)
    unwrapped = env.unwrapped
    road = unwrapped.road_rect
    state = np.array(
        [
            300.0,
            256.0,
            0.5 * (road[0] + road[1]),
            0.5 * (road[2] + road[3]),
            0.0,
        ],
        dtype=np.float64,
    )
    unwrapped._set_state(state)
    unwrapped.light = unwrapped.LIGHT_RED
    unwrapped.remaining_steps = 10

    _, reward, terminated, truncated, info = env.step(np.array([300.0, 256.0], dtype=np.float32))

    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    assert info["is_fail"] is True
    assert info["is_success"] is False

    env.close()
