import gymnasium as gym
import gym_pusht
import imageio.v2 as imageio
import numpy as np


def _normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.array([0.0, -1.0], dtype=np.float32)
    return vec / norm


def interactive_action(obs, t, had_contact=False):
    agent_pos = obs[:2].astype(np.float32)
    block_pos = obs[2:4].astype(np.float32)

    # Aim through the block center from the current agent side so contact is sustained.
    approach_dir = _normalize(block_pos - agent_pos)
    perp_dir = np.array([-approach_dir[1], approach_dir[0]], dtype=np.float32)
    through_distance = 70.0 if had_contact else 56.0
    lateral_wobble = 4.0 * np.sin(0.18 * t) + 2.0 * np.sin(0.07 * t + 1.2)
    target = block_pos + approach_dir * through_distance + perp_dir * lateral_wobble

    return np.clip(target, 0.0, 512.0).astype(np.float32)


def observe_action(obs):
    agent_pos = obs[:2].astype(np.float32)
    block_pos = obs[2:4].astype(np.float32)
    retreat_dir = _normalize(agent_pos - block_pos)
    target = block_pos + retreat_dir * 140.0
    return np.clip(target, 0.0, 512.0).astype(np.float32)


def rollout(
    video_path,
    wind_enabled=True,
    wind_speed=200.0,
    steps=120,
    seed=0,
    contact_steps=14,
    pause_steps=10,
    no_action_steps=20,
):
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode="rgb_array",
        wind_enabled=wind_enabled,
        wind_speed=wind_speed,
        wind_warmup_frames=150,
        wind_ramp_frames=50,
        wind_gain=8.0,
    )
    if seed is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(seed=seed)
    frames = []
    logs = []
    prev_action = None
    had_contact = False
    cycle_steps = contact_steps + pause_steps

    for t in range(steps):
        if t < no_action_steps:
            # Keep the agent still for the first N rendered frames.
            desired_action = obs[:2].astype(np.float32)
            action_alpha = 1.0
            phase = "idle"
        else:
            active_t = t - no_action_steps
            in_contact_phase = (active_t % cycle_steps) < contact_steps
            if in_contact_phase:
                desired_action = interactive_action(obs, active_t, had_contact=had_contact)
                action_alpha = 0.25
                phase = "push"
            else:
                # Pause phase: move away from T to let wind effect be visible.
                desired_action = observe_action(obs)
                action_alpha = 0.45
                phase = "observe"

        if prev_action is None:
            action = desired_action
        else:
            action = (1.0 - action_alpha) * prev_action + action_alpha * desired_action
            action = np.clip(action, 0.0, 512.0).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        had_contact = info["n_contacts"] > 0
        prev_action = action
        logs.append(
            (
                t,
                float(action[0]),
                float(action[1]),
                phase,
                int(info["n_contacts"]),
                float(info["wind_direction"]),
                float(info["wind_scale"]),
                float(info["wind_target_vy"]),
            )
        )
        if terminated or truncated:
            # Continue collecting frames across episodes to keep video length stable.
            if seed is None:
                obs, info = env.reset()
            else:
                obs, info = env.reset(seed=seed + t + 1)
            prev_action = None
            had_contact = False

    env.close()
    imageio.mimsave(video_path, frames, fps=10)  # env render_fps=10
    print(f"saved: {video_path}")
    print("first 12 step logs:")
    for row in logs[:12]:
        print(row)
    n_contact_steps = sum(1 for row in logs if row[4] > 0)
    print(f"contact_steps_in_video: {n_contact_steps}/{len(logs)}")


if __name__ == "__main__":
    # Use the same seed for A/B comparison. Set seed=None if you want random initial states each run.
    rollout("demo_no_wind.mp4", wind_enabled=False)
    rollout("demo_with_wind.mp4", wind_enabled=True)
