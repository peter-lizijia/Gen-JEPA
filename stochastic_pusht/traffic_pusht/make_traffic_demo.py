import gymnasium as gym
import gym_pusht  # noqa: F401
import imageio.v2 as imageio
import numpy as np


def _clip_action(action):
    return np.clip(action, 0.0, 512.0).astype(np.float32)


def traffic_action(obs, info, t):
    """Simple visualization policy: red light waits, green light pushes left."""
    agent_pos = obs[:2].astype(np.float32)
    block_pos = obs[2:4].astype(np.float32)
    light = info["light"]

    if t < 30:
        # Keep still for a short warmup to clearly see light switching.
        target = agent_pos
    elif light == "RED":
        target = block_pos + np.array([86.0, 6.0 * np.sin(0.12 * t)], dtype=np.float32)
    else:
        target = block_pos + np.array([-24.0, 0.0], dtype=np.float32)

    return _clip_action(target)


def rollout(video_path="demo_traffic.mp4", steps=220, seed=0):
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode="rgb_array",
    )
    obs, info = env.reset(seed=seed)

    frames = []
    logs = []
    resets = 0

    for t in range(steps):
        action = traffic_action(obs, info, t)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        logs.append((t, info["light"], int(info["remaining_steps"]), bool(info["is_success"]), bool(info["is_fail"])))

        if terminated or truncated:
            resets += 1
            obs, info = env.reset(seed=seed + t + 1)

    env.close()

    if not frames:
        raise RuntimeError("No frames rendered.")
    if frames[0].shape != (224, 224, 3):
        raise RuntimeError(f"Unexpected frame shape: {frames[0].shape}, expected (224, 224, 3)")

    imageio.mimsave(video_path, frames, fps=10)
    print(f"saved: {video_path}")
    print(f"num_frames: {len(frames)}")
    print(f"num_resets: {resets}")
    print("first 12 logs:")
    for row in logs[:12]:
        print(row)


if __name__ == "__main__":
    rollout()
