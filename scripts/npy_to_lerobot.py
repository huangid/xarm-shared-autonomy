"""Convert a kNN-format .npy demo set (see convert_demos.py) into a LeRobot
dataset suitable for BC/DiffusionPolicy training via scripts/train_bc.sh.

Input .npy is a dict: {episode_idx: {key: (T, D) array, ...}} with keys
obs.fingertip_pos, obs.fingertip_quat, obs.gripper, obs.fingertip_pos_rel_fixed,
obs.fingertip_pos_rel_held, obs.ee_linvel_fd, obs.ee_angvel_fd,
action.fingertip_pos, action.fingertip_quat, action.gripper.

Output is a LeRobotDataset with the same 14D observation.state / 6D
observation.environment_state / 8D action layout used by
source/pilot_models/config/state_dp_cfg.json and
xarm_env.py::_get_bc_pilot_action, so it works unchanged for any task.

Usage:
    python scripts/npy_to_lerobot.py \
        --input logs/data/threeblocks_seq_demos.npy \
        --repo_id <hf_user>/threeblocks_expert_50 \
        --task "stack three blocks in the bin" \
        --push_to_hub
"""
import argparse

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 15  # sim.dt (1/120) * decimation (8) in xarm_env_cfg.py -> 15 Hz control rate

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
        "names": {
            "axes": [
                "fingertip_x", "fingertip_y", "fingertip_z",
                "fingertip_qw", "fingertip_qx", "fingertip_qy", "fingertip_qz",
                "gripper",
                "linvel_x", "linvel_y", "linvel_z",
                "angvel_x", "angvel_y", "angvel_z",
            ]
        },
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": [
                "rel_fixed_x", "rel_fixed_y", "rel_fixed_z",
                "rel_held_x", "rel_held_y", "rel_held_z",
            ]
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "axes": [
                "x", "y", "z", "qw", "qx", "qy", "qz", "gripper",
            ]
        },
    },
}


def episode_to_frames(ep: dict):
    state = np.concatenate([
        ep["obs.fingertip_pos"],
        ep["obs.fingertip_quat"],
        ep["obs.gripper"],
        ep["obs.ee_linvel_fd"],
        ep["obs.ee_angvel_fd"],
    ], axis=-1).astype(np.float32)

    env_state = np.concatenate([
        ep["obs.fingertip_pos_rel_fixed"],
        ep["obs.fingertip_pos_rel_held"],
    ], axis=-1).astype(np.float32)

    action = np.concatenate([
        ep["action.fingertip_pos"],
        ep["action.fingertip_quat"],
        ep["action.gripper"],
    ], axis=-1).astype(np.float32)

    assert state.shape[-1] == 14 and env_state.shape[-1] == 6 and action.shape[-1] == 8, (
        f"unexpected shapes: state={state.shape}, env_state={env_state.shape}, action={action.shape}"
    )

    for t in range(state.shape[0]):
        yield {
            "observation.state": state[t],
            "observation.environment_state": env_state[t],
            "action": action[t],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to kNN-format .npy demo file.")
    parser.add_argument("--repo_id", type=str, required=True, help="Output LeRobot dataset repo id.")
    parser.add_argument("--root", type=str, default=None, help="Local root dir (default: HF cache).")
    parser.add_argument("--task", type=str, required=True, help="Natural-language task description.")
    parser.add_argument("--robot_type", type=str, default="xarm7")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    args = parser.parse_args()

    demos = np.load(args.input, allow_pickle=True).item()

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=FEATURES,
        root=args.root,
        robot_type=args.robot_type,
        use_videos=False,
    )

    for ep_idx in sorted(demos.keys()):
        for frame in episode_to_frames(demos[ep_idx]):
            frame["task"] = args.task
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"Saved episode {ep_idx} ({demos[ep_idx]['obs.fingertip_pos'].shape[0]} steps).")

    print(f"Converted {len(demos)} episodes into LeRobot dataset at {dataset.root}")

    if args.push_to_hub:
        dataset.push_to_hub()
        print(f"Pushed to hub: {args.repo_id}")


if __name__ == "__main__":
    main()
