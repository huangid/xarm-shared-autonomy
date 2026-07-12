"""Convert recorded JSON rollouts into the kNN .npy dataset format.

Reads logs/rollouts/<dir>/episode_XXXX/robot/*.json, keeps only successful
episodes (per meta/stats.json), and writes a single .npy dict the kNN pilot
can load.
"""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--rollout_dir", type=str, required=True,
                    help="Path to logs/rollouts/<dir>")
parser.add_argument("--output", type=str, required=True,
                    help="Output .npy path")
parser.add_argument("--all", action="store_true",
                    help="Include all episodes, not just successful ones")
parser.add_argument("--append", action="store_true",
                    help="Append to existing output instead of overwriting.")
args = parser.parse_args()

rollout = Path(args.rollout_dir)

# Which episodes succeeded — read meta/stats.json.
success_map = {}
meta_stats = rollout / "meta" / "stats.json"
if meta_stats.exists():
    stats = json.loads(meta_stats.read_text())
    for ep_name, info in stats.items():
        success_map[ep_name] = bool(info.get("success", False))

OBS_KEYS = [
    "obs.fingertip_pos", "obs.fingertip_quat", "obs.gripper",
    "obs.fingertip_pos_rel_fixed", "obs.fingertip_pos_rel_held",
    "obs.ee_linvel_fd", "obs.ee_angvel_fd",
]

# Source keys in the JSON (base_action = the pilot/spacemouse command)
ACT_SRC_KEYS = ["base_action.fingertip_pos", "base_action.fingertip_quat", "base_action.gripper"]
# Target keys the kNN format expects
ACT_DST_KEYS = ["action.fingertip_pos", "action.fingertip_quat", "action.gripper"]
ALL_SRC_KEYS = OBS_KEYS + ACT_SRC_KEYS

data = {}
start_idx = 0
if args.append and Path(args.output).exists():
    existing = np.load(args.output, allow_pickle=True).item()
    data = dict(existing)
    start_idx = max(existing.keys()) + 1 if existing else 0
    print(f"Appending to {len(data)} existing episodes.")
ep_dirs = sorted([d for d in rollout.iterdir() if d.name.startswith("episode_")])
kept = 0
for ep_dir in ep_dirs:
    ep_name = ep_dir.name
    if not args.all and success_map and not success_map.get(ep_name, False):
        continue  # skip failed episodes

    robot_dir = ep_dir / "robot"
    step_files = sorted(robot_dir.glob("*.json"))
    if not step_files:
        continue

    buffers = {k: [] for k in ALL_SRC_KEYS}
    for sf in step_files:
        entry = json.loads(sf.read_text())
        if not all(k in entry for k in ALL_SRC_KEYS):
            continue  # skip incomplete timesteps (e.g. terminal step)
        for k in ALL_SRC_KEYS:
            buffers[k].append(entry[k])

    # Build episode dict, renaming base_action.* -> action.*
    ep_dict = {}
    for k in OBS_KEYS:
        ep_dict[k] = np.array(buffers[k], dtype=np.float32)
    for src, dst in zip(ACT_SRC_KEYS, ACT_DST_KEYS):
        ep_dict[dst] = np.array(buffers[src], dtype=np.float32)
    data[start_idx + kept] = ep_dict
    kept += 1

if kept == 0:
    print("WARNING: no episodes kept. Use --all to include unsuccessful ones, "
          "or check that episodes succeeded.")
else:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, data, allow_pickle=True)
    print(f"Added {kept} new episodes. Total now: {len(data)} in {args.output}")