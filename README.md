# Efficient and Reliable Teleoperation through Real2Sim2Real Shared Autonomy

## Links

- **Code:** https://github.com/shuosha/Residual_Copilot
- **Paper:** https://arxiv.org/abs/2603.22787

---

## Table of Contents

- [Installation](#installation)
- [Quick Demo](#quick-demo)
- [Inference](#inference)
  - [Evaluating Pilot and Copilot Policies](#evaluating-pilot-and-copilot-policies)
  - [Visualizing Residual Corrections](#visualizing-residual-corrections)
  - [Real-world Deployment](#real-world-deployment)
- [Training](#training)
  - [Training Residual Copilot](#training-residual-copilot)
  - [Training Guided Diffusion Copilot](#training-guided-diffusion-copilot)
  - [Adding New Tasks and Models](#adding-new-tasks-and-models)
- [Three Blocks: SpaceMouse Teleop Workflow](#three-blocks-spacemouse-teleop-workflow)
- [HuggingFace Collection](#huggingface-collection)

## Installation

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA 12.8 (matches IsaacSim 5.1.0 / PyTorch 2.7.0 cu128; for other versions see the [IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))

### Setup

```bash
git clone --recurse-submodules https://github.com/shuosha/Residual_Copilot.git
cd Residual_Copilot
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
UV_HTTP_TIMEOUT=300 uv sync   # Isaac Sim wheels are large; extended timeout prevents download failures
```

To verify that all environments are registered correctly:
```bash
python scripts/list_envs.py   # should list XArm-{GearMesh,NutThread,PegInsert}-{Residual,GuidedDiffusion}
```

> **First run:** Isaac Sim will prompt you to accept the EULA and will download/cache simulator assets, which may take several minutes.

> **Note:** Isaac Sim can exhaust static TLS slots, causing `cannot allocate memory in static TLS block`. Preload `libgomp` before running any script:
> ```bash
> export LD_PRELOAD=/lib/x86_64-linux-gnu/libgomp.so.1
> ```
> The provided shell scripts handle this automatically.

## Quick Demo

Generate side-by-side comparison videos of the **Residual Copilot** vs. an unassisted **Replay** baseline:

```bash
bash scripts/demo_side_by_side.sh NutThread              # launches with Isaac Sim viewer
bash scripts/demo_side_by_side.sh GearMesh --headless   # no viewer (for remote machines or no display)
bash scripts/demo_side_by_side.sh PegInsert --no-clean  # keep intermediate files
```

Output: `logs/demos/<task>/demo_*.mp4` — annotated videos with action overlays (red = base action, pink = residual, blue = net).

## Inference

### Evaluating Pilot and Copilot Policies

```bash
# Pilot + copilot with recording
python scripts/play.py \
  --task NutThread --pilot kNNPilot --copilot ResidualCopilot \
  --num_envs 16 --record

# Pilot only
python scripts/play.py --task NutThread --pilot kNNPilot --num_envs 16
```

**Arguments:** `--task` (`GearMesh` / `GearMeshIntent` / `PegInsert` / `NutThread` / `ThreeBlocks`), `--pilot` (see below), `--copilot` (optional), `--num_envs` (default 1), `--record` (save to `logs/rollouts/`), `--no_rand` (disable domain randomization), `--vis_obs` (show held/target visualization markers), `--checkpoint` (override the HF copilot download with a local path; applies to RL-Games copilots `ResidualBC` / `ResidualCopilot` as well as guided-diffusion copilots `GuidedDiffusionBC` / `GuidedDiffusionExpert` / `DiSCo`).

`GearMeshIntent` is a gear-mesh variant where each demo episode targets one of three gear slots (small/medium/large); `ThreeBlocks` is a multi-goal block-sorting task (pick up to 3 blocks into a bin) primarily driven by `SpaceMousePilot` for live shared-autonomy teleop.

**Pilots:**

| Name | Description |
|------|-------------|
| `kNNPilot` | kNN Pilot |
| `BCPilot` | Teleop BC policy |
| `ExpertPilot` | Expert BC policy |
| `NoisyPilot` | Expert BC + mistakeful behaviors (noisy actions) |
| `LaggyPilot` | Expert BC + laggy behaviors (repeat actions) |
| `ReplayPilot` | Replay recorded episodes |
| `SpaceMousePilot` | Live human teleop via a 3Dconnexion SpaceMouse (real HID device required) |

**Copilots:**

| Name | Description |
|------|-------------|
| `ResidualCopilot` | Residual RL trained with Residual Copilot |
| `ResidualBC` | Residual RL trained with BC  pilot |
| `GuidedDiffusionBC` | Guided Diffusion trained on teleop data |
| `GuidedDiffusionExpert` | Guided Diffusion trained on expert data |
| `DiSCo` | Guided-diffusion copilot with DiSCo sampling (seed/inpaint toward the pilot's reference action) + beta-blend; used for live shared-autonomy assistance, e.g. with `SpaceMousePilot` on `ThreeBlocks` |

All checkpoints are auto-downloaded from HuggingFace on first use. To evaluate a **local** RL-Games copilot checkpoint instead, pass `--checkpoint <path/to/FactoryXarm.pth>`:

```bash
python scripts/play.py \
  --task GearMeshIntent --pilot NoisyPilot --copilot ResidualCopilot \
  --checkpoint logs/rl_games/FactoryXarm/<run_name>/nn/FactoryXarm.pth \
  --num_envs 15 --no_rand --record
```

Combined with `--no_rand`, envs are run in deterministic order (one episode per env), so `--num_envs N` evaluates exactly `N` ordered episodes.

For guided-diffusion copilots (`GuidedDiffusionBC`/`GuidedDiffusionExpert`/`DiSCo`), `--checkpoint` points at a LeRobot `pretrained_model` directory instead of an RL-Games `.pth`:

```bash
# DiSCo copilot assisting a live SpaceMouse pilot, from a locally-trained checkpoint
python scripts/play.py \
  --task ThreeBlocks --pilot SpaceMousePilot --copilot DiSCo --num_envs 1 \
  --checkpoint outputs/train/threeblocks_expert_50_bc_expert/checkpoints/060000/pretrained_model
```

### Visualizing Residual Corrections

**Per-episode and collage videos** from recorded rollouts:

```bash
# <recording_dir> is created by `play.py --record` under logs/rollouts/,
# named eval_<task>_with_<copilot>_and_<pilot> (e.g. eval_GearMesh_with_ResidualCopilot_and_kNNPilot)

# Annotated single-episode videos
python scripts/vis/to_videos.py \
  logs/rollouts/<recording_dir> \
  --single --annotate

# Collage grid
python scripts/vis/to_videos.py \
  logs/rollouts/<recording_dir> \
  --collage --annotate --cols 4 --scale 0.5
```

When `--annotate` is enabled, action arrows are drawn per frame: **red** = base action, **pink** = residual, **blue** = net action.

### Real-world Deployment

See [Residual_Copilot_Deployment](https://github.com/shuosha/Residual_Copilot_Deployment) for real-world setup, hardware configuration, and deployment instructions.

## Training

### Training Residual Copilot

Train the RL residual copilot using PPO with a pilot model:

```bash
python scripts/train.py \
  --task XArm-GearMesh-Residual \
  --pilot kNNPilot \
  --num_envs 128 \
  --headless
```

**Arguments:** `--task` (full gym ID, e.g. `XArm-GearMesh-Residual`), `--pilot` (pilot model), `--num_envs` (default 128), `--checkpoint` (resume), `--distributed` (multi-GPU), `--track` (W&B logging), `--wandb_project_name` (W&B project name, defaults to task config name), `--wandb_name` (W&B experiment name, defaults to log directory). Logs saved to `logs/rl_games/`.

### Training Guided Diffusion Copilot

Diffusion policies are trained with [LeRobot](https://github.com/huggingface/lerobot). Training data can either come from successful rollout data of an existing policy or real-world data collected using the [deployment repo](https://github.com/shuosha/Residual_Copilot_Deployment).

**1. Collect data:**

```bash
python scripts/collect_data.py \
  --task GearMesh --pilot kNNPilot \
  --num_envs 16 --num_episodes 500 \
  --output_dir logs/data/gearmesh_knn_500 --headless
```

**2. Augment (optional):**

```bash
python scripts/augment_data.py \
  --in logs/data/gearmesh_train.npy --target-total 2000 \
  --pos-aug 0.02 --rot-aug-deg 5
```

**Visualize training data** with fingertip position heatmaps. You can point to either an `.npy` file or a data root directory:

```bash
# From .npy file
python scripts/vis/plot_data.py path/to/camera_image.jpg \
  --npy-path logs/data/gearmesh_train.npy \
  --out logs/vis/gearmesh_heatmap.png

# From data root directory
python scripts/vis/plot_data.py path/to/camera_image.jpg \
  --data-root logs/data/gearmesh_knn_500 \
  --out logs/vis/gearmesh_heatmap.png
```

**3. Train DiffusionPolicy:**

```bash
bash scripts/train_bc.sh <dataset_path> <job_name>
```

The `dataset_path` can be a HuggingFace dataset ID or a locally collected dataset. Ensure the observation and action spaces are consistent with the target task, as defined in [`source/pilot_models/config/state_dp_cfg.json`](source/pilot_models/config/state_dp_cfg.json).

**Available HF datasets:** Expert — `shashuo0104/0126_gearmesh_expert_2000`, `shashuo0104/0129_peginsert_expert_2000`, `shashuo0104/0129_nutthread_expert_2000`. Augmented Teleop — `shashuo0104/0121_gearmesh_teleop_aug_2000`, `shashuo0104/0121_peginsert_teleop_aug_2000`, `shashuo0104/0121_nutthread_teleop_aug_2000`.

### Adding New Tasks and Models

#### New Assembly Task

1. **Create USD assets** for the held and fixed objects (local paths or uploaded to HuggingFace)
2. **Define asset configs** in `assembly_tasks_cfg.py` — subclass `HeldAssetCfg` and `FixedAssetCfg`
3. **Define task config** as `AssemblyTask` subclass — set data paths, rewards, success threshold
4. **Create env config** in `xarm_env_cfg.py` — subclass `XArmEnvCfg`
5. **Register** in `__init__.py` with `gym.register()`
6. **Verify** with `python scripts/list_envs.py`

#### New Pilot Model

1. Implement in `source/pilot_models/` with `get_actions(episode_idx, pos, quat, grip)` (returns an 8D pos+quat+gripper base action) and `clear(env_ids)` methods for retrieval-style pilots (see `knn_pilot.py`), or `act(obs)` / `reset()` for learned policies (see `bc_pilot.py`)
2. Register in `_init_pilot()` in `xarm_env.py`
3. Add mapping in `source/utils/constants.py` (`PILOT_NAME_MAP`)

## Three Blocks: SpaceMouse Teleop Workflow

`ThreeBlocks` is a multi-goal block-sorting task (pick up to 3 blocks into a bin) driven live by a physical [SpaceMouse](https://3dconnexion.com/) via `SpaceMousePilot`, rather than pre-recorded teleop/expert `.npy` data. This is the end-to-end loop for collecting demos, converting them to the kNN dataset format, and training a residual copilot on top.

**1. Drive the task by hand** (sanity-check the SpaceMouse connection and controls):

```bash
python scripts/play.py --task ThreeBlocks --pilot SpaceMousePilot --num_envs 1
```

**2. Record demos:**

```bash
python scripts/play.py --task ThreeBlocks --pilot SpaceMousePilot --num_envs 1 --record
```

**3. Append the recording to your dataset** (`convert_demos.py` reads `logs/rollouts/<dir>`, keeps only successful episodes per `meta/stats.json`, and writes/appends to a single `.npy`):

```bash
python scripts/convert_demos.py \
  --rollout_dir logs/rollouts/eval_ThreeBlocks_with_SpaceMousePilot \
  --output logs/data/threeblocks_seq_demos.npy --append
```

**4. Verify the output loads in kNN format:**

```bash
python -c "import numpy as np; print('episodes:', len(np.load('logs/data/threeblocks_seq_demos.npy', allow_pickle=True).item()))"
```

**5. Test the kNN pilot** against the newly appended demos:

```bash
python scripts/play.py --task ThreeBlocks --pilot kNNPilot --num_envs 1
```

**6. Train the residual copilot** (long-running — disable auto-suspend first if training on a laptop/workstation):

```bash
# disable auto-suspend (long run)
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'

# train
python scripts/train.py --task XArm-ThreeBlocks-Residual --pilot kNNPilot --num_envs 128 --headless
```

**7. Run the trained copilot live**, assisting the SpaceMouse pilot:

```bash
python scripts/play.py \
  --task ThreeBlocks --pilot SpaceMousePilot --copilot ResidualCopilot --num_envs 1 \
  --checkpoint logs/rl_games/FactoryXarm/none/nn/last_FactoryXarm_ep_400_rew_-109.26714.pth
```

**8. Alternative: run a DiSCo guided-diffusion copilot live**, from a BC checkpoint trained on the recorded demos (see [Training Guided Diffusion Copilot](#training-guided-diffusion-copilot) to train one against `threeblocks_seq_demos.npy`):

```bash
python scripts/play.py \
  --task ThreeBlocks --pilot SpaceMousePilot --copilot DiSCo --num_envs 1 \
  --checkpoint outputs/train/threeblocks_expert_50_bc_expert/checkpoints/060000/pretrained_model
```

## HuggingFace Collection

All data, models, and assets are hosted as a [HuggingFace collection](https://huggingface.co/collections/shashuo0104/residual-copilot), auto-downloaded on first use.

| Repo | Contents |
|------|----------|
| [`residual_copilot_assets`](https://huggingface.co/datasets/shashuo0104/residual_copilot_assets) | Robot USD/URDF, object meshes, camera params |
| [`residual_copilot_models`](https://huggingface.co/shashuo0104/residual_copilot_models) | BC pilots, RL copilot checkpoints, DP baselines |
| [`residual_copilot_data`](https://huggingface.co/datasets/shashuo0104/residual_copilot_data) | Teleoperation trajectories |

See each HuggingFace repo for the detailed file structure.
