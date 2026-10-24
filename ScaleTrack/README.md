# ScaleTrack

ScaleTrack provides the pretraining pipeline of Behavior Foundation Models for Humanoid Robots.

This guide covers environment setup, motion preparation, training, evaluation, and policy export. Run all commands from the ScaleTrack root unless stated otherwise.

> [!IMPORTANT]
> ScaleTrack provides a foundational implementation for BFM pretraining; however, training efficiency remains an area for further optimization. As the codebase continues to be consolidated and refined, it may still contain errors, unhandled edge cases, or incomplete documentation. If you encounter an issue or unexpected behavior, please feel free to [open an issue](https://github.com/zengweishuai/ScaleBFM/issues).

## 🧭 Table of contents

- [🛠️ 1. Prepare the environment](#️-1-prepare-the-environment)
- [🦾 2. Prepare asset description files](#-2-prepare-asset-description-files)
- [🎯 3. Prepare retargeted motions](#-3-prepare-retargeted-motions)
- [📦 4. Package motions](#-4-package-motions)
- [🚀 5. Train a policy](#-5-train-a-policy)
- [🎮 6. Play a policy](#-6-play-a-policy)
- [📤 7. Export and check a policy](#-7-export-and-check-a-policy)
- [🙏 8. Acknowledgements](#-8-acknowledgements)

## 🛠️ 1. Prepare the environment

### Python

ScaleTrack requires a **Python 3.11** environment.

```bash
git clone https://github.com/zengweishuai/ScaleBFM.git
cd ScaleBFM/ScaleTrack
# Activate your environment with Python 3.11
```

### Isaac Sim

Install Isaac Sim and verify that it launches:

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

isaacsim # Add `--headless` on a machine without a display. If the process runs as root, also pass `--allow-root`.
```

### IsaacLab

Install IsaacLab from source. We recommend checking out commit `18c7c58d7a6758b6119401945b881e21c8ec0392` to ensure version alignment with ScaleTrack:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 18c7c58d7a6758b6119401945b881e21c8ec0392
./isaaclab.sh -i
```

### Packages

Return to the ScaleTrack root and install the task package and bundled RSL-RL implementation:

```bash
pip install -e source/scaletrack
pip install -e source/my_rsl_rl
```

## 🦾 2. Prepare asset description files

ScaleTrack uses USD robot assets in IsaacLab. The repository provides utilities for converting both MJCF and URDF descriptions to USD.

> [!NOTE]
> The Unitree G1 (29 DoF) assets have already been converted and are ready to use. The bundled MJCF and USD descriptions are located at `source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml` and `source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.usda`, respectively. You can use them directly without running either conversion utility.

### Convert MJCF to USD

```bash
python scripts/pretrain/description_file/convert_mjcf_to_usd.py \
  /path/to/robot.xml \
  /path/to/output/robot.usda \
  --make-instanceable
```

Optional flags include `--headless`, `--fix-base`, and `--import-sites`.

### Convert URDF to USD

```bash
python scripts/pretrain/description_file/convert_urdf_to_usd.py \
  /path/to/robot.urdf \
  /path/to/output/robot.usda \
  --make-instanceable
```

Optional flags include `--headless`, `--fix-base`, `--merge-joints`, `--self-collision`, and `--replace-cylinders-with-capsules`.

For another robot, place the generated USD file and its dependent meshes in the project assets directory, then update the corresponding robot configuration to reference the new asset.

> [!IMPORTANT]
> After conversion, open the generated USD file in Isaac Sim, right-click the worldBody in the sidebar, deactivate, and ctrl+S to save the modification. This step is required to ensure that the asset loads successfully when referenced in IsaacLab.

## 🎯 3. Prepare retargeted motions

ScaleTrack expects robot trajectories that have already been retargeted to the selected embodiment.

The processing code for retargeted motions is naturally compatible with samples obtained by [ScaleRetarget](https://github.com/zengweishuai/ScaleBFM/tree/main/ScaleRetarget). If you adopt other retargeting methods, please ensure that each sample follows the format below:

Each motion must be stored as one `.pkl` file loadable with `joblib.load`:

```python
import joblib

motion = joblib.load("path/to/motion.pkl")
```

The loaded dictionary must contain:

| Key | Shape/type | Description |
| --- | --- | --- |
| `root_pos` | `(N, 3)` | Root translation |
| `root_rot` | `(N, 4)` | Root quaternion in `(x, y, z, w)` order |
| `dof_pos` | `(N, num_joints)` | Joint positions in radians |
| `fps` | `int` or `float` | Source frame rate |

## 📦 4. Package motions

Motion preparation has two steps:

1. Package retargeted trajectories into processed `.npz` motion files.
2. Create a YAML file that tells the scripts which motions to load.

### Package retargeted motions

The packaging utility searches the input directory recursively, resamples every trajectory to a common frame rate, computes velocities, replays the robot in IsaacLab, and saves one processed `.npz` file per motion.

```bash
python scripts/pretrain/data_process/package_motions.py \
  --data_dir /path/to/retargeted_motions \
  --robot_type g1_29dof \
  --output_dir /path/to/processed_motions \
  --output_fps 50 \
  --num_envs 4096 \
  --headless
```

The most useful options are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--data_dir` | required | Directory containing retargeted motion files |
| `--data_format` | `pkl` | Input filename extension |
| `--output_dir` | `source/scaletrack/data/motions` | Directory for processed `.npz` files |
| `--output_fps` | `50` | Output frame rate |
| `--num_envs` | `4096` | Number of motions processed in parallel |
| `--robot_type` | `g1_29dof` | Robot configuration used for replay |
| `--headless` | disabled | Run without opening the simulator GUI |

Reduce `--num_envs` if GPU memory is limited.

### Create a motion YAML file

The `create_yaml.py` recursively finds processed `.npz` files under the supplied directory and writes one YAML file for loading motions:

```bash
python scripts/pretrain/data_process/create_yaml.py \
  --data_dir /path/to/processed_motions \
  --output_path /path/to/motions.yaml
```

## 🚀 5. Train a policy

> [!NOTE]
> If training runs out of GPU memory (OOM), consider reducing the number of simulator environments, for example by adding `--num_envs 4096`.

### Single GPU

```bash
python scripts/pretrain/rsl_rl/train.py \
  --task G1-BFM-Transformer-Tracking \
  --motion_file source/scaletrack/data/yaml_files/train_motions.yaml \
  --run_name my_run \
  --logger wandb \
  --log_project_name ScaleBFM \
  --headless
```

### Multiple GPUs on one node

Set `GPUS_PER_NODE` to the number of GPUs to use:

```bash
python -m torch.distributed.run \
  --nnodes 1 \
  --nproc_per_node "${GPUS_PER_NODE}" \
  scripts/pretrain/rsl_rl/train.py \
  --task G1-BFM-Transformer-Tracking \
  --motion_file source/scaletrack/data/yaml_files/train_motions.yaml \
  --run_name my_distributed_run \
  --log_project_name ScaleBFM \
  --distributed \
  --headless
```

### Multiple GPUs across multiple nodes

Launch the command below on every node. Set `NUM_NODES` to the total number of nodes, `GPUS_PER_NODE` to the number of processes and GPUs used on each node, and `RANK` to that node's unique zero-based index. All nodes must use the same `MASTER_ADDR` and `MASTER_PORT`.

```bash
python -m torch.distributed.run \
  --nnodes "${NUM_NODES}" \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --node_rank "${RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  scripts/pretrain/rsl_rl/train.py \
  --task G1-BFM-Transformer-Tracking \
  --motion_file source/scaletrack/data/yaml_files/train_motions.yaml \
  --run_name my_distributed_run \
  --log_project_name ScaleBFM \
  --distributed \
  --headless
```

> [!IMPORTANT]
> **Motion loading across ranks**
>
> 1. `torchrun` sets `LOCAL_RANK`, `RANK`, and `WORLD_SIZE` for every process. A process with `LOCAL_RANK=i` runs its simulator and policy on `cuda:i`, while `RANK` uniquely identifies it across all nodes.
> 2. Every rank opens the same YAML file supplied to `--motion_file`, preserving the same motion names, paths, and ordering.
> 3. On each node, only `LOCAL_RANK=0` reads the referenced `.npz` files from disk. It loads them in parallel, concatenates their motion arrays, and writes the complete motion library and its metadata to shared CPU memory.
> 4. All ranks wait at a distributed barrier, then attach to the shared memory on their node. Each node maintains one complete shared-memory copy of the motion library, which is reused by all local ranks.
>
> ```text
>                          ┌─────────────────────┐
>                          │  same motions.yaml  │
>                          └──────────┬──────────┘
>                     ┌───────────────┴───────────────┐
>                     ▼                               ▼
>        ┌─────────────────────────────┐   ┌─────────────────────────────┐
>        │           Node 0            │   │           Node 1            │
>        │                             │   │                             │
>        │        LOCAL_RANK=0         │   │        LOCAL_RANK=0         │
>        │     loads motions once      │   │     loads motions once      │
>        │              │              │   │              │              │
>        │              ▼              │   │              ▼              │
>        │   Shared CPU motion copy    │   │   Shared CPU motion copy    │
>        │              │              │   │              │              │
>        │       ┌──────┴──────┐       │   │       ┌──────┴──────┐       │
>        │       ▼             ▼       │   │       ▼             ▼       │
>        │  local rank 0  ... rank N-1 │   │  local rank 0  ... rank N-1 │
>        │     cuda:0       cuda:N-1   │   │     cuda:0       cuda:N-1   │
>        └─────────────────────────────┘   └─────────────────────────────┘
> ```

> [!NOTE]
> **Optional test motions**
>
> To evaluate on a separate motion set during training, create its YAML file with the same utility and pass the optional `--test_motion_file` argument:
>
> ```bash
> --test_motion_file source/scaletrack/data/yaml_files/test_motions.yaml
> ```

> [!WARNING]
> **Distributed online evaluation treats training and test motions differently**
>
> The training set can be too large to evaluate repeatedly on every rank. To improve efficiency, training motions are sorted by length and divided with `motion_range[RANK::WORLD_SIZE]`. Each training motion is therefore evaluated exactly once across all ranks, while the strided assignment spreads short and long clips more evenly to reduce load imbalance and barrier waits. The training YAML must contain at least `WORLD_SIZE` motions; otherwise, at least one rank receives no motion and evaluation stops with an assertion error.
>
> The test set is handled differently. Every rank evaluates the complete test set, then rank zero combines the per-rank results and averages them. These repeated evaluations provide a more accurate and stable estimate of test performance without requiring an extensive repeated evaluation of the much larger training set.

### Fast training with a learned world model (SuperMimic)

`G1-BFM-Transformer-Tracking-SuperMimic` trains the same BFM policy as PPO, following [SuperMimic](https://arxiv.org/abs/2610.28480). The policy uses the same humanoid transformer, observations, rewards, terminations and mode masking. Each update does three things:

1. It collects 2.5 simulator steps on average (update-to-data ratio 0.4), with action noise N(0, 0.1²), into a per-environment replay buffer.
2. It fits a differentiable world model on 8-step autoregressive windows.
3. It back-propagates the return of 32-step imagined rollouts through the world model into the actor (BPTT). A critic bootstraps the return beyond the horizon and is fitted to λ-returns.

Observations, rewards and soft terminations in imagination are computed analytically from the predicted state, using forward kinematics of the robot MJCF. `scripts/supermimic/check_parity.py` checks them against the simulator. Checkpoints keep the PPO format, so the commands in sections 6 and 7 apply unchanged.

**Prepare the data.** Build the flat-ground train/eval YAML files (BONES-SEED filtered with the paper's rules, any4hdmi, LAFAN1). Pack each file into a memory-mappable motion library, so that ~140k clips load in seconds instead of decompressing every `.npz` at start-up:

```bash
python scripts/pretrain/data_process/build_supermimic_manifest.py \
  --datasets_root /path/to/datasets --output_dir /path/to/supermimic_flat
python scripts/pretrain/data_process/pack_motion_library.py \
  --motion_file /path/to/supermimic_flat/train.yaml --output_dir /dev/shm/sm_lib/train
python scripts/pretrain/data_process/pack_motion_library.py \
  --motion_file /path/to/supermimic_flat/eval.yaml --output_dir /dev/shm/sm_lib/eval
# Optional: remove LAFAN clips whose test segments are in the eval set
python scripts/pretrain/data_process/split_lafan_holdout.py \
  --train_library /dev/shm/sm_lib/train --eval_library /dev/shm/sm_lib/eval \
  --train_manifest /path/to/supermimic_flat/train.yaml --output_dir /path/to/holdout
```

**Train.** For this task, `--motion_file` and `--test_motion_file` take packed library directories. Multi-GPU launching works as above:

```bash
python -m torch.distributed.run --nnodes 1 --nproc_per_node "${GPUS_PER_NODE}" \
  scripts/pretrain/rsl_rl/train.py \
  --task G1-BFM-Transformer-Tracking-SuperMimic \
  --motion_file /dev/shm/sm_lib/train \
  --test_motion_file /dev/shm/sm_lib/eval \
  --run_name my_supermimic_run --logger tensorboard --distributed --headless \
  agent.supermimic.cuda_graph=true \
  agent.supermimic.budget_mode=hours agent.supermimic.max_train_hours=2.75 \
  agent.supermimic.eval_interval_updates=1000
```

The main options are in `agents/rsl_rl_supermimic_cfg.py`:

| Option | Meaning |
|---|---|
| `budget_mode`, `max_train_hours` / `max_iterations` | Training budget, in wall-clock hours or in updates. The learning rate follows a cosine decay to 0.03× over the first 78% of the policy phase, then stays constant. Choose a budget that the run actually reaches; otherwise the decay never happens. |
| `cuda_graph` | Replays the actor pass (imagination forward and backward) as one CUDA graph. |
| `imagination_batch_size`, `wm_batch_size` | Imagined rollouts and world-model windows per GPU. |
| `actor_checkpointing`, `actor_micro_batch_size` | Memory savers for large batches on small GPUs. |
| `lr_scale_with_world_size` | Scales the learning rates by √(number of GPUs), as in the paper. |
| `eval_interval_updates` / `eval_interval_minutes` | Test-set evaluation and checkpointing period. |

**Speed.** The imagination is fused with `torch.compile`, but it is still launch bound. On an RTX 5090, one policy update contains 0.19 s of GPU kernels within 0.64 s of wall time. `cuda_graph=true` removes that host overhead. Two environment variables help further:

- `SCALETRACK_COMPILE_CACHE_DIR` stores the compiled artifacts per GPU, keyed by source code, configuration and shapes, so that later jobs skip the ~10 min compilation.
- `SCALETRACK_NCCL_LIB` preloads a newer NCCL. With the NCCL bundled with Isaac Sim, more than 2 ranks hit illegal memory accesses on RTX 5090.

Compared with PPO (17.0 s per iteration on 8 RTX 5090), a SuperMimic training step with the CUDA graph takes 0.65 s (16 RTX 5090): **26× faster per training step**.

## 🎮 6. Play a policy

Load a saved checkpoint and replay a motion set in simulation:

### General command

```bash
python scripts/pretrain/rsl_rl/play.py \
  --task G1-BFM-Transformer-Tracking \
  --load_run my_run \
  --checkpoint model_1000.pt \
  --motion_file source/scaletrack/data/yaml_files/play_motions.yaml \
  --num_envs 1
```

To record a video, add `--video --video_length 500`; recordings are saved under the loaded run directory.

To select a control mode, add `--mode_index INDEX`. The selected mode is visualized with its corresponding activated links. If not specified, playback defaults to Whole-Body Mode. See the following table for details.


| Mode index | Control mode | # Links | Activated links |
| ---: | --- | ---: | --- |
| `0` | Root Mode | 1 | Pelvis |
| `1` | Bimanual Mode | 2 | Left Hand, Right Hand |
| `2` | Root-and-Hand Mode | 3 | Pelvis, Left Hand, Right Hand |
| `3` | End-Effector Mode | 4 | Left Hand, Right Hand, Left Foot, Right Foot |
| `4` | Root-and-End-Effector Mode | 5 | Pelvis, Left Hand, Right Hand, Left Foot, Right Foot |
| `5` | Upper-Body Mode | 6 | Left Shoulder, Right Shoulder, Left Elbow, Right Elbow, Left Hand, Right Hand |
| `6` | Root-and-Upper-Body Mode | 7 | Pelvis, Left Shoulder, Right Shoulder, Left Elbow, Right Elbow, Left Hand, Right Hand |
| `7` | Whole-Body Mode | 14 | Pelvis, Left Shoulder, Right Shoulder, Left Elbow, Right Elbow, Left Hand, Right Hand, Torso, Left Hip, Right Hip, Left Knee, Right Knee, Left Foot, Right Foot |

To use local rather than global tracking, add `--local_tracking` together with `--mode_index INDEX`. The selected mode must include the configured root/anchor link; otherwise, playback stops with an error. At every step, playback calculates observations using the current reference-motion root position in place of the simulated robot root position. The reference markers are translated to the simulated robot root so the visualized target follows the robot while preserving the reference pose.

> [!NOTE]
> **Pretrained checkpoints**
>
> Download the pretrained checkpoints from [Huggingface](https://huggingface.co/WeishuaiZeng/ScaleBFM/tree/main/checkpoint), then place them under `logs/rsl_rl/g1_bfm_tracking_exp` with the following structure:
>
> ```text
> logs/rsl_rl/g1_bfm_tracking_exp/
> ├── humanoid_transformer_m/
> │   └── model_22200.pt
> └── humanoid_transformer_xl/
>     └── model_22200.pt
> ```
>
> The run directory is passed to `--load_run`, and the checkpoint filename is passed to `--checkpoint`.

### Relatively Small model

```bash
python scripts/pretrain/rsl_rl/play.py \
  --task G1-BFM-Transformer-Tracking \
  --load_run humanoid_transformer_m \
  --checkpoint model_22200.pt \
  --motion_file source/scaletrack/data/example/example.yaml \
  --num_envs 1
```

### Relatively Large model

The XL checkpoint requires policy dimensions that match the architecture used during training:

```bash
python scripts/pretrain/rsl_rl/play.py \
  --task G1-BFM-Transformer-Tracking \
  --load_run humanoid_transformer_xl \
  --checkpoint model_22200.pt \
  --motion_file source/scaletrack/data/example/example.yaml \
  --num_envs 1 \
  agent.policy.embedding_dim=384 \
  agent.policy.num_heads=6 \
  agent.policy.ff_dim=384 \
  agent.policy.num_layers=6
```


## 📤 7. Export and check a policy

> [!NOTE]
> ScaleTrack currently supports exporting only the Humanoid Transformer policy. To export another model architecture, you may follow the same workflow and adapt the policy wrapper and onboard reconstruction to your model.

ScaleTrack provides two export scripts:

- `play_export_check_humanoid_transformer.py` launches Isaac Sim, obtains the robot and policy configuration from the task, exports the TensorRT model and deployment metadata, and checks the exported policy in simulation.
- `play_export_check_humanoid_transformer_onboard.py` recompiles the policy directly on the deployment computer without launching Isaac Sim. This is for the aarch64 robot computer because the TensorRT model compiled on a Linux x86_64 workstation cannot be transferred and executed there directly. The script reconstructs the policy and robot kinematics from a checkpoint, saved mode table, export metadata, and MJCF file.

### Export and check in Isaac Sim


#### General command

```bash
python scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py \
  --task G1-BFM-Transformer-Tracking \
  --load_run my_run \
  --checkpoint model_1000.pt \
  --motion_file source/scaletrack/data/example/example.yaml \
  --num_envs 1
```

Add `--headless` to export and run the simulator comparison without opening the Isaac Sim GUI.

The script writes `mode_table.pt`, `<checkpoint>_tensorrt.pt`, and `<checkpoint>_tensorrt_metadata.json` beside the loaded checkpoint under `logs/rsl_rl/<experiment>/<load_run>/`.


#### Export the provided humanoid_transformer_m

```bash
python scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py \
  --task G1-BFM-Transformer-Tracking \
  --load_run humanoid_transformer_m \
  --checkpoint model_22200.pt \
  --motion_file source/scaletrack/data/example/example.yaml \
  --num_envs 1
```

#### Export the provided humanoid_transformer_xl

```bash
python scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py \
  --task G1-BFM-Transformer-Tracking \
  --load_run humanoid_transformer_xl \
  --checkpoint model_22200.pt \
  --motion_file source/scaletrack/data/example/example.yaml \
  --num_envs 1 \
  agent.policy.embedding_dim=384 \
  agent.policy.num_heads=6 \
  agent.policy.ff_dim=384 \
  agent.policy.num_layers=6
```

> [!NOTE]
> If deployment runs on the same Linux system and CUDA device used for compilation, the exported model can be used directly without recompilation.

### Recompile onboard for aarch64

TensorRT artifacts are platform-specific. Do not copy the `<checkpoint>_tensorrt.pt` file compiled on the Linux x86_64 workstation to the aarch64 robot computer. Instead:

1. Generate `mode_table.pt` with `play_export_check_humanoid_transformer.py` on the workstation.
2. Transfer the original PyTorch checkpoint, `mode_table.pt`, `<checkpoint>_tensorrt_metadata.json`, and the robot MJCF file to the aarch64 computer.
3. Run `play_export_check_humanoid_transformer_onboard.py` on the aarch64 computer to compile a new TensorRT model against its local CUDA, TensorRT, PyTorch, and Torch-TensorRT installation.

```bash
python scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer_onboard.py \
  --checkpoint path/to/checkpoint.pt \
  --mode_table path/to/mode_table.pt \
  --metadata path/to/checkpoint_tensorrt_metadata.json \
  --xml_path path/to/robot.xml
```

The onboard script writes the compiled model next to the supplied checkpoint. The metadata supplies the policy dimensions, transformer architecture, observation layout, joint defaults, action scales, future offsets, selected bodies, and joint ordering, allowing both provided model sizes to be reconstructed without hard-coded model or G1 configuration values. The resulting artifact is compiled for the onboard platform.

## 🙏 8. Acknowledgements

ScaleTrack is built on [Isaac Sim](https://developer.nvidia.com/isaac/sim), [IsaacLab](https://github.com/isaac-sim/IsaacLab), [RSL-RL](https://github.com/leggedrobotics/rsl_rl), and [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking). We thank their authors and contributors for their open-source work.
