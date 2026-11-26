# parkour_training — Go2 parkour teacher/student

Upstream: [CAI23sbP/Isaaclab_Parkour](https://github.com/CAI23sbP/Isaaclab_Parkour),
an IsaacLab port of Extreme-Parkour.

| Directory | Observation input |
|---|---|
| `parkour_depth/` | Depth camera (main line, adopted) |
| `parkour_lidar/` | 3D LiDAR (hedge against the A2 Pro lacking a depth sensor) |

The two trees differ in **six files only**:

```
parkour_tasks/.../go2/agents/parkour_rl_cfg.py
parkour_tasks/.../go2/parkour_mdp_cfg.py
parkour_tasks/.../go2/parkour_student_cfg.py              (LIDAR_CFG)
scripts/rsl_rl/modules/distillation_with_extractor.py
scripts/rsl_rl/modules/feature_extractors/depth_backbone.py   (LidarScanBackbone)
scripts/rsl_rl/modules/feature_extractors/__init__.py
```

## Training environment

Teacher training ran **only in the `parkour_il` conda env** (rsl-rl-lib 2.3.3).
`env_isaaclab` (3.1.2) crashes on API drift.

Note: that environment has since been deleted — see the root README.

## Teacher observation spec (753-dim)

```
proprio 53 + scandot 132 + priv_explicit 9 + priv_latent 29 (ignored) + history 530
```

The scandot grid is 12×11. It is **not a real sensor** — at submission time it has to
be synthesised by an adapter.

## Run lineage

Training logs live on the server; this repository holds code only.

| Run | Meaning |
|---|---|
| `2026-06-24_10-58-09_학습성공_0.45m` | 0.45 m step climb succeeded (the Korean suffix reads "training succeeded") |
| `2026-08-05_01-38-20` | **parkour teacher** (RMA, no distillation), `model_4200.pt` |
| `2026-08-28_13-13-12_teacher_overspeed` | teacher with overspeed penalty |
| `2026-09-01_08-37-52_encoder_depth` | depth encoder (adopted, latent 0.0257) |
| `2026-09-04_06-50-41_encoder_lidar_front` | front mount (offset 0.338/0/0.081) |

Suffixes: `teacher_*`/`student_*` (distillation), `encoder_*` (perception encoder),
`sanity_*` (pre-flight check), `_ft` (fine-tune), `_smoke` (smoke test).
