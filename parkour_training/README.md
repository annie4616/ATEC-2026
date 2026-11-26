# parkour_training — Go2 parkour teacher/student 학습

업스트림: [CAI23sbP/Isaaclab_Parkour](https://github.com/CAI23sbP/Isaaclab_Parkour)
(Extreme-Parkour의 IsaacLab 포팅)

| 디렉터리 | 관측 입력 |
|---|---|
| `parkour_depth/` | depth 카메라 (본체. 최종 채택) |
| `parkour_lidar/` | 3D LiDAR (A2 Pro에 depth 센서 없을 가능성 대비) |

두 트리의 차이는 **6개 파일뿐**:

```
parkour_tasks/.../go2/agents/parkour_rl_cfg.py
parkour_tasks/.../go2/parkour_mdp_cfg.py
parkour_tasks/.../go2/parkour_student_cfg.py          (LIDAR_CFG)
scripts/rsl_rl/modules/distillation_with_extractor.py
scripts/rsl_rl/modules/feature_extractors/depth_backbone.py   (LidarScanBackbone)
scripts/rsl_rl/modules/feature_extractors/__init__.py
```

## ⚠️ 학습 환경

teacher 학습은 **`parkour_il` conda env에서만** 한다 (rsl-rl-lib 2.3.3).
`env_isaaclab`(3.1.2)은 API drift로 크래시한다.

## teacher 관측 스펙 (753-dim)

```
proprio 53 + scandot 132 + priv_explicit 9 + priv_latent 29(무시) + history 530
```

scandot은 12×11 격자. **실제 센서가 아니라** 제출 시 어댑터로 합성해야 한다.

## 학습 run 계보 (서버 logs/ 에만 존재, 여기엔 코드만)

| run | 의미 |
|---|---|
| `2026-06-24_10-58-09_학습성공_0.45m` | 0.45m 단차 등반 성공 |
| `2026-08-05_01-38-20` | **parkour teacher** (RMA, distillation 없음), `model_4200.pt` |
| `2026-08-28_13-13-12_teacher_overspeed` | overspeed 패널티 teacher |
| `2026-09-01_08-37-52_encoder_depth` | depth 인코더 (최종 채택, latent 0.0257) |
| `2026-09-04_06-50-41_encoder_lidar_front` | front 마운트 (offset 0.338/0/0.081) |

접미사: `teacher_*`/`student_*`(distillation), `encoder_*`(인지 인코더),
`sanity_*`(사전 점검), `_ft`(파인튜닝), `_smoke`(스모크 테스트)
