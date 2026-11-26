# ATEC 2026 Task A — Depth-camera Parkour Walking (A2 Pro + Piper X)

Depth-camera 기반 parkour 보행 파이프라인. Teacher(privileged scandots) 정책의 보행을
고정한 채, **depth 카메라 인코더**가 teacher의 scandots latent를 재현하도록 distill하여
험지(잔디→경사→머드→자갈→계단→bins)를 걷는다. 방향은 실배포 가능한 **LiDAR 빨간
경계선(rail) 탐지**로 잡는다.

> LiDAR 전용 버전(`solution_taska_lidar_student.py` 등)은 별도 브랜치로 관리 예정.
> 이 브랜치는 **depth 버전**만 담는다.

## 구성

```
Isaaclab_Parkour/     학습 리포 (extreme-parkour / IsaacLab)
  parkour_isaaclab/     env / mdp / terrains (observations.py: image_features = depth obs)
  parkour_tasks/        default_cfg.py(CAMERA_CFG), go2 configs
  scripts/rsl_rl/       train / play / export + modules
    modules/feature_extractors/depth_backbone.py   DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
    modules/on_policy_runner_with_extractor.py     learn_vision + ENCODER_DEPTH_NOISE aug
demo/                 배포 어댑터 + 모델
  solution_taska_depth_student.py    _depth() 전처리, DEPTH_NOISE, DEPTH_MODEL_TAG
  models/*.pt                        정책/인코더 (Git LFS)
course/               course 주행 driver + 실행 스크립트
  course_depth_a2.py                 MultiMeshRayCasterCamera depth + rail 탐지
  scripts/*.sh                       학습/export/검증 스크립트
```

## 파이프라인

1. **Teacher**: privileged 132-pt scandots → scan_encoder → 32-D latent → actor (사전 학습됨,
   `logs/.../2026-08-28_13-13-12_teacher_overspeed/model_47700.pt`).
2. **Distill** (`train_encoder_depth.sh` / `train_encoder_depth_noise.sh`): teacher actor FROZEN
   (`FREEZE_ACTOR=1`) + `LATENT_ONLY=1`. depth 이미지(60×106 raycast → crop/resize 58×87 →
   `/2.0-0.5`) → depth 인코더가 teacher scandots latent에 MSE 회귀.
   - `ENCODER_DEPTH_NOISE=0.3`: raw depth에 거리비례 곱셈노이즈 → sim2real 강건성.
3. **Export** (`export_depth_*.sh` → `play_teacheractor_export.py`): frozen teacher actor(`policy.pt`)
   + depth 인코더(`depth_latest.pt`) JIT.
4. **배포/검증** (`run_course_depth_*.sh` → `course_depth_a2.py`): course 지형에 depth 카메라를
   raycast, 어댑터가 58×87로 전처리해 인코더 → latent → teacher actor. 방향은 rail 탐지
   (`GUIDE_STEER=1`, `LIDAR_INCLUDE_RAILS=1`), near-rail 목표거리 `GUIDE_KEEP_D`(권장 1.2),
   `GUIDE_KD=0.8`, `GUIDE_CLAMP=0.70`.

## 모델 (demo/models/)

| 파일 | 설명 |
|---|---|
| `policy_taska_depth_student.pt` | teacher actor (frozen 보행) |
| `depth_taska_depth_student.pt` | depth 인코더 (noise-free, latent_loss 0.026) |
| `policy_taska_depth_noise_student.pt` | teacher actor (noise-aug 쌍) |
| `depth_taska_depth_noise_student.pt` | depth 인코더 (30% noise-aug, 배포 권장) |
| `policy_taska_teacher_scandots.pt` | TEACHER_SCANDOTS fallback (privileged scandots 직접 구동) |

배포 시 `DEPTH_MODEL_TAG=noise_`로 noise-aug 인코더 선택 (기본 `""`=noise-free).

## 결과 (course, bins 도달 = x≈14 기준)

- noise-free @ 노이즈 0%: 완주, peak power 79%
- noise-free @ 30% 노이즈: **x=12.5 넘어짐**
- **noise-aug @ 30% 노이즈: 완주 x=14.7, peak power 75%** ← 강건성 개선 확인

## 외부 의존 (별도 준비 필요, 리포에 미포함)

- **Teacher 체크포인트** `2026-08-28_13-13-12_teacher_overspeed/model_47700.pt` — distill/export의 frozen actor.
- **A2 로봇 asset** `atec_rl_lab.assets.robots.a2` + USD (`atec_robot_model/robot/a2/`).
- **course_sim / epkg** — 대회 원본 course 환경 (course driver가 sys.path로 참조).
- conda env: `parkour_il`(학습/export, rsl-rl 2.3.3), `env_isaaclab`(course driver).
