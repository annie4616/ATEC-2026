# ATEC 2026 Task A — LiDAR Parkour Walking (A2 Pro + Piper X)

LiDAR 기반 parkour 보행 파이프라인. **depth 카메라 대신 온보드 3D LiDAR**로 험지
(잔디→경사→머드→자갈→계단→bins)를 걷는다. A2 Pro에 depth 센서가 없을 가능성에 대비한
depth 버전(`sohyun_depth/`)의 LiDAR 대응본이다.

Teacher(privileged scandots) 정책의 보행을 고정한 채, **LiDAR 인코더**(`LidarScanBackbone`,
5760-ray 1D-conv)가 teacher의 scandots latent를 재현하도록 distill한다. 방향은 실배포 가능한
**LiDAR 빨간 경계선(rail) 탐지**로 잡는다(GT 월드 위치/yaw 미사용).

## 구성

```
Isaaclab_Parkour/     학습 리포 (extreme-parkour / IsaacLab, LiDAR 버전)
  parkour_isaaclab/     env / mdp / terrains (observations.py: lidar_features)
  parkour_tasks/        default_cfg.py, parkour_student_cfg.py(LIDAR_CFG), go2 configs
  scripts/rsl_rl/       train / play / export + modules
    modules/feature_extractors/depth_backbone.py   LidarScanBackbone(1D conv 5760ray) + 2D path(LIDAR_2D)
    modules/on_policy_runner_with_extractor.py      learn_vision (LATENT_ONLY distill)
demo/                 배포 어댑터 + 모델
  solution_taska_lidar_student.py    _lidar() 전처리, LIDAR_MODEL_TAG, rail 탐지, WARP_ENCODER
  models/*.pt                        정책/인코더 (overspeed teacher, 직접 커밋)
course/               course 주행 driver + 실행 스크립트
  course_student_a2.py               RayCaster LiDAR + rail 탐지(_detect_rail) + 진단(LATENT_DIAG)
  scripts/*.sh                       학습/export/검증 스크립트
```

## 파이프라인

1. **Teacher**: privileged 132-pt scandots → scan_encoder → 32-D latent → actor
   (`2026-08-28_13-13-12_teacher_overspeed/model_47700`, 속도추종 페널티 학습본).
2. **Distill** (`train_encoder_lidar_overspeed.sh`): teacher actor FROZEN (`FREEZE_ACTOR=1`) +
   `LATENT_ONLY=1`. 5760-ray LiDAR 스캔(`scan/max_dist - 0.5`) → `LidarScanBackbone`(1D conv)가
   teacher scandots latent에 MSE 회귀. iter 50000, latent_loss 최저 **0.032**.
3. **Export** (`export_lidar_overspeed_50000.sh` → `play_teacheractor_export.py`): frozen teacher
   actor(`policy.pt`) + LiDAR 인코더(`depth_latest.pt`) JIT.
4. **배포/검증** (`run_course_lidar_overspeed_strongrail.sh` → `course_student_a2.py`): course 지형에
   RayCaster LiDAR, 어댑터가 인코더 → latent → teacher actor. 방향은 rail 탐지(`GUIDE_STEER=1`).

## 모델 (demo/models/)

| 파일 | 설명 |
|---|---|
| `policy_taska_lidar_overspeed_student.pt` | teacher actor (frozen 보행) |
| `depth_taska_lidar_overspeed_student.pt` | LiDAR 인코더 (`LidarScanBackbone`, latent_loss 0.032) |

어댑터에서 `LIDAR_MODEL_TAG=overspeed_`로 선택.

## 배포 설정 (검증본과 동일하게 재현하려면 반드시)

`run_course_lidar_overspeed_strongrail.sh` 참조. 핵심 env:

- **`WARP_ENCODER=0`** — 필수. LiDAR 인코더는 1D `LidarScanBackbone`(77-D proprio → combination_mlp
  32+77=109). 기본값 1이면 88-D pose-append로 shape mismatch. **제출 시 반드시 명시.**
- **`LIDAR_MODEL_TAG=overspeed_`** — overspeed 모델 선택.
- **`USE_GT_ODOM=0`** (기본) — GT 월드 pose 미사용.
- rail 조향: `GUIDE_STEER=1 GUIDE_KD=0.8 GUIDE_CLAMP=0.70 GUIDE_LATERR_CLAMP=0.8 GUIDE_KEEP_D=1.2`.
  - 함정: `GUIDE_LATERR_CLAMP=0.6`이면 게인 위상어긋남으로 **180° 스핀**. 0.8이 정답.
- `ENV_MODE=auto` — 지형 flag를 LiDAR 높이-std로 추론(GT 라벨 미사용).

## 결과 (course, bins 도달 = x≈14 기준)

- **bins 완주 (final x=15.0)**, 스핀·stall 없음, rail 평행 추종 안정.
- **Power**: bins 이전 run_max **93%**(계단 등반 순간), 그 외 구간 ≤44%. 정격 100% 미만.
- **Privileged 정보 미사용**: 방향=LiDAR rail(센서 프레임, `quat_rotate_inverse`), roll/pitch=IMU
  gravity 유래, 지형=LiDAR 높이-std, scandot latent=온보드 LiDAR 인코더. GT 월드 pose/yaw/scandot/
  지형라벨 모두 배포 결정에 미사용(driver가 `_live_yaw`/`_live_pos` 주입하나 `GUIDE_STEER` 경로가
  우회).

## 시뮬레이터 함정 (수정 반영됨)

- **LiDAR miss-fill**: `course_student_a2.py`의 `_lidar_scan`이 빔 miss(하늘/틈)를 **0.0**으로 채워야
  학습(observations.py `nan_to_num(nan=0.0)`)과 일치. 과거 10.0으로 채워 정규화 후 +0.5(학습 -0.5)로
  정반대 → 인코더 OOD 입력 → latent 오염. **본 리포는 0.0로 수정됨.** 실물 LiDAR 드라이버의 miss
  처리도 학습과 맞출 것.

## 진단 도구

- `LATENT_DIAG=1` + `INJECT_SCANDOTS=1` + `FLAT_LATENT_CLAMP=0`: 매 스텝 student LiDAR latent vs
  teacher scan_encoder(진짜 132 scandot) latent의 mse/cos 로깅. 인코더 재현 품질 검증용
  (miss-fill 정상 시 cos≈0.93, mse≈0.02).
- `ZERO_LATENT=1`: scandots latent를 0으로 → 지형정보 없는 순수 보행 (방향 편향 진단).

## 외부 의존 (별도 준비 필요, 리포에 미포함)

- **Teacher 체크포인트** `2026-08-28_13-13-12_teacher_overspeed/model_47700.pt`.
- **A2 로봇 asset** `atec_rl_lab.assets.robots.a2` + USD.
- **course_sim / epkg** — 대회 원본 course 환경.
- conda env: `parkour_il`(학습/export, rsl-rl 2.3.3), `env_isaaclab`(course driver).
