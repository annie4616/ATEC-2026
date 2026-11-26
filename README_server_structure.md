# ATEC 2026 로봇 챌린지 — 작업 전체 정리

> 작성 2026-11-26. 4개 폴더에 흩어진 ATEC 2026 작업을 하나로 정리한 문서.
> 대회 종료 후 재현·회고용. 각 폴더 안의 개별 README는 그대로 두고, 이 문서가 진입점.

---

## 0. 한눈에 보기

ATEC 2026 Simulation Challenge — IsaacLab 기반 사족보행 로봇 대회.
예선 **93팀 중 10위**로 홍콩 실물 대회 진출.

```
/hdd0/sohyun/ATEC2026/
├── README.md              ← 이 문서
├── 1_main_workspace/   74G   메인 작업공간. Task A/D/E, parkour, 제출본
├── 2_realworld_stack/ 410M   실기예선 통합 스택 (시뮬 + 온보드 미니PC)
├── 3_taska_videos/    680M   Task A 영상 아카이브 (mp4 178개)
├── 4_taskd_sweeps/    580M   Task D cross 파라미터 스윕 결과
├── 5_github_export/    33M   GitHub 공개본 (코드만)
└── _cleanup_records/         디스크 정리 기록
```

> 2026-11-26 정리: 폴더명을 바꾸고 `/hdd0/sohyun/`에 **심볼릭 링크를 남겼다.**
> 기존 스크립트의 절대경로(`/hdd0/sohyun/atec_2026_tesk_e/...`)가 그대로 동작한다.

**대회 태스크** (공식 readme 기준): Off-road Navigation(A), Tabletop Manipulation(E),
Garbage Collection(B), Obstacle Traversal(D).
우리가 실제로 진행한 건 **A, D, E** 세 개.

---

## 1. 실행 환경 (필수)

conda env 2개를 **분리 유지**한다. 섞으면 API drift로 깨진다.

| env | 용도 | 비고 |
|---|---|---|
| `env_isaaclab` | ATEC2026 본체 (Task A/D/E) | IsaacLab 3.1.2 |
| `parkour_il` | parkour teacher/student 학습 | rsl-rl-lib 2.3.3. **여기서만 학습** |

스크립트는 `conda activate` 대신 절대경로로 호출한다:
`/home/sohyun/.conda/envs/env_isaaclab/bin/python`

### 반드시 필요한 환경변수

```bash
export PYTHONPATH=/hdd0/sohyun            # atec_2026_tesk_e.* import 해결
export TMPDIR=/hdd0/sohyun/tmp            # 기본 /tmp/isaaclab는 타인 소유 → PermissionError
export CUDA_VISIBLE_DEVICES=2             # 작업별로 다름, 아래 표 참조
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_x   # 이 머신에선 MPS 우회 필수
```

### GPU 배정 (경험칙)

| 작업 | GPU | 이유 |
|---|---|---|
| IsaacLab 일반 | 7 | |
| Task D ACT 수집/학습/eval | 2 | GPU3 렌더 경합 회피 |
| parkour teacher 재생 | 2 | |

---

## 2. 메인 작업공간: `atec_2026_tesk_e/` (191GB)

> ⚠️ 이 폴더는 git 저장소가 **아니다**. `ATEC2026_Simulation_Challenge/.git`이 있으나
> HEAD·config·objects가 전부 없는 깨진 껍데기라 히스토리 복구 불가.

### 2.1 구성

| 경로 | 크기 | 내용 |
|---|---|---|
| `ATEC2026_Simulation_Challenge/` | 75G | 공식 스타터 repo + 우리 확장. 아래 2.2 |
| `taska_exp/` | 69G | Task A steering 실험 출력 (406개 run) |
| `task_e/` | 27G | Task E 독립 작업 디렉터리 |
| `Isaaclab_Parkour/` | 20G | parkour 학습 (Go2 teacher/student) |
| `taska_pdheads/` | 861M | Task A PD head 버전별 평가 영상 |
| `new_robot_a2pro_piperx/` | 132M | A2 Pro + Piper X 로봇 모델 합성 |
| `task_a_real_exp/` | 107M | A2 실물 대비 실험 + 배포 번들 |
| `IsaacLab/` | 61M | 업스트림 IsaacLab 2.3.2 (에셋 제거된 부분 사본) |
| `Isaaclab_Parkour_lidar/` | 18M | LiDAR 변형 코드 포크. **logs 비어 있음** |
| `yewon/`, `yewon_1/` | 각 7.7M | 동료 제출본. **두 폴더 내용 동일** (중복) |

### 2.2 `ATEC2026_Simulation_Challenge/` 주요 하위

- `source/atec_rl_lab/` — 설치형 태스크/환경 패키지 (`pip install -e`)
- `scripts/` — 진입점 전부. 아래 2.3
- `atec_robot_model/` — USD/URDF 로봇·물체·씬 에셋 + 베이스라인 정책
- `datasets/` — HDF5 데모 14종 (task_a, task_e, task_d_* 10종)
- `runs/`, `logs/` — 학습 결과
- `submission_*/` — 제출 번들 4개. 아래 2.4

### 2.3 진입점 스크립트

**RL 학습** (`scripts/rsl_rl/`)
| 파일 | 용도 |
|---|---|
| `train.py` | PPO 학습. `--task --num_envs --seed --max_iterations --video` |
| `play.py` | 체크포인트 재생 |
| `diag_crosswalk_poses.py` | reset 후 로봇/박스/구덩이 실제 월드 좌표 확인 |

**모방학습 ACT** (`scripts/act/`)
| 파일 | 용도 |
|---|---|
| `collect_demos_task_a.py` | Task A 직진보행 데모 수집 (teacher = B2 보행정책 + 특권 yaw) |
| `collect_demos_task_d.py` | Task D 스크립트 오라클 수집. x>3.5 도달 시 성공 |
| `collect_demos_task_e.py` | Task E pick-and-place 수집. **object 3만 튜닝됨** |
| `train_task_d.py` | 계층적 ACT 학습 (image+qpos → velocity cmd). isaaclab 없이 실행 가능 |
| `train_task_e.py` | Task E ACT 학습 |
| `eval_act_task_d.py` | Task D ACT 평가. 성공 = box x∈[-1.4,0.7] & z<-0.5 |
| `filter_demos.py` | 정지 구간 제거 |
| `filter_demos_quality.py` | Task E 데모 품질 필터 (qvel/jerk/tail_std/length) |
| `classify_failures.py` | 수집 로그에서 실패 유형 분류 (오라클 수정 가능 vs RL 필요) |
| `experiment_climb.py` | 박스 등반 수동 스크립트 실험 |
| `experiment_cross_speed.py` | CROSS 속도 스윕 |
| `experiment_platform.py` | 플랫폼 전략 (구덩이 대신 플랫폼 등반) |

**Task A steering** (`scripts/steering_head_wz/`, `scripts/steering_ytrack/`)
train/eval 쌍으로 구성: `head`, `head_odom`, `gru`, `gru_v2`, `patch`, `stack`.
eval 전용: `ensemble`, `redline_pd`, `iplanner`, `viplanner`.
배포·재생: `play_taska_parkour_a2.py`, `play_taska_lidar_student.py`.

**Task D 진단** (`scripts/task_d/`) — `diag_*` 21개. 좌표계·관절순서·LiDAR·지형 검증용.

**루트 러너 셸** — `run_parkour_teacher_play.sh`, `run_taska_parkour_a2.sh`,
`run_train_act_taskd.sh`, `run_train_act_taskd_rgbd.sh`. 환경변수가 박혀 있어 그대로 쓰면 됨.

### 2.4 제출 번들

| 번들 | 내용 |
|---|---|
| `submission_2/` | `policy.pt` + `policy_act.pt` + `joint_action_seq.npy` |
| `submission_e/` | Task E. `policy_act.pt`, `server.py`, `run.sh` |
| `submission_mem/` | 메모리 재생 방식. `policy_student_walkacross.pt`, `proprio_jprel_start.npy` |
| `submission_taskd_teacher/` | parkour teacher 방식. `climb_replay.npz` + `.keep` 정책 3종 |
| `yewon/task_d_demo/` | 동료 제출본. 2정책 런타임 전환 (b2rough PUSH → parkour teacher CLIMB) |

> `solution.py`는 **파일명 변경 불가** (대회 규정).

---

## 3. 태스크별 결과

### Task A — Off-road Navigation (Hiking)

**문제:** 빨간 40cm 경계선을 따라 코스 주행. 실물엔 경계선이 있고 sim엔 없음.

**접근 변천:**
1. 보행 정책 자체는 수렴 (vx 한계 2.63)
2. 진짜 병목은 **방향 판단**이었음 — 정책이 아니라 steering 문제
3. frozen DINOv2 CLS + MLP → wz 헤드 ("PD8")
4. LiDAR rail 탐지 → rule-based 평행 추종
5. depth 인코더 복원 (latent 0.0257, LiDAR 능가)

**정량 결과** (`taska_exp/PRETRAINED_NAV_SURVEY.md`, 2026-07-31):

| 조건 | 완주율 |
|---|---|
| 직선 baseline (PD8 + kd0.2) | 8/10 (80%) |
| 폭 확대 (5, 6m) | 0/5 |
| 곡선 경로 | 2/5 |
| 90도 꺾임 | 1/5 |

→ **일반화 실패가 명확.** sim 과적합.

**배포본:** `task_a_real_exp/PD8_deploy_for_colleague/` — eval-legal (특권정보 미사용).

### Task D — Obstacle Traversal

**문제:** 박스 밀기 + 구덩이 건너기 + 0.4m 단차 등반.

**최종 해법:** 하드코딩 push + parkour teacher climb, 2정책 런타임 전환.
- PUSH 단계: +14점 성공
- CLIMB 단계: teacher가 OOD로 실패

**cross 최종 락본:** box050/bail190/v230 config 3/3 CROSSED (max_x 3.51).
활성 컨트롤러 4개 byte 동일 확인됨.

**메모리 재생 방식:** 88데모 외워 재생 시도 → 서버 시작 불일치로 박스 미스, +2에서 중단.
**최종 점수 16점.**

### Task E — Tabletop Manipulation

`task_e/` 독립 디렉터리. state machine + ACT.
object별 config 분리 (`config_sugar.py`, `config_my_banana.py`, `config_my_mustard.py`).
데이터 3단계: `trajectory.hdf5` → `_filtered` → `_strict`.

---

## 4. parkour 학습 (`Isaaclab_Parkour/`)

업스트림: `https://github.com/CAI23sbP/Isaaclab_Parkour.git` (Extreme-Parkour의 IsaacLab 포팅).
`parkour_isaaclab/`, `parkour_tasks/` 둘 다 `pip install -e`.

### 학습 run 68개의 3단계

| 시기 | run 수 | 특징 |
|---|---|---|
| 06-24 ~ 06-29 | 40 | 타임스탬프만. 초기 시행착오 |
| 08-05 | 4 | |
| 08-16 ~ 09-04 | 24 | 접미사로 의미 표기. 네이밍 규율 정착 |

### 핵심 run

| run | 의미 |
|---|---|
| `2026-06-24_10-58-09_학습성공_0.45m` | 0.45m 단차 등반 성공 |
| `2026-08-05_01-38-20` | **parkour teacher** (RMA, distillation 없음), `model_4200.pt` |
| `2026-08-28_13-13-12_teacher_overspeed` | overspeed 패널티 teacher |
| `2026-09-01_08-37-52_encoder_depth` | depth 인코더 (최종 채택) |
| `2026-09-03_07-16-35_encoder_lidar_overspeed` | LiDAR 인코더 + overspeed |
| `2026-09-04_06-50-41_encoder_lidar_front` | front 마운트 (offset 0.338/0/0.081) |

접미사 어휘: `teacher_*`/`student_*` (distillation), `encoder_*` (인지 인코더 계열),
`sanity_*` (각 encoder run 직전 점검), `_ft` (파인튜닝 이어하기), `_smoke` (스모크 테스트).

### teacher 관측 스펙 (753-dim)

```
proprio 53 + scandot 132 + priv_explicit 9 + priv_latent 29(무시) + history 530
```
scandot은 12×11 격자. 제출 시 어댑터로 합성 필요 (실제 센서 아님).

---

## 5. 로봇 플랫폼 변경: B2 → A2 Pro + Piper X

`new_robot_a2pro_piperx/` — URDF→USD 합성으로 20-DoF articulation asset 완성.
FixedJoint 및 arm root API 제거가 핵심.

스크립트: `compose_a2_piperx.py`, `verify_articulation.py`, `check_arm_stow.py`,
`sweep_arm_stow.py`, `check_collision.py`, `show_standing.py`.

**주의:** 관절 매핑은 **identity**. 순열 적용 금지 (과거 버그 원인).

---

## 6. 나머지 3개 폴더

이 셋은 같은 것의 변형이 **아니다.** 종류가 다르다.

| 폴더 | 성격 | 코드 유무 |
|---|---|---|
| `ATEC_gmis/` | **소스 저장소** (실기예선 통합 스택) | 있음 |
| `atec_real_world/` | Task A 영상 아카이브 | 거의 없음 (178 mp4) |
| `atec2026_taskd_eunyeong/` | Task D 스윕 결과 아카이브 | 파이썬 1개 |

### 6.1 `ATEC_gmis/` (410MB) — 실기예선 통합 스택

홍콩 실물 대회(A2 Pro + PiPER)용. **세 폴더 중 유일하게 문서화된 코드베이스.**
자체 README가 있고, 의도적으로 **두 스택을 한 repo에** 담았다:

- **시뮬레이터 스택** — IsaacLab + GPU. 알고리즘이 맞다는 걸 증명하는 쪽
- **온보드 스택** (`onboard/`) — 로봇에서 **실제로 도는** 코드.
  Beelink 미니PC(Intel N150, 4 E-core, 12GB, **GPU 없음**) 타깃.
  Isaac / IsaacLab / gymnasium / CUDA **전부 미사용**

| 하위 | 크기 | 내용 |
|---|---|---|
| `onboard/` | 14M | ★ 배포 스택. `oio/`=하드웨어 경계(DDS 다리, CAN 팔, E-Stop + mock) |
| `eunha_rl/` | 96M | EUNHA 정책 계보. crouch/grasp 정책, IK 뱅크, RMA 러너 |
| `sohyun_depth/` | 89M | Task A depth 카메라 parkour 보행 |
| `a2pro_rl/` | 67M | A2+PiperX USD/URDF 에셋 |
| `course_mission/` | 57M | 미션 FSM, 인지, 항법, 팔 제어. `solution_v1~v4` |
| `course_sim/` | 36M | 홍콩 6셀 S코스 환경 구축 |
| `sohyun_lidar/` | 26M | LiDAR 대응본 (A2에 depth 센서 없을 가능성 대비) |
| `from_eunha/` | 12M | 물려받은 보행 정책 체크포인트 |
| `docker/` | 80K | 온보드 런타임 이미지화 |

**핵심 성적** (onboard/README 기재): Task 1 **80/80**, Task 2 **120/160**
(정직한 항법·물체 인식 기준).

**주요 진입점:**
- `onboard/run.py` — 미션 진입점. `--backend mock`(데스크톱) / `--backend real --nic enp2s0`(로봇).
  백엔드 선택만이 유일한 차이. 로봇은 공용 장비라 체크인 전엔 못 만지기 때문
- `onboard/tools/offsim_gate.py` — 수락 게이트 (시뮬레이터·GPU·절대경로 전부 금지 확인)
- `deploy/run_mission.sh` — 정책 교체 가능한 전체 미션.
  ⚠️ 전체가 `{ }`로 감싸여 있다. bash가 실행 중 파일을 재읽기 때문에,
  실행 중 수정하면 오프셋이 밀려 **미션이 두 번 실행된 사고**가 있었음 (`0903_2222_svO`)
- `course_mission/perception.py` — numpy만 사용, torch·Isaac 없음.
  `scripts/tune_detector.py`로 GPU 없이 1초 만에 채점 가능하게 분리

**미니PC 병목:** `GRASP_PERCEPTION_SPEEDUP_PLAN.md`에 분석됨.
파지 정책은 무죄(42→7 MLP, 마이크로초), **인지가 범인**.
`perception.py:1130 _blobs`(순수 파이썬 union-find, GIL 점유)와
`perception.py:208 label_map`(23만 픽셀 × 프로토타입)이 핫스팟.
→ detect_hz 1.2 → 5Hz 개선 (브랜치 `perf/blobs-fast`)

### 6.2 `atec_real_world/` (680MB) — Task A 영상 아카이브

**코드가 거의 없다.** `walk-these-ways/` 밖은 **185개 파일 중 178개가 mp4**, 파이썬 0개.
각 실험 디렉터리는 `parkour_follow.mp4` + `parkour_overview.mp4` 한 쌍.

| 하위 | 크기 | 내용 |
|---|---|---|
| `walk-these-ways/` | 184M | **유일한 코드.** MIT Go1 sim-to-real 스타터킷 벤더링 + A2 config 추가 |
| `course_s/` | 194M | student 정책 코스 주행 ~24변형 |
| `exps/` | 104M | ablation 아카이브 (latentonly, enc_frozen, overspeed, warp…) |
| `depth/` | 81M | depth student ablation (노이즈 레벨, zerolatent, keepd…) |
| `course_t/` | 47M | teacher 정책 코스 주행 |
| `docs/` | 4.2M | 대회 공식 PDF 2개 |
| `course_ts/` | **0** | 완전히 비어 있음 |

**문서가 전혀 없다.** 유일한 `.md`는 업스트림 walk-these-ways README.
**178개 영상에 인덱스가 없고, 디렉터리 이름이 유일한 설명이다.**
→ 세 폴더 중 문서화 공백이 가장 크다.

이 영상들을 만든 코드는 `ATEC_gmis/sohyun_depth`, `ATEC_gmis/sohyun_lidar`,
`atec_2026_tesk_e/Isaaclab_Parkour*`에 있다.

### 6.3 `atec2026_taskd_eunyeong/` (580MB) — Task D 스윕 아카이브

**80개 파일 중 63개가 mp4, 파이썬은 1개.** 셸 스크립트 0개.
스크립트 오라클(학습 정책 아님)로 박스 밀기 + 건너기를 튜닝한 기록.

**스윕 축:** `BOX_PUSH_DONE_X`(박스 목표), `PUSH_X_BAIL_RX`(bail 임계),
cross 속도(v2.30 등), 벽 높이, 박스 회전, y 허용오차(tol018).

**유일한 파이썬 = 최종 산출물:**
`final_taskd_artifacts/push_controller_cross_FINAL_LOCKED_box050_bail190_v230.py`

> "Velocity-command state machine for Task D: push the box into the pit, then cross."

6단계 FSM: `BACK_UP → ALIGN_Y → PUSH_Y → ALIGN_X → PUSH_X → CROSS`.
핵심 통찰 — **+x로 밀면 박스가 플랫폼 벽에 낀다. 먼저 −y로 빼내야 한다.**
yaw를 0으로 고정해 body-frame 명령이 월드에 바로 대응되게 함.

**최종 확정값** (`FINAL_box050_bail190_v230_3of3.txt`):
```
BOX_PUSH_DONE_X = 0.5     PUSH_X_BAIL_RX = -1.90
speed = 2.30              RUN_UP_X = -2.2
RUN_UP_TOL = 0.15         y tolerance = 0.18

3회 반복: max_x=+3.51 CROSSED, steps=1327 (3회 전부 동일)
```
→ 완전 결정적. 재현 검증 완료.

`taskd_oracle_demos_50/` — 오라클 데모 50개 (`trajectory.hdf5`). 모방학습 입력.

⚠️ 위 txt의 **절대경로 3개가 전부 끊겨 있다.**
`/hdd0/sohyun/atec2026_taskd/...`는 존재하지 않고(실제는 `atec2026_taskd_eunyeong`),
나머지 둘도 이제 이 폴더 안으로 들어왔다. 흩어진 스크래치 폴더를 한곳에 모으면서 메모를 안 고친 것.

---

## 6.5 폴더 간 관계

```
atec_2026_tesk_e/          ← 작업 트리 (모든 게 여기서 실행됨)
    │
    ├─→ ATEC_gmis/sohyun_depth/   : Isaaclab_Parkour 79파일 100% 동일 (logs만 제거)
    ├─→ ATEC_gmis/sohyun_lidar/   : 84중 78 동일, 6개가 LiDAR 수정면
    ├─→ atec2026_taskd_eunyeong/  : Task D 스윕 출력 수확
    └─→ atec_real_world/          : Task A 영상 출력 싱크
```

**`Isaaclab_Parkour`가 6곳에 중복 벤더링되어 있다:**
`atec_2026_tesk_e/Isaaclab_Parkour`, `_lidar`, `ATEC_gmis/sohyun_depth/`, `ATEC_gmis/sohyun_lidar/`,
`tmp/atec_gmis_depth/`(89M), `tmp/atec_gmis_lidar_stage/`(26M).
뒤 둘은 납품용 export 스테이징 잔해 — 지워도 된다.

LiDAR 수정면 6파일 (이것만 다름):
```
parkour_tasks/.../go2/agents/parkour_rl_cfg.py
parkour_tasks/.../go2/parkour_mdp_cfg.py
parkour_tasks/.../go2/parkour_student_cfg.py      (LIDAR_CFG)
scripts/rsl_rl/modules/distillation_with_extractor.py
scripts/rsl_rl/modules/feature_extractors/depth_backbone.py   (LidarScanBackbone)
scripts/rsl_rl/modules/feature_extractors/__init__.py
```

---

## 7. 알려진 함정

1. **★ git 저장소 2개가 2026-11-13에 내용물이 지워졌다.**
   `ATEC2026_Simulation_Challenge/.git`과 `ATEC_gmis/.git` 둘 다 HEAD·config·objects가 없는
   빈 디렉터리 껍데기만 남았다 (`du` = 0 bytes). 두 폴더 모두 11-13 04:59~05:06 타임스탬프.
   **히스토리·remote·브랜치 전부 복구 불가.** 코드 자체는 멀쩡하다.
   경로 이름에서 건진 것: `ATEC_gmis`에 `perf` 브랜치와 `origin` remote가 있었다.
   → GitHub에 푸시해 둔 게 있으면 거기서만 복구 가능.
2. **`yewon_1/` = `yewon/task_d_demo/` 완전 동일** — 27분 뒤 복사본. 지워도 무방.
3. **`Isaaclab_Parkour_lidar/logs/` 완전히 비어 있음** — LiDAR 결과는 본체 `Isaaclab_Parkour/`에 있음. README도 본체 사본이라 LiDAR 설명 없음.
4. **`IsaacLab/`은 61MB** — 정상 체크아웃보다 훨씬 작음. 에셋 제거된 부분 사본으로 추정. 의존성으로 쓰기 전 확인 필요.
5. **TMPDIR 미설정 시 PermissionError** — 기본 `/tmp/isaaclab`가 타인 소유.
6. **GPU당 프로세스 3개 이하** — 4개 이상 금지. `OMP_NUM_THREADS`도 같이 제한.
7. **CPU는 절반만** — 384코어 서버에서 `taskset -c 0-191`.
8. **큰 수정은 사본 생성 후** — 원본 in-place 수정 금지. `course_student_a2.py`를 반복 수정해
   8/28 완주 환경을 재현 불가능하게 만든 전례 있음.
9. **같은 exp_name 중복 실행 금지** — ACT 학습 시 결과 덮어씀.
10. **루트에 잔해 다수** — `.log` ~50개, `IsaacGym_Preview_4_Package.tar.gz` 201MB,
    `no_rotate`라는 유닉스 소켓(크래시 잔해).
11. **빈 디렉터리 3개** — `atec_real_world/course_ts/`, `ATEC_gmis/atec_robot_model/`,
    `Isaaclab_Parkour_lidar/logs/`. 전부 파일 0개.
12. **`atec2026_taskd_eunyeong`의 txt 경로가 끊김** — `/hdd0/sohyun/atec2026_taskd/`는 없는 경로.
13. **미니PC 코드는 허락 없이 수정 금지** — `atec_dmis` 위 코드·설정·이미지. 읽기·진단만 자유.
14. **팔과 몸통은 버스가 분리** — 팔=CAN(can0), 몸통=DDS(이더넷). 미니PC에 각각 직결, 상호연결 없음.
    팔 단독 시험은 `tools/arm_bench.py`.

---

## 8. 디스크 정리 기록

*(실행 시 기록)*
