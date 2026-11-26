# grasp_rl — B2+Piper 파지+리프트 강화학습

ATEC 2026 실환경 예선 과제 2(쓰레기 분리수거)용 파지 정책 학습 환경.
파이프라인 중 **파지+리프트 구간만** RL 담당 (접근=보행 정책, 도달=IK 스크립트, 이송=IK+보행).

- 태스크: `ATEC-Grasp-B2Piper-v0` (stage 1, base 고정) / `ATEC-Grasp-B2Piper-Stage2-v0` (base 자유)
- 로봇: Unitree B2 + Piper 팔 (`UNITREE_B2_PIPER_CFG`), 그리퍼 prismatic 2관절 개방폭 70 mm
- 에피소드: 4 s = 200 스텝 @ 50 Hz (RL 창은 파지+리프트만)

## 방법론 요약

### 1. 사전 준비 — crouch 자세와 pregrasp 뱅크

파지는 **크라우치 자세에서** 수행된다 (바닥 물체에 팔이 닿아야 하므로).

1. `scripts/extract_crouch_pose.py` — 기존 crouch_sweep 정책 체크포인트(읽기 전용)마다
   c=1 명령으로 앉힌 뒤 정착 자세를 평균 → `banks/crouch_poses.json`
   (깊이별 12개 다리 관절각 + base 높이). 학습 기본 깊이는 **0.2 m** (`GRASP_CROUCH_DEPTH`로 선택).
2. `scripts/build_reach_map.py` — 루트를 crouch 자세에 고정 + 무중력에서 팔 6관절을
   소프트 리밋 내 균일 샘플링(FK) → 도달 지도(`reach_map.npz`, base 좌표계 2 cm 복셀)와
   **pregrasp 뱅크**(`pregrasp_bank.npz`, top-down 접근이 되는 (x,y) 격자별 팔 관절해).
3. 매 에피소드 리셋은 **RSI(reference state initialization)**: 뱅크에서 팔 자세를 뽑아 쓰고,
   물체를 손끝 아래(±5 cm 노이즈, yaw ±180°)에 배치 → 탐색 없이도 파지 시도부터 시작.

### 2. 환경 설계 (graspenv/)

- **관측 42차원** = EE pose(base 좌표계) 7 | 표면 파지점-TCP 벡터(EE 좌표계) 3 | 접근축/툴축 6 |
  그리퍼 개방폭 1 | 직전 액션 7 | 파지 파라미터 5 | 물체 OBB 치수 3 | base 상태 10.
  base 상태 슬롯은 stage 1에서 상수 — stage 2에서 실제값으로 채워 **관측 레이아웃을 유지한 채
  stage-1 가중치를 재사용**하기 위한 예약 슬롯.
- **액션 7차원** = Differential IK 상대 포즈 6 (scale 0.02 = 2 cm/step, 고정) + 그리퍼 binary 1.
- **물체는 절차적 프리미티브** (cuboid/cylinder 랜덤 선택, USD 에셋 아님):
  폭·길이·높이를 prestartup 단계에서 축별 독립 스케일링(`replicate_physics=False` 필요),
  리셋마다 질량 0.03–0.4 kg / 마찰 0.2–0.9 / 자세 랜덤화.
- **접촉 물리를 신뢰할 수 있게 고정**: contact_offset 0.002 / rest_offset 0.0,
  solver position iteration 16 (로봇·물체 모두). 손가락은 전용 액추에이터
  (stiffness 1500, effort 30 N)로 분리 — 공용 팔 게인(stiffness 80)으로는 파지력이 ~2.8 N뿐이라
  물체가 미끄러져 빠지는 문제를 수정한 것.

### 3. 2단계 커리큘럼 (base 고정 → frozen crouch 다리)

- **Stage 1** — `fix_root_link=True`로 base를 crouch 자세에 고정하고 팔+그리퍼만 학습.
- **Stage 2** — base를 풀고, **동결(frozen)된 crouch 정책**이 다리 12관절을 구동
  (`mdp/leg_actions.py`의 zero-dim ActionTerm: crouch_sweep depth-0.2 체크포인트를 그대로 평가,
  45-obs/12-action 인터페이스 재현). RL 정책의 액션 공간은 7차원 그대로, base 상태 관측만
  실제값으로 교체. 추가 종료 조건 `base_fell`(기울어짐/주저앉음) 및 RSI 물체 배치를
  앞발 밟힘 방지 위해 x ≥ 0.50 m로 제한.

### 4. 물체 크기 커리큘럼 (runs/)

`train_grasp.py --width_range/--length_range/--height_range`가 prestartup 스케일 범위로 매핑된다.

| run | 날짜 | 환경 | 폭 | 길이 | 높이 |
| --- | --- | --- | --- | --- | --- |
| `grasp1a` (+`_ext`/`_grip`/`_sw` 재학습) | 07-23~24 | stage 1 | 40–50 mm | =폭 | 60–130 mm |
| `grasp1b` | 07-24 | stage 1 | 30–60 mm | 30–60 mm | 60–130 mm |
| `grasp2` | 07-25 | **stage 2** | 30–60 mm | 30–60 mm | 60–130 mm |
| `grasp1c` | 07-25 | stage 2 | 30–60 mm | **30–250 mm** (길쭉한 물체) | 60–130 mm |
| `grasp1d` | 07-25 | stage 2 | 30–**66** mm | 30–250 mm | **20**–130 mm |

- 1a 계열은 그리퍼 액추에이터 보강 등 환경 수정을 반영한 반복 재학습.
- 폭 상한은 0.060 → **0.066 m로 완화** (2026-07-25 승인): 대회 소품 330 ml 캔(직경 66 mm) 대응.
  높이 하한 20 mm는 수박껍질류 납작 물체 대응.
- 그래스 정책 평가는 `scripts/play_grasp.py` (성공 카운트 + 근접 카메라 녹화),
  `scripts/eval_matrix.sh` (프리미티브 회귀 + 캔/껍질 스탠드인 + task_b USD 물체 매트릭스).
  `videos/`에 66 mm 캔·납작 껍질·stage-2 free-base 플레이 결과 영상.

## 목표 함수 (rewards / terminations)

핵심 판정자 `stable_grasp` = **양손가락 모두 물체와 접촉(힘 > 1 N) AND |v_물체 − v_EE| < 0.1 m/s**
(접촉만 보면 "끌고 다니기"에 보상이 새므로 속도 일치를 함께 요구).

| 항 | 정의 | weight |
| --- | --- | --- |
| `reach` | `1 − tanh(‖TCP − 파지점‖ / 0.1)` — 파지점은 물체 **표면**(윗면 15 mm 아래, 바닥 10 mm 클램프) | 2.0 |
| `align` | (툴 z축이 연직 하방) × (0.5 + 0.5·손가락축·물체 단축 정렬) | 0.5 |
| `grasp` | `stable_grasp` 성립 여부 | 2.0 |
| `lift` | `clamp(바닥 이격 / 0.10, 0, 1)` — **stable_grasp로 게이트** (튕겨 올리기 보상 차단) | 10.0 |
| `success` | 성공 종료 시 터미널 보너스 | 2500.0 |
| `action_rate` | 액션 변화율 L2 | −0.01 |
| `arm_vel` | 팔 관절속도 L2 | −5e-4 |

- 보상은 dt(0.02 s)가 곱해지므로 터미널 보너스는 2500×0.02=50 — 타임아웃까지 들고 버티며
  모을 수 있는 lift 보상(~35)을 이겨야 **조기 성공 종료가 우세**해지도록 설정한 값.
- **성공 판정(종료)**: `stable_grasp` AND 바닥 이격 ≥ 0.05 m가 **25 제어스텝 연속** 유지
  (높이 스냅숏이 아닌 지속 유지 기준).
- 그 외 종료: time_out 4 s / `object_out_of_zone` (물체가 base에서 1 m 이탈·비정상 높이) /
  stage 2 `base_fell` (기울어짐 > 임계 또는 base 높이 < 0.12 m).

**PPO** (rsl_rl `OnPolicyRunner`, `graspenv/agents.py`): actor-critic [256,128,64] ELU,
gamma **0.995**(150–250스텝 에피소드에 0.99는 근시안), lam 0.95, adaptive lr (desired KL 0.01),
entropy 0.006, 24 steps/env, 2048 envs (stage 2는 1024). crouch 체크포인트를 초기 가중치로
쓰지 않고 **from scratch** (액션 공간이 다름).

## 참고한 코드 (출처)

| 대상 | 출처 |
| --- | --- |
| 환경 골격 (manager-based cfg, DiffIK 액션, reach/lift 리워드 커널) | vendored isaaclab 2.3.2의 `Isaac-Lift-Cube-Franka-v0` lift 태스크 패턴 (Phase 0에서 스모크 검증, 39.4k steps/s) |
| `crouchpkg/` | 외부 `logs/rsl_rl/260630/crouchpkg` **사본** (규칙 3). flat velocity 보행 env(`atec_rl_lab.train.locomotion.velocity` `UnitreeB2FlatEnvCfg`)를 서브클래스한 명령형 stand↔crouch 태스크 — frozen 다리 정책과 crouch 자세의 원본 |
| `scripts/extract_crouch_pose.py` | 리포 `scripts/rsl_rl/play.py`의 체크포인트 로딩 패턴 |
| `scripts/train_grasp.py` | IsaacLab 표준 rsl_rl 학습 스크립트 패턴 (AppLauncher + `RslRlVecEnvWrapper` + `OnPolicyRunner`) |
| 로봇/에셋 | 리포 `source/atec_rl_lab`의 `UNITREE_B2_PIPER_CFG` (b2_piper.usda) — atec_rl_lab은 반드시 이 리포 사본으로 resolve (train 스크립트에서 assert) |

## 구조

```
grasp_rl/
  graspenv/             # ATEC-Grasp-B2Piper-v0 / -Stage2-v0 등록 + env cfg
    grasp_env_cfg.py    # 씬/액션/관측/이벤트/리워드/종료 (stage 1·2)
    agents.py           # PPO 러너 cfg
    mdp/                # rewards, terminations, events(RSI), observations, leg_actions(frozen crouch)
  crouchpkg/            # crouch 환경 정의 사본 (원본: 외부 logs/rsl_rl/260630/crouchpkg — 규칙 3에 따라 복사)
  scripts/
    extract_crouch_pose.py   # crouch_sweep 체크포인트(읽기 전용)별 안정 crouch 자세 추출
    build_reach_map.py       # 고정 루트+무중력 FK 샘플링 → 도달 지도 + 사전파지 관절 뱅크
    train_grasp.py           # 학습 (stage/크기 범위는 CLI 인자)
    play_grasp.py            # 평가/녹화 (프리미티브 또는 --object_usd 로 task_b USD 물체)
    eval_matrix.sh           # 체크포인트 1개의 전체 파지력 매트릭스
    smoke_grasp_env.py       # env 스모크 테스트
    probe_stage2.py          # stage 2 안정성 프로브
  banks/
    crouch_poses.json        # 깊이별 12개 다리 관절각 + base 높이
    reach_map.npz            # base 좌표계 2cm 복셀 도달 영역 (+ _0.22 변형)
    pregrasp_bank.npz        # top-down 접근 가능한 (x,y)별 팔 관절해 (RSI 리셋용, + _0.22 변형)
  runs/                 # 학습 로그/체크포인트 (표 참조)
  videos/               # 평가 영상 (66mm 캔, 납작 껍질, stage-2 free base 등)
```

## 실행 규칙

- **GPU 5 전용**: 모든 명령 앞 `CUDA_VISIBLE_DEVICES=5`
- 실행 위치: `real_world_preliminary/ATEC2026_Simulation_Challenge` (리포 루트)
- python: `~/.conda/envs/isaaclab/bin/python`
- `logs/rsl_rl/crouch_sweep/runs/`는 읽기 전용
- 이 빌드에서 `simulation_app.close()`는 fast shutdown → **print는 반드시 flush=True**

## 실행 예

```bash
cd real_world_preliminary/ATEC2026_Simulation_Challenge

# stage 1a 학습 (폭 40–50 mm 기본)
CUDA_VISIBLE_DEVICES=5 ~/.conda/envs/isaaclab/bin/python \
    grasp_rl/scripts/train_grasp.py --headless --device cuda:0 \
    --num_envs 2048 --max_iterations 5000 --run_name grasp1a

# stage 2 (free base + frozen crouch legs), 1b 체크포인트에서 이어서
CUDA_VISIBLE_DEVICES=5 ~/.conda/envs/isaaclab/bin/python \
    grasp_rl/scripts/train_grasp.py --headless --device cuda:0 \
    --task ATEC-Grasp-B2Piper-Stage2-v0 --num_envs 1024 \
    --width_range 0.030,0.060 --resume grasp_rl/runs/<1b_run>/model_XXXX.pt --run_name grasp2

# 평가/녹화
CUDA_VISIBLE_DEVICES=5 ~/.conda/envs/isaaclab/bin/python \
    grasp_rl/scripts/play_grasp.py --headless --device cuda:0 \
    --checkpoint grasp_rl/runs/<run>/model_XXXX.pt --video

# 전체 파지력 매트릭스 (프리미티브 + 캔/껍질 + task_b USD)
bash grasp_rl/scripts/eval_matrix.sh grasp_rl/runs/<run>/model_XXXX.pt /tmp/eval_out
```

## Phase 로그

- **Phase 0** (완료): 조사 보고 — vendored isaaclab 2.3.2 API 기준, `Isaac-Lift-Cube-Franka-v0` 스모크 39.4k steps/s,
  b2_piper.usda 그리퍼 prismatic 2관절(개방폭 70mm), `UNITREE_B2_PIPER_CFG` 존재 확인
- **Phase 1** (완료): 골격 + crouchpkg 사본 + crouch 자세 추출(7개 깊이) + 도달 지도/사전파지 뱅크(깊이 0.2)
  - 관찰: crouch0.26 정책은 무이벤트 결정론 조건에서 z=0.091까지 내려감(직립 유지). 파지용 기본 깊이는 0.2 사용.
- **Phase 2** (완료): `graspenv/` 패키지 — `ATEC-Grasp-B2Piper-v0` 등록, 42차원 관측(베이스 상태 10 슬롯 포함),
  IK-rel(scale 0.02)+그리퍼 binary 액션, 절차적 물체(폭 30~60/높이 60~130mm 독립, prestartup scale,
  replicate_physics=False), 질량/마찰/자세 reset 랜덤화, stable_grasp(양손가락 접촉+속도일치),
  25스텝 유지 성공 판정, pregrasp_bank RSI 리셋. smoke: obs 42 ✓ 폭 상한 54.5mm ✓ RSI 평균 8cm ✓ NaN 없음
- **Phase 3** (완료): stage 1 학습 — 1a(40–50mm) 및 반복 수정 재학습(`_ext`/`_grip`/`_sw`), 1b(30–60mm)
  - 수정: 손가락 전용 액추에이터(stiffness 1500) — 공용 게인으로는 파지력 부족으로 미끄러짐
- **Phase 4** (완료): stage 2 — frozen crouch 다리 + free base(`grasp2`), 이후 물체 범위 확장
  (`grasp1c` 길이 250mm, `grasp1d` 폭 66mm 캔/높이 20mm 껍질; 폭 상한 0.066 완화 2026-07-25)
