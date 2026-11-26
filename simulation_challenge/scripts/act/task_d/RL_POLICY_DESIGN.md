# Task D 전용 RL 정책 설계

## 0. 왜 보행 정책만으로는 안 되는가

기존 `ATEC-Isaac-Velocity-Flat/Rough-Unitree-B2-v0`(velocity-tracking 보행)는
"명령받은 속도 `[vx, vy, wz]`로 평지/험지를 걷기"만 학습한다. Task D에는:

- 밀어야 할 **박스**(8 kg)가 환경에 없고,
- 넘어야 할 **구덩이/플랫폼 지형**이 학습 지형에 없으며,
- "박스를 구덩이로 밀고 x>3.5 통과"라는 **보상/목표**가 없다.

→ 평지 정책은 구덩이에서 빠지거나 박스를 못 민다(실측: 박스 -3.0→-2.3에서 데드락).
따라서 Task D의 씬·관측·보상·종료를 모두 반영한 **전용 RL task**를 새로 등록해야 한다.

이 설계는 기존 `LocomotionVelocityRoughEnvCfg`(velocity_env_cfg.py)를 베이스로,
Task D 평가환경(`tasks/task_d/env_cfg.py`)의 씬/보상/종료를 RL 학습용으로 이식한다.

---

## 1. 목표(성공) 정의 — 평가환경과 동일하게

`tasks/task_d/`(평가용)에서 그대로 가져온다:

- 로봇 시작 `(-3, 0, 0.8)`, 박스 시작 `(-3, 1.6, 0.5)`, 8 kg, 마찰 0.9.
- 구덩이 x∈[-0.7, 0.7], 깊이 1.0, 폭 1.3~1.4; 플랫폼은 y 한쪽.
- **성공 = 로봇 root x > 3.5** (평가환경 `robot_x_greater_than`, x_threshold=3.5).
- 박스를 구덩이에 채우면(`box_x ∈ [-0.7, 0.7]`) 발판이 되어 통과가 쉬워진다.

학습은 이 성공을 **보상으로 유도**해 정책이 스스로 "박스 밀기 → 통과" 전략을
발견하게 한다(우리가 상태머신으로 짜지 않는다).

---

## 2. 씬(Scene) — velocity 씬에서 교체할 것

베이스 `MySceneCfg` 기준:

| 항목 | 기존(velocity) | Task D용 |
|---|---|---|
| terrain | `ROUGH_TERRAINS_CFG` | `TASK_D_TERRAIN_CFG` (pit/platform, tasks/task_d/terrain.py) |
| box | 없음 | `RigidObjectCfg` 박스 추가 (평가환경과 동일 스펙: 0.8×1.0×0.6, 8 kg, 마찰 0.9, init (-3,1.6,0.5)) |
| robot | MISSING → B2 | `UNITREE_B2_CFG` (보행 학습과 동일; 팔 없는 순수 B2면 충분, Piper팔 불필요) |
| height_scanner | base 위 grid | 유지 (구덩이 인지에 필수) |
| lidar | 없음 | 선택. height_scan만으로도 지형 인지 가능 → 1차엔 생략 |

학습은 단일 지형(Task D 고정맵)이라 `num_envs`를 한 맵에 복제. env_spacing만 충분히.

---

## 3. 관측(Observations) — 보행 관측 + Task D 목표 정보 추가

`PolicyCfg`/`CriticCfg`에 기존 보행 항(base_lin/ang_vel, projected_gravity,
joint_pos/vel, last_action, height_scan)은 **그대로 유지**. 추가:

1. **velocity_commands 제거 또는 고정** — Task D는 외부 속도명령이 목표가 아니라
   "통과"가 목표. velocity command 관측 항을 빼거나, +x 전진을 상수로 고정.
   (간단히: command 항 유지하되 ranges를 lin_vel_x=(0.5,1.0) 양수로 좁혀
    "앞으로 가라"는 약한 바이어스만 줌. 1차 권장.)
2. **box_pos_b** (신규) — 로봇 base 프레임에서 본 박스 상대 위치 (3D).
   박스를 어디서 밀지 정책이 알아야 함. (mdp 관측함수 신규 작성 필요)
3. **target_dx** (신규, 선택) — 성공선(x=3.5)까지 남은 x거리, 박스의 구덩이까지 x거리.
   학습 가속용. height_scan이 구덩이를 보여주므로 1차엔 생략 가능.

> critic(asymmetric)에는 노이즈 없는 정답 박스/로봇 절대좌표를 더 줘도 됨(특권 정보).

---

## 4. 보상(Rewards) — 기존 보행 보상 유지 + Task D 진행 보상 추가

기존 보행 안정화 보상(track_lin_vel, joint penalties, undesired_contacts,
feet_*, upward 등)은 **걷기 품질 유지용으로 그대로 두고 weight 재사용**.
그 위에 Task D 진행 보상을 신규로 얹는다(전부 신규 mdp 함수):

| 보상 | 설명 | 부호/형태 |
|---|---|---|
| `progress_x` | 로봇 root x의 전진량(Δx per step) | (+), dense — 핵심 추진 신호 |
| `box_progress_x` | 박스 x의 +방향 이동량 | (+), dense — 박스 밀기 유도 |
| `box_in_pit` | 박스 x∈[-0.7,0.7] 진입 | (+), one-time bonus (평가 +14 대응) |
| `reach_goal` | 로봇 x>3.5 도달 | (+), 큰 terminal bonus |
| `box_alignment` | 로봇이 박스 -x측 뒤+y정렬에 있을 때 | (+), shaping (밀 자세 유도, 선택) |
| `fall` / out-of-pit | 로봇이 구덩이로 추락 | (−) 또는 종료 |

설계 원칙: **progress_x + box_progress_x(dense) + reach_goal(terminal)** 3개가 최소 핵심.
나머지는 shaping. 보상 부호/스케일은 학습 보며 튜닝.

---

## 5. 종료(Terminations)

| 종료 | 기존 | Task D |
|---|---|---|
| time_out | 유지 | 유지 (episode_length 늘림, 박스밀기 시간 필요) |
| illegal_contact | 유지 | 유지(베이스 충돌) — 단 박스 접촉은 허용해야 하니 sensor body 범위 주의 |
| terrain_out_of_bounds | 유지 | 유지 |
| **reach_goal_x** | 없음 | 신규: root x>3.5 → 성공 종료 (평가환경 `robot_x_greater_than`) |
| **fell_in_pit** | 없음 | 신규(선택): root z가 구덩이 바닥 수준 → 실패 종료 |

---

## 6. 커리큘럼(Curriculum) — 성공률을 위한 핵심

평지 정책이 막힌 근본 이유는 난이도. 커리큘럼으로 점진 학습:

1. **구덩이 폭/깊이**를 작게 시작 → 성공률 오르면 키움
   (`PitAndPlatformTerrainCfg.pit_width_range`, `platform_height_range`를
    terrain curriculum으로 조절; 평가환경 기본 (1.3,1.4)/(1.0,1.2)).
2. **박스 질량**을 가볍게 시작 → 점진적으로 8 kg까지.
3. 기존 `terrain_levels` 커리큘럼 메커니즘 재활용 가능.

---

## 7. 구현 위치 / 파일 (제안)

기존 train 구조를 그대로 따른다:

```
source/atec_rl_lab/atec_rl_lab/train/locomotion/
  taskd/                                  # 신규 패키지 (velocity와 형제)
    taskd_env_cfg.py                      # LocomotionVelocityRoughEnvCfg 상속 + 씬/관측/보상/종료 override
    mdp/
      observations.py                     # box_pos_b 등 신규 관측
      rewards.py                          # progress_x, box_progress_x, box_in_pit, reach_goal ...
      terminations.py                     # reach_goal_x, fell_in_pit
    config/quadruped/unitree_b2/
      __init__.py                         # gym.register("ATEC-TaskD-RL-Unitree-B2-v0", ...)
      taskd_env_cfg.py                    # UnitreeB2 바인딩 (joint_names 등, rough_env_cfg.py 본떠)
      agents/rsl_rl_ppo_cfg.py            # PPO 설정 (b2 것 복사 후 experiment_name=taskd_b2)
```

> 또는 더 가볍게: 기존 `tasks/task_d/env_cfg.py`(평가환경)에 RL 학습용
> observations/rewards 그룹을 추가하고 train.py에서 그 task를 직접 학습.
> 단 평가환경은 카메라/관측이 평가용으로 짜여 있어, **velocity 학습 베이스를
> 확장하는 위(train/) 경로가 RL 학습엔 더 깔끔**하다.

학습 실행 (검증된 형태):
```
CUDA_VISIBLE_DEVICES=7 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
  python -u scripts/rsl_rl/train.py --task ATEC-TaskD-RL-Unitree-B2-v0 \
  --headless --num_envs 4096 --max_iterations 5000 --video
```

산출 정책 → `demo/solution_rl.py` 형태로 감싸 `play_atec_task.py`로 평가.

---

## 8. 1차 구현 범위 제안 (MVP)

전부 한 번에 말고, 학습이 도는 최소 버전부터:

1. 씬: Task D 지형 + 박스 추가, B2 로봇 (관측/보상은 보행 그대로 + progress_x만)
2. 보상: `progress_x`(dense) + `reach_goal`(terminal) + 기존 보행 안정화만
3. 종료: `reach_goal_x` 추가
4. 학습 돌려 **로봇이 박스 없이도 +x로 구덩이를 넘으려 시도하는지** 확인
5. 그 다음 `box_progress_x` / `box_in_pit` / `box_pos_b` 관측 추가해 박스 밀기 유도
6. 커리큘럼(구덩이/박스 난이도)으로 성공률 끌어올림

이 MVP가 돌면, 그때부터 보상 가중치/커리큘럼 튜닝이 본 작업.
```
```
