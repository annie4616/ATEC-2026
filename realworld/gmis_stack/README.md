# ATEC2026 실기예선 — A2 Pro + PiPER 통합 스택

두 개의 스택이 한 레포에 있습니다.

- **시뮬레이터 스택** — 대회장(30분 사진 텍스처 코스)을 구성하고, walking / crouch / grasp
  정책을 셸 환경변수로 갈아끼우며 미션을 채점·녹화합니다. **알고리즘이 맞다는 것을 증명하는**
  쪽이고, IsaacLab/IsaacSim + GPU가 필요합니다.
- **온보드 배포 스택 [`onboard/`](onboard/)** — **로봇에서 실제로 도는** 쪽입니다.
  Beelink(Intel N150, E-코어 4개, 12 GB, GPU 없음)를 대상으로 하고
  Isaac / IsaacLab / gymnasium / CUDA를 하나도 쓰지 않습니다.

이 구분이 중요한 이유: 시뮬 스택은 로봇에서 **한 줄도 돌지 않습니다.** 모든 액추에이터 쓰기가
`set_joint_position_target`으로 PhysX에 들어가고, 상태는 전부 `env.step()`이 주며, 50 Hz를
지킬 벽시계 스케줄러가 어디에도 없습니다.

모든 경로는 **리포 루트 기준 상대경로**로 해석됩니다.

## 폴더 구조

```
ATEC_gmis/
├── onboard/                 ★ 실기 배포 스택 (미니PC에서 실행되는 코드)
│   ├── oio/                 하드웨어 경계 — DDS 다리, CAN 팔, 센서, E-Stop + 전부의 목(mock)
│   ├── rt/ control/ estimate/ perceive/ mission/ bringup/
│   ├── tools/               오프심 게이트·현장 도구·검증 (offsim_gate, dump_motor_map …)
│   ├── configs/ artifacts/  하드웨어 계약(yaml) + 정책 5종 + sha256 매니페스트
│   └── README.md            상세 문서 (한국어: README.ko.md)
├── deploy/                  시뮬 실행 진입점
│   ├── run_mission.sh       풀 미션(수집+배달, 채점) — 정책 스왑 지원
│   ├── run_walk.sh          보행+코스 주행 영상
│   ├── walk_course.py       방향 지시 코드(사본) — LiDAR 레일 항법 프로토타입 포함
│   └── logs/                실행 출력(config.json, report_venue.json, mp4, src/ 스냅샷)
├── from_eunha/              보행 정책 체크포인트
│   ├── 0824/                model_22998.pt + adaptation.pt  ← 기본(최신)
│   ├── speed_0.75/          구 챔피언 보행 쌍(스왑 예시용)
│   └── 0821/adaptation.pt
├── eunha_rl/
│   ├── s075/crouch_runs/…/model_2999.pt    crouch 정책(깊이 0.24)
│   ├── s075/runs/…/model_900.pt            grasp 정책(구 챔피언)
│   ├── s0821/banks_ik/                     pre-grasp 뱅크(IK 격자)
│   ├── s0821/adapt_act.txt                 적응망 활성함수(relu)
│   ├── detached/champ_*.txt                grasp 챔피언 포인터(리포-상대 경로)
│   └── epkg/                               RMA 보행 러너(rma.py)
├── course_mission/          미션 FSM·인식·항법·팔 제어 (runner/launch/perception/navest…)
│                            + Isaac 없이 도는 층: mathx·navfuse·armkin·catalogue·deploy
├── course_sim/              대회장 환경 구성(홍콩 6셀 S코스 기본·COURSE_VENUE=legacy로 구7지형)
├── grasp_rl/                파지+리프트 RL 학습 환경(graspenv/crouchpkg/a2pkg)
├── exp_runs/                파지 실험 래퍼·감사 스크립트 (runs/·logs/는 gitignore)
├── a2pro_rl/                로봇 자산(a2_piperx USD/URDF, 소품 USD, a2pkg cfg)
└── source/atec_rl_lab/      IsaacLab 태스크 등록(대회 기본 패키지)
```

## 요구 환경

| | 시뮬 스택 | 온보드 스택 |
|---|---|---|
| 파이썬 | IsaacLab/IsaacSim conda(예: `isaaclab`) | 평범한 python 3.10+ |
| GPU | 1장 (기본 `GPU=0`) | **없음** |
| 의존성 | IsaacLab 전체 | `onboard/requirements-onboard.txt` (numpy·torch-cpu·pyyaml, 로봇에선 +SDK) |

## 실행 — 시뮬

```bash
# 1) 보행 + 코스 주행 영상 (~5분)
PY=~/.conda/envs/isaaclab/bin/python bash deploy/run_walk.sh

# 2) 풀 미션 (사진 월드, 30분, 채점 + overview/follow 녹화)
PY=~/.conda/envs/isaaclab/bin/python bash deploy/run_mission.sh

# 3) 실기 조건(항법·물체 모두 정직)으로 재현
NAV_HONEST=1 OBJ_HONEST=1 PY=~/.conda/envs/isaaclab/bin/python bash deploy/run_mission.sh
```

출력은 `deploy/logs/<날짜_태그>/`: `mission.log`(채점 라인), `report_venue.json`(시도별 장부),
`config.json`(읽힌 환경변수 전부 + 체크포인트 5개 경로·sha256), `src/`(실행 시점 소스 스냅샷),
`mission_venue_{overview,follow}.mp4` (`NO_VIDEO=1`이면 녹화 생략).

## 실행 — 온보드 (로봇 없이도 전부 돌아갑니다)

```bash
# 수용 게이트 — 시뮬레이터·GPU·절대경로 없이 도는지 (73개 검사)
python onboard/tools/offsim_gate.py

# 목(mock) 로봇으로 전 스택 구동
python -m onboard.run --backend mock --gate timer --minutes 1

# 안전 계약 — 주입된 고장 4종
for f in lowstate_stall inverted estop joint_fault; do
  python onboard/tools/replay_run.py --inject $f --quiet; done

# 실물
NIC=enp2s0 bash onboard/bin/run.sh
```

하드웨어 접근은 `onboard/oio/` 한 층에만 있고, 그 위 전부(정책·추정·안전·미션)는 목으로
완주시킵니다. A2 Pro는 체크인 데이까지 만질 수 없는 공용 장비라서, **로봇이 있어야만 돌려볼 수
있는 코드는 디버깅할 시간이 없는 그 하루에 처음 실행되기** 때문입니다.
자세한 내용은 [`onboard/README.ko.md`](onboard/README.ko.md).

## 정책 스왑 (시뮬 스택 모듈화의 핵심)

정책 3종은 전부 **셸 변수로 교체** — 코드 수정 불필요. 경로는 리포 루트 기준 상대경로.

```bash
# walking: 구 챔피언 쌍으로 교체
WALK_CKPT=from_eunha/speed_0.75/model_14999.pt \
ADAPT_CKPT=from_eunha/0821/adaptation.pt ADAPT_ACT=relu \
  bash deploy/run_mission.sh

# crouch: 다른 crouch 체크포인트로 교체
CROUCH_CKPT=eunha_rl/s075/crouch_runs/<내_런>/model_2999.pt bash deploy/run_mission.sh

# grasp: 다른 grasp 체크포인트로 교체
GRASP_CKPT=exp_runs/runs/<내_런>/model_1999.pt bash deploy/run_mission.sh
```

새 체크포인트는 형식만 맞으면 폴더 아무 데나 두고 상대경로로 지정하면 됩니다
(walking: RSL-RL actor 53→512→256→128→12 + adaptation(hist 50/feat 45);
crouch/grasp: 본 계보의 RSL-RL 체크포인트).

온보드 스택은 같은 정책을 `onboard/artifacts/`에 **sha256 매니페스트와 함께 복사해서** 씁니다
(`python onboard/tools/bundle_artifacts.py`). 챔피언 파지 정책이 `exp_runs/runs/`(gitignore)에
있어 레포에 없었기 때문이고, 코드 봉인에 들어가야 하는 것은 그 번들입니다.

## 대회장 환경 구성 노브

| 변수 | 기본 | 의미 |
|---|---|---|
| `VENUE_PHOTO` | 1 | 사진 텍스처 월드(대회 재현) |
| `BASKET` | 1 | 등 바구니 적재(트립당 3개) |
| `NAV_HONEST` | 0 | 1 = 지도 사전정보 없는 실전형 항법 |
| `OBJ_HONEST` | — | 1 = 물체 진리값 차단(검출로만 파지·판정) |
| `SEED` | — | `MISSION_LAYOUT_SEED` 물체 배치 시드 |
| `NO_VIDEO` | 0 | 1 = 녹화 생략(장시간 안정성) |
| `GPU` | 0 | CUDA 장치 |

그 외 조향·집기 튜닝 노브(`VX, LAT_GAIN, AIM_X, CORRIDOR, MAX_CARRY,` …)는
`deploy/run_mission.sh` 상단 주석 참조.

## 검증 상태

**점수는 조건을 밝히지 않으면 의미가 없습니다.** `deploy/logs/*/config.json`의
`MISSION_NAV_HONEST` / `MISSION_OBJ_HONEST`로 전 런을 분류한 실제 집계:

| 조건 | 런 | 과제1 | 과제2 최고 | 최고 런 |
|---|---|---|---|---|
| 항법·물체 모두 **오라클** | 68 | 80/80 (65) | 120/160 | `0903_1905_svG` (can 2, rind 2, lunchbox 2) |
| 항법 **정직** + 물체 오라클 | 36 | 80/80 (32) | 120/160 | `0905_2104_hs16a` |
| 항법·물체 모두 **정직** | 40 | 80/80 (28) | **20/160** | `0907_1426_fv16` (can 1) |

마지막 줄이 실기 조건입니다. 그리고 40런 중 **39런이 과제2 0점**입니다 — 즉
**과제1 80점은 견고하고, 과제2는 정직 조건에서 아직 재현되지 않습니다.** 위 두 줄의
120/160은 시뮬 진리값을 읽는 경로가 살아 있는 상태의 숫자이므로 실기 기대값이 아닙니다.

그 외:

- `run_walk.sh`: 0824 쌍이 사진 월드 전 구간(−19→+14.5 m)을 88초에 직립 완주
- 적응망 활성함수는 **relu 고정** — elu는 회전·재가속에서 트래킹 붕괴(실측 0/160)
- 온보드 스택: `offsim_gate` **73/73**, 안전 주입 4종 전부 기대 상태,
  폐루프 항법 0.8 m 이탈에서 64 mm 수렴(8시드 경계이탈 0)
- 목 미션 완주 **컨테이너에서 50.00 Hz, dt p99 20.0 ms, 최악 틱 20.2 ms, late 0 / overrun 0**
  (호스트는 49.98 Hz·최악 34 ms·overrun 1 — 호스트에서는 SCHED_FIFO 요청이 거부되기 때문.
  컨테이너는 `--cap-add SYS_NICE` + `--ulimit rtprio`로 실제로 실시간 우선순위를 받습니다)
- `check_docs`는 **100개 검사 중 91개 통과**(도커 배포 계약 15개 게이트 포함, 전부 통과).
  실패 9건은 전부 위 점수 표가 이 레포에 없는 런 로그(`deploy/logs/`의 대부분은 gitignore)를
  인용하기 때문이며, 코드가 아니라 출처 기록의 문제입니다 — 아래 **주의** 참조

## 주의

- 원본 개발 트리(`../real_world_preliminary/ATEC2026_Simulation_Challenge`)와 독립된
  **사본**입니다. 원본 코드는 수정하지 않았고, 이 사본에만 모듈화 훅(launch.py의
  `MISSION_CROUCH_CKPT`/`MISSION_GRASP_CKPT` env, champ 파일 상대경로 해석)이 있습니다.
- `eunha_rl/detached/champ_*.txt`는 grasp 챔피언의 **리포-상대 경로**를 담습니다.
- 실기 배포 갭 감사(54건, BLOCKER 12 / SILENT-WRONG 33)는 원본 트리의
  `course_mission/DEPLOY.md`에 있습니다. `onboard/`가 그 P0 항목들의 구현입니다.
