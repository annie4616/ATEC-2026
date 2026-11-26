# `onboard/` — A2 Pro + PiPER-X, 미니PC 배포 스택

*(English: [README.md](README.md))*

로봇에 실제로 올라가는 코드입니다. 대상은 Beelink(Intel N150, E-코어 4개, 12 GB, GPU 없음)이고,
`rsync onboard/ robot:` 한 번에 올라가도록 자립적으로 구성했습니다. **Isaac Sim / IsaacLab /
gymnasium / CUDA 를 하나도 쓰지 않습니다.**

이 저장소의 나머지는 *알고리즘이 맞다*는 증명입니다(정직 조건에서 과제1 80/80, 과제2 120/160).
다만 그 코드는 로봇에서 한 줄도 돌지 않습니다 — 모든 액추에이터 쓰기가
`robot.set_joint_position_target`을 통해 PhysX로 들어가고, 상태는 전부 `env.step()`이 주며,
벽시계 스케줄러가 어디에도 없습니다. 이 디렉터리가 **로봇에서 실제로 도는 쪽**입니다.

## 빠른 시작

```bash
# 1. 수용 게이트 — 시뮬레이터·GPU·절대경로 없이 도는지
python onboard/tools/offsim_gate.py

# 2. 목(mock) 로봇으로 전 스택 구동 (이 머신에서)
python -m onboard.run --backend mock --gate timer --minutes 1

# 3. 안전 계약
for f in lowstate_stall inverted estop joint_fault; do
  python onboard/tools/replay_run.py --inject $f --quiet; done

# 4. 폐루프 코스 추종
python onboard/tools/check_nav.py

# 5. 과제2: 트래커 오프라인 채점 후, 녹화 프레임으로 collector 구동
python onboard/tools/check_collector.py --frames <dataset_hk>
python -m onboard.run --backend mock --with-collector --frames <dataset_hk> --minutes 2

# 6. 미니PC: 설치하고, 아직 뭐가 없는지 확인
bash onboard/bin/setup_minipc.sh --check     # 보고만
bash onboard/bin/setup_minipc.sh             # 설치
python onboard/tools/doctor.py               # 준비상태

# 7. 팔만 / 몸통만 따로 (버스가 분리되어 있으므로 독립 시험 가능)
python -m onboard.tools.arm_bench                    # 목(mock)
python -m onboard.tools.arm_bench --backend real     # CAN만: 링크·enable·피드백
ARM=0 NIC=enp2s0 bash onboard/bin/run.sh             # 다리만 (팔 없이)

# 8. 실물
NIC=enp2s0 bash onboard/bin/run.sh
```

## 팔과 몸통은 미니PC에 따로 붙습니다

둘 사이에 데이터 링크는 없습니다. 팔은 몸통 위에 **기구적으로만** 얹혀 있고
(`kinematics.mount_xyz`), 좌표 변환은 미니PC 안에서 소프트웨어로 처리합니다.

| | 몸통 (A2 Pro, 12축) | 팔 (PiPER-X, 6축+집게) |
|---|---|---|
| 전송 | 이더넷 / DDS | CAN 1 Mbps |
| 인터페이스 | `enp2s0` 등 NIC | `can0` |
| 드라이버 | `oio/a2_dds.py` | `oio/piper_can.py` |
| 설정 | `configs/a2pro.yaml` | `configs/piperx.yaml` |
| 레이트 | 500 Hz 재발행 스레드 | 50 Hz 커맨드 |

두 드라이버는 서로를 import하지 않고 `run.py`의 `build()` 한 곳에서만 만납니다. 그래서
한쪽만 꽂아놓고 시험할 수 있고, **시험해야 합니다** — 케이블이 하나만 연결된 상태에서
"둘 다 안 됨"과 "한쪽만 안 됨"을 구분하지 못하면 5분 준비창을 거기에 씁니다.

* **다리만**: `ARM=0 bash onboard/bin/run.sh` (기존 `--no-arm`)
* **팔만**: `onboard/tools/arm_bench.py`. `onboard.run`은 LowState 없이는 시작을
  거부하므로(prep 첫 단계가 `legs.wait_for_state`) 팔만 꽂고서는 `run.py`로 들어갈 수
  없습니다. 그래서 별도 진입점입니다.

  기본은 **읽기 전용**입니다 — 링크·상태 레이트·관절 한계·집게 피드백까지만 보고
  모터는 건드리지 않습니다. 인수 직후 첫 질문은 "CAN이 살아 있나"이지 "팔이 닿나"가
  아니고, 첫 질문에 답하려고 주변 공간을 비울 필요는 없어야 합니다.
  움직임은 `--move`(home 램프 + 관절별 단독 왕복), `--grip`(집게 개폐)으로 켭니다.

## 구조

| | |
|---|---|
| `oio/` | 하드웨어 경계. DDS 다리, CAN 팔, 센서, 조종기, E-Stop, 그리고 **그 전부의 목** |
| `rt/` | 드리프트 없는 50 Hz 케이던스, 나이 도장(age-stamped) 메일박스 |
| `control/` | RMA 다리 정책, 팔 IK, 리미터, 안전 상태머신 |
| `estimate/` | 자세, 다리 오도메트리, 지도 없는 코스 필터 |
| `perceive/` | 카메라 정합, 깊이 합성, LiDAR 레일, 지각 프로세스 |
| `mission/` | 하이킹 FSM, 벽시계 예산 |
| `bringup/` | 5분 준비, 시작 게이트, go/no-go 자가진단 |
| `tools/` | 현장 도구와 위의 모든 검사 |
| `artifacts/` | 정책 파일 5종 + sha256 매니페스트 |

설계를 지탱하는 규칙 3개이고, `tools/offsim_gate.py`가 강제합니다.

1. `onboard/` 안에서 시뮬레이터를 import 하지 않는다.
2. 절대경로 금지.
3. 하드웨어 접근은 `oio/` 밖으로 나가지 않는다 — 그 위는 전부 목으로 대체 가능.

**3번이 핵심입니다.** A2 Pro는 체크인 데이까지 만질 수 없는 공용 장비입니다. 제어 루프·상태머신·
미션을 돌려보는 유일한 방법이 "로봇이 있어야 한다"라면, 그 전부가 **디버깅할 시간이 없는 그 하루에
처음으로 실행**되게 됩니다.

## 다시 쓰지 않고 재사용한 것

`course_mission/{mathx,navfuse,armkin,perception,catalogue,deploy}.py` 와
`eunha_rl/epkg/rma.py` 는 **그대로 import** 합니다 — 이미 Isaac 없이 돌고 이미 검증됐습니다.
`eunha_rl/epkg/__init__.py`가 gymnasium을 import 하므로 `rma.py`는 경로로 직접 로드하고
(`vendor/loadmod.py`), 이건 `scripts/check_mathx.py`가 이미 쓰는 방식입니다.
IsaacLab의 `DifferentialIKController`는 `vendor/diffik.py`로 **그대로 복사**했습니다 —
순수 torch(CPU 코어에서 1회 123 µs)이고, 무엇보다 **파지 정책이 학습된 바로 그 컨트롤러**입니다.

## 하드웨어 계약

`a2pkg/rough_env_cfg.py`, `epkg/rma.py`, `course_mission/mission.py` 세 곳을 교차검증한 값이며,
코드에 상수로 두지 않고 `configs/a2pro.yaml`에 한 번만 적습니다.

```
obs45 = [ ω_b·0.25 (3) | g_b (3) | cmd (3) | q−q_def (12) | q̇·0.05 (12) | a_{t−1} (12) ]
history 50 프레임(오래된→최신) → AdaptationNet (ReLU) → latent 8
actor([obs45, latent]) : 53 → 512 → 256 → 128 → 12   (ELU)
q_target = q_def + [0.125 hip | 0.25 thigh, calf] · a      50 Hz, kp 120, kd 4
관절 순서 = FR, FL, RR, RL × (hip, thigh, calf)
q_def = [-0.1, 0.8, -1.5,  +0.1, 0.8, -1.5,  -0.1, 1.0, -1.5,  +0.1, 1.0, -1.5]
```

여기서 **틀리기 쉽고 틀리면 비싼** 것 네 가지:

* `a_{t−1}`은 정책의 **원본 출력**입니다(스케일 전). 시뮬은 관절 목표까지 두 번에 걸쳐 갑니다
  (`a · scale/0.5` 후에 액션텀이 `default + 0.5·a`). 실기에선 위 한 줄로 접힙니다 —
  **0.5를 또 곱하면 보행 진폭이 반이 됩니다.**
* 적응망 활성함수는 **체크포인트에 없습니다.** ReLU여야 합니다 — ELU는 회전에서 속도 트래킹이
  붕괴합니다(실측 0/160).
* 집게는 시뮬에서 **대칭 프리즘 2관절**이라 시뮬의 "width"는 손가락 한 쪽이고 실제 개구는 그
  **2배**입니다. 변환은 `piper_can.py` 한 곳에만 있습니다.
* 학습된 파지 정책을 구동하는 것은 반드시 `armkin.jacobian_training_convention()`을 써야
  합니다. IsaacLab의 프레임 버그를 **일부러 재현**한 함수이고, "올바른" 자코비안으로 바꾸면
  기준선 2–3/6 대비 **0/5**였습니다.

## 센서

전방 카메라 1대 + IMU + 엔코더가 지각 입력의 전부입니다. `perception.py`의 모든 튜닝 상수는
640×360, fx = fy = 488.67, 주점 중앙, 마운트 (0.3381, 0.0336, 0.0525), 22° 하향에서 적합된
값입니다. 그래서 `perceive/camadapter.py`가 **실제 렌즈가 무엇이든 그 가상 핀홀로 정합**하고,
`perception.py`는 한 줄도 건드리지 않습니다. `tools/check_camadapter.py`가 실기 포즈 경로와
`perception.cam_pose_body`가 **2e-15 m**까지 같음을 증명합니다.

### 남은 실기 확인 3건과 각각의 대가

| | 어긋났을 때 |
|---|---|
| 내장 카메라가 깊이를 주는가 | **과제1은 무영향**(`detect_belt`는 깊이를 안 씀). 과제2는 타격 — 아래 표 |
| 실제 K·왜곡·화각은 | `camadapter`가 리맵. 체커보드 캘리브 필요 |
| 마운트가 정말 22° 하향인가, 영상–자세 동기는 | 피치 2° 오차 = 2 m에서 **0.31 m** 거리 오차 |
| LiDAR가 SDK로 접근 가능한가 | 카메라 단독 항법으로 폴백(시뮬에서 검증된 경로) |

**실제 녹화 코스 프레임 80장 측정** — 깊이 채널이 없어서 합성 깊이로 물체 검출을 돌릴 때.
재현: `python onboard/tools/bench.py --frames <dataset_hk> --limit 80 --depth-study`.

| 깊이 소스 | 검출 유지율 |
|---|---|
| 진짜 깊이 (대조군) | 100 % |
| 지면평면 단독 | 32 % |
| LiDAR 16 라인 | 49 % |
| LiDAR 32 라인 | 57 % |
| LiDAR 64 라인 | 76 % |
| LiDAR 128 라인 | 84 % |
| LiDAR + 지면평면 구멍 메움 | **19 %** |

**마지막 줄이 기억할 값입니다.** LiDAR 구멍을 지면평면으로 메우는 건 무조건 나아 보이지만
그냥 비워두는 것보다 2배 넘게 나쁩니다. `detect_arrays`는 *지면 위 높이*로 판별하는데, 메운
픽셀은 정의상 높이 0이라 **찾으려던 물체를 납작하게** 만듭니다. NaN은 정직하고, 평면 샘플은
자신만만한 오답입니다. 그래서 `depthsynth.synth()`는 메우지 않습니다.

## 타이밍

데스크톱 코어 1개 단일 스레드 실측에 N150의 E-코어 4–6배 패널티를 적용한 값. 400회 블록
9개의 중앙값이고 **공용 머신**이라 절대값은 ±10 % 로 보십시오 — 의미가 있는 건 서로의 비율과
N150 열입니다:

| | 이 코어 | N150 |
|---|---|---|
| 다리 제어 1스텝 (obs + 적응망 + actor) | 146 µs | 20 ms 예산의 **0.6–0.9 ms** |
| `detect_belt` | 5.3 ms | 21–32 ms |
| `detect_bins` | 7.1 ms | 29–43 ms |
| `detect_arrays` | 96 ms | 384–576 ms → **1.7–2.6 Hz** |
| 지면평면 깊이 | 1.9 ms | 8–11 ms |

항법 지각은 여유롭게 실시간이고 **예산을 먹는 건 물체 검출뿐**입니다. 그래서
`mission_hk.yaml`은 물체 검출 2 Hz, 벨트 검출 10 Hz로 돌립니다.

**지각은 스레드가 아니라 별도 프로세스에서 돕니다.** 스레드일 때는 멀쩡해 보였습니다 —
루프가 50.00 Hz를 유지했으니까요. 그러다 213 ms짜리 제어 틱이 하나 나왔고, **그 틱 자체의
작업은 0.25 ms**였습니다. 시간은 "스케줄되지 못한 것"에 쓰였습니다: 검출기의 연결요소·블롭
루틴이 순수 파이썬 루프라 도는 내내 GIL을 잡습니다. 스레드에서 200 ms 검출 프레임은 곧
**모터 명령 스트림의 200 ms 구멍**입니다. 검출기를 자식 프로세스로 옮긴 뒤 최악 틱은 32 ms.

## 안전

`ZERO_TORQUE → RAMP → STAND → RUN → (DAMP | FAULT)`, 그리고 `RUN`에서 갈라지는 `RECOVER`.

* **DAMP와 FAULT는 끈끈합니다(sticky).** 일반 미션 로직은 STAND를 요청할 이유가 많은데
  (예산 만료, 완주), DAMP 상태에서 그 요청은 **거부**됩니다. DAMP를 벗어나려면 RAMP를 거친
  명시적 재무장이 필요하고, FAULT는 거기에 사람이 추가로 필요합니다 — 규정이 보조 리셋을
  회계하는 방식과 정확히 같습니다.
* **댐핑은 패킷이 아니라 상태입니다.** 500 Hz 쓰기 스레드가 걸린 명령을 재발행하므로, DDS
  프레임 하나가 유실돼도 로봇이 보행 중 자세 목표를 붙잡고 있는 일이 없습니다.
  (참조 구현 `unitree_rl_gym`은 종료 시 댐핑을 **1회만** 발행합니다.)
* 중단 조건: LowState 정체(>100 ms), 자세 한계 초과, 조종기 abort 버튼, 하드웨어 E-Stop.
  네 가지 전부 `tools/replay_run.py`의 주입 시험으로 검증합니다.
* **자율 기립은 점수 항목입니다** — 사람이 손대면 −10, 스스로 일어나면 0.
* 나가는 모든 목표에 변화율·관절한계 클램프. 두 참조 스택에는 **없는** 층입니다.
  다리 상한은 평평한 숫자가 아니라 관절별 속도정격에서 유도합니다(`a2pro.yaml`의 `auto`):
  실제 정책 2399스텝 실측으로 스텝당 목표 변화가 p50 0.002 / p99 0.033 / max 0.43 rad이라,
  팔에 맞는 0.2 상한은 **정책이 균형을 되찾는 트랜지언트를 정확히 깎고 있었습니다.**

## 휘슬 전

규정은 5분을 **한 번** 줍니다. `bringup/prep.py`가
`통신 → 센서 → 모터 → 기립 → 추정기 원점 → 자가진단 → READY`를 단계별 시간과 함께 돌리고,
FAIL이 있으면 진행을 거부합니다. 지각 프로세스는 **제어 루프가 생기기 전에** 띄웁니다 —
`spawn`이 1.2 s이고, 나중에 하면 그 1.2 s가 그대로 명령 공백이 되기 때문입니다.

`bringup/startgate.py`가 런을 래치하며 되돌릴 수 없습니다. 게이트가 열리기 전에는 **아무것도
속도를 명령하지 않습니다** — 부정 출발은 시도 무효, 3회면 실격이라 조심해서 될 일이 아니라
구조로 막아야 합니다.

## 현장 첫 1시간

1. `bash onboard/bin/estop_test.sh` — 무엇이 움직이기 전에 먼저.
2. `python onboard/tools/dump_motor_map.py --nic <iface>` — **건너뛰지 마십시오.**
   정책 관절 순서는 Go2의 Unitree SDK 순서와 문자 그대로 같아서 매핑은 십중팔구 항등입니다.
   그런데 A2 Pro는 두 참조 스택이 다루지 않는 신형이고, **매핑이 틀려도 크래시하지 않습니다** —
   로봇은 서고, 스모크 테스트를 통과하고, **처음 회전하려 할 때 접힙니다.**
3. `python -m onboard.tools.arm_bench --backend real` — 팔 CAN 링크. 다리와 독립이므로
   순서를 기다릴 필요가 없고, 여기서 실패하면 piper_sdk의 CAN 활성화 스크립트 문제이지
   로봇 문제가 아닙니다. 공간이 확보되면 `--move --grip`으로 실제 구동까지.
4. `python onboard/tools/calib_extrinsics.py` — 체커보드, 피치 오차 **0.3° 미만** 목표.
5. `NIC=... bash onboard/bin/prep.sh` — 5분 시퀀스를 실제로 계측.

## 과제2: collector

`mission/collector.py`가 `attach_collector()` 자리를 채웁니다 — 대상 선택, 접근, 파지, 배달.
`--with-collector`로 켭니다. 물체 정체성은 시뮬 등록부가 아니라 `perceive/objtrack.py`에서,
파지 성공은 물체의 진짜 자세가 아니라 집게 폭·전류·리프트에서, 통 위치는 **마지막으로 본 곳**을
생 오도메트리로 실어 나른 값(`estimate/binmem.py`)에서 옵니다.

**점수를 기대하기 전에 이걸 보십시오.** 정직 항법 **+** 정직 물체검출(= 실기 조건)로 돌린
시뮬 18런 실측:

| | |
|---|---|
| 파지 시도 | 265회 |
| 파지 성공 | 5회 (1.9 %), **전부 캔** |
| 올바른 통 투입 | 1회 |
| 과제2 최고 | 20/160, 18런 중 17런이 0점 |

클래스 패턴은 운이 아닙니다. 출하 파지 체크포인트는 45–70 × 45–130 × 30–120 mm 범위 물체로
파인튜닝됐고, 그 안에 드는 건 캔(66 × 66 × 115)뿐입니다. rind(200 × 50 × 20)와
lunchbox(150 × 110 × 60)는 밖이고, 그들에겐 스크립트 핀치가 유일한 경로입니다. 별개로,
시도의 3분의 1이 물체에서 1 m 넘게 벗어난 곳을 조준했습니다 — 조준 오차는 이봉분포로
100 mm 미만 68 %, 1 m 초과 32 %이고 그 사이는 비어 있습니다. 이건 정밀도가 아니라
**다른 물체를 조준했다**는 신호입니다.

그래서 이 모듈의 절반은 `mission/ledger.py`입니다. 시도마다 **무엇을 근거로 했고 어디서
죽었는지**를 남깁니다 — 트랙 뷰 수, 뷰 산포, 뱅크 이탈, 폐합 시점 TCP 잔차, 집게 폭·전류·리프트,
통 사이팅의 나이와 이동거리. 가장 중요한 필드는 `reobs_delta`입니다: 접근할 때 믿은 추정과
크라우치에서 다시 잰 추정의 **불일치**. 실기엔 진리값이 없지만, 두 독립 관측의 불일치는 잴 수
있습니다.

`tools/check_collector.py --frames <dir>`가 녹화 프레임으로 트래커를 오프라인 채점하며,
**클래스 오류**(그 클래스 물체가 1 m 안에 없음 — 검출기 정밀도가 45–59 %)와 나머지의
**측위 오차**를 분리해 보고합니다. 둘은 정반대의 처방을 원하고, 하나의 숫자로 뭉뚱그리는 것이
정밀도 문제를 트래커 문제로 오인하는 경로입니다.
