# `docker/` — 실기 런타임을 이미지로

*(English: [README.md](README.md))*

미니PC에서 실제로 도는 것을 패키징한 것입니다. 코드는 [`onboard/`](../onboard/) 그대로이고,
이 디렉터리는 **그것이 로봇에 올라가고 그 뒤로 변하지 않게** 하는 방법입니다.

## 왜 기계 한 대짜리를 컨테이너로 하나

규정이 검사 시점에 코드·파라미터·시스템 구성을 잠그고 코드 봉인을 요구하는데 —
**"8월에 세팅한 그 기계에서 됐다"는 git 태그로 재현되지 않습니다.** 이미지 다이제스트는
됩니다. 미니PC 재설치가 `setup_minipc.sh` 한나절이 아니라 몇 분이 되고, 백업 장비가
"아마 같을" 두 번째 설치가 아니라 **동일한 바이트**를 돌립니다.

## 사용

```bash
bash docker/build.sh                       # 빌드, 짧은 git rev 로 태깅
bash docker/run.sh doctor                  # 무엇이 설치·설정·가정 상태인가
bash docker/run.sh identity                # git rev + 설정·가중치 다이제스트 (검수용)
bash docker/run.sh gate                    # 수용 게이트를 이미지 안에서
NIC=enp2s0 bash docker/run.sh estop        # 가장 먼저, 무엇이든 움직이기 전에
NIC=enp2s0 bash docker/run.sh motormap --nic enp2s0
NIC=enp2s0 bash docker/run.sh run --backend real --nic enp2s0 --with-collector
```

`build.sh`가 이미지 **다이제스트**를 출력합니다. 봉인 제출에는 태그가 아니라 그것을
적으십시오 — 태그는 옮길 수 있고 다이제스트는 못 옮깁니다.

## 플래그, 그리고 빼면 생기는 일

`run.sh`와 `compose.yml`이 이걸 들고 있습니다. 하나씩 빼면 전부 **다른 것처럼 보이는**
고장이 납니다:

| 플래그 | 왜 |
|---|---|
| `--network host` | DDS는 TCP 클라이언트가 아닙니다. CycloneDDS는 `192.168.123.x`에서 멀티캐스트로 로봇을 찾고, SocketCAN의 `can0`도 네트워크 인터페이스입니다. 브리지 네트워크에서는 로봇이 그냥 없고, 증상은 오류 없는 "no LowState"입니다 |
| `--ipc host` | 지각이 자식 프로세스입니다. 기본 64 MB `/dev/shm`은 부하에서 불투명하게 죽는 고전적 원인. 다만 `--ipc host`는 호스트의 `/dev/shm`을 그대로 쓰므로 `--shm-size`는 **무효**입니다 — 예전에는 나란히 지정돼 있어서 제공하지 않는 보호를 제공하는 것처럼 읽혔습니다 |
| `--cap-add SYS_NICE` | 제어 스레드가 `SCHED_FIFO`를 요청합니다. 없으면 **조용히** 일반 우선순위로 돕니다 — 살아남지만 지각 버스트가 50 Hz 모터 스트림의 지터가 됩니다 |
| `--ulimit rtprio=99` | 능력만으로는 부족하고 한도가 허용해야 합니다. **호스트에서는 실시간 우선순위를 아예 못 잡았고 컨테이너에서는 잡습니다** |
| `--device /dev/videoN` | 카메라가 SDK 채널이 아니라 UVC 장치일 때 |

## 이미지에 든 것과 안 든 것

소스 13 MB, 90개 파일 — `onboard/` + 런타임이 실제로 import 하는 `course_mission` 7개
모듈 + 경로 로드되는 `eunha_rl/epkg/rma.py`. 그 중 12 MB가 정책 가중치입니다. **루트**
`.dockerignore` 화이트리스트가 정확히 그 집합입니다.

이 파일은 컨텍스트 루트에 있어야 하는데 2026-09-09까지 그렇지 않았습니다. `docker/`에
있었고 Docker는 거기를 보지 않습니다(BuildKit의 다른 후보는
`docker/Dockerfile.dockerignore` 뿐). 그래서 화이트리스트가 무효였고, 컨텍스트는 `.git`
154 MB를 포함한 레포 전체 485 MB였으며, `COPY onboard/`가 추적되지 않는 `__pycache__`를
다이제스트가 의미를 가져야 할 이미지에 구워 넣고 있었습니다. 옮긴 뒤 실측: 13.04 MB,
90개 파일, `__pycache__` 없음, 로그 없음.

안 든 것: 시뮬레이터, 학습 트리, 레포 README, 녹화 데이터셋. `check_docs.py`는 런타임
이미지를 인식해 체크아웃을 전제한 검사를 건너뜁니다. (`doctor.py`에는 그런 분기가 없고
필요도 없습니다 — 검사 대상 중 체크아웃에 의존하는 것이 없습니다.)

빌드는 이미지 안에서 `onboard/tools/offsim_gate.py`가 통과하지 못하면 **실패합니다.**
자기 수용 게이트를 못 넘는 배포 이미지는 존재하면 안 됩니다.

## 의존성 고정

`unitree_sdk2py`가 `opencv-python`을 의존합니다 — 핀 없이, 그것도 GUI 빌드로. 설치하면
환경에 OpenCV가 둘이 되고 `import cv2`는 나중에 설치된 쪽으로 해석되며 버전이 빌드마다
떠다닙니다. 규정이 검사 시점에 잠그는 코드에서 전이 의존성이 혼자 버전을 바꾸는 건
결함입니다: **테스트한 이미지와 배포하는 이미지가 다릅니다.** 이제 핀 헤드리스 빌드를
강제 재설치하고 **모든 핀을 대조해 드리프트에서 빌드가 실패합니다.**

`unitree_sdk2py`가 요구하는 버전의 CycloneDDS는 소스 전용입니다 — 0.10.2 휠이 없고,
pip 오류(`Could not locate cyclonedds`)는 원인이 아니라 증상을 말합니다. C 라이브러리를
먼저 만들고 `CYCLONEDDS_HOME`을 pip보다 먼저 세웁니다. 그 순서를 틀리는 것이 이 설치가
실패하는 가장 흔한 방식입니다.

## 동등성: 컨테이너가 같은 것을 계산하는가

컨테이너화는 스택을 다른 Python, 다른 torch 빌드, 다른 BLAS 위로 옮깁니다.
`onboard/tools/parity.py`가 고정 입력에서 제어 경로의 모든 양을 지문으로 남기고 대조합니다:

```bash
python onboard/tools/parity.py --out host.json
LOGS=$PWD bash docker/run.sh parity --out /logs/container.json   # /logs -> /app/onboard/logs
python onboard/tools/parity.py --compare host.json container.json
```

2026-09-09 재측정 — 호스트(python 3.11.15, torch 2.7.0+cu128, numpy 1.26.0) 대
이미지(python 3.11.16, torch 2.5.1+cpu, numpy 2.1.3):

| | |
|---|---|
| 정책 가중치 | 6/6 sha256 동일 |
| 제어 경로에서 지문을 뜬 모든 양 | **20/20 비트 단위 동일** |
| 최대 차이 | 0.0 |

이는 이 페이지의 이전 실측(13/20 비트 동일, torch 항목 최대 3.8e-6 차이, 최악의 경우
집게 개구 100 mm에 대해 TCP 지령 변위 76 나노미터)보다 강한 결과이고, 코드 변경 때문이
아닙니다 — 양쪽이 BLAS를 단일 스레드로 고정하면 두 빌드에서 내적의 합산 순서가 같아지기
때문입니다. 20/20은 이 기계 쌍의 현재 측정값으로 받아들이되 보장으로 보지는 마십시오.
호스트 BLAS가 달라지면 torch 항목이 다시 1e-6 수준의 차이로 돌아갈 수 있고, 그것도 이미
하드웨어가 표현할 수 있는 범위보다 한참 아래였습니다.

**GPU는 쓰지도 필요하지도 않았습니다.** 런타임이 설계상 CPU 전용이므로 동등성도 CPU
문제입니다.
## 런북: 탁자 위의 로봇에서 주행까지

**단계보다 시간 예산을 먼저 읽으십시오.** 이 런북은 예전에 "주최측이 제공하는 12시간의
장비 테스트"를 전제로 0:00→12:00 연속 일정으로 짜여 있었습니다. 그 숫자는 실재하지만
**별도 테스트 장소에서 온라인 예약으로 쓰는 경기 *이전* 시간**입니다(0907 안내서 §VII.8.
홍콩은 2026-08-10 – 09-04, 예약 마감 09-03). 경기장에서의 시간이 아니고, 첫 코스 주행이
5:30에야 나오는 계획은 실제로 주어지는 창에서 실행할 수 없습니다.

당일 실제로 받는 시간(0907 §VII.8):

| 언제 | 얼마 | 무엇에 쓰는가 |
|---|---|---|
| 경기 이전 | 예약제, 별도 장소 | 기간 내 예약하지 않으면 통째로 소멸 |
| **체크인 데이** | **2시간** | 장비 확인, 팀 장비 설치, 알고리즘 배포, 테스트·디버깅. 사전 예약 필수 |
| **경기 당일** | 배정 시각 전 **1시간 이상** | 확인, 팀 장비 설치, 전원 투입, 프로그램 배포, 테스트 |

그 다음이 준비 5분 + 과제 30분이고, 시계는 멈추지 않습니다(§IV.3).

아래 두 단계는 **게이트**입니다 — 그게 틀리면 그 뒤는 전부 무의미하고, 그 중 하나는
조용히 실패합니다. 2시간 안에 체크인 데이 전부가 들어가야 하므로 순서가 전보다 더
중요합니다.

측정값은 이미지의 configs 위에 마운트하는 디렉터리로 들어갑니다. 캘리브레이션 한 번에
이미지를 다시 굽지 않아도 되도록:

```bash
cp -r onboard/configs ./venue-configs      # 당일 시작에 한 번
export CONFIGS=$PWD/venue-configs NIC=<iface>
```

### 호스트 사전 준비 (미니PC, 이 모든 것에 앞서)

아래 어느 것도 이것 없이는 돌지 않고, `setup_minipc.sh`가 설치해 주지 않습니다:

```bash
docker --version                       # 없으면 `run.sh`가 먼저 거부합니다
docker load < atec-onboard.tar.gz      # USB로 들고 온 이미지
bash ~/src/piper_sdk/can_activate.sh can0 1000000   # 팔의 CAN 인터페이스
```

`can_activate.sh`는 PiPER SDK에 들어 있습니다. 미니PC의 `~/src/piper_sdk`에 클론하십시오.
`can0`가 내려가 있으면 컨테이너가 실기 백엔드 실행을 거부하고 한 줄로 알려줍니다 —
다리만 돌리려면 `ARM=0`.

### 당일 전에 (당일이 아니라 지금 전부)

```bash
bash docker/build.sh                                  # 태그 + 다이제스트
bash docker/run.sh identity                           # git rev, 설정·가중치 다이제스트
python onboard/tools/parity.py --out host.json
LOGS=$PWD bash docker/run.sh parity --out /logs/container.json
python onboard/tools/parity.py --compare host.json container.json
docker save atec-onboard:latest | gzip > atec-onboard.tar.gz   # USB로 들고 갑니다
```

미니PC에서 `docker load < atec-onboard.tar.gz`, 그 다음 `bash docker/run.sh doctor`.
로봇에 의존하는 행 말고는 전부 이미 초록이어야 합니다. **여기서 아낀 1분이 체크인
데이에는 없는 1분입니다.**

### 체크인 데이 — 2시간

#### 0:00–0:10 — 장착과 E-Stop, 전원 넣기 전

미니PC는 **지정된 구멍만** 써서, 이탈 방지 리테이너와 네트워크 케이블 스트레인 릴리프와
함께 장착합니다 — 규정이 신고한 방식을 요구하고 검수합니다. 검수받은 독립 안전 E-Stop
링크는 오프보드 장비로의 유일하게 허용되는 물리 연결입니다(§IV.4). 그 다음, 무엇이든
움직일 수 있게 되기 전에:

```bash
NIC=$NIC bash docker/run.sh estop
```

**게이트 1. 건너뛰지 말고 나중으로 미루지 마십시오.** 주행 중 E-Stop 작동은 수동 지원
1회로 계산되고(−10, 런당 3회), 작동하지 않는 E-Stop은 로봇 값을 합니다.

#### 0:10–0:25 — 링크와 모터 맵, 로봇은 받침대 위에 발이 뜬 채로

로봇 내장 상위 제어기(`sport_mode`)를 **끄십시오**(댐핑 상태에서 `L2+R2`, 또는 앱).
켜져 있으면 이 스택이 보내는 모든 LowCmd와 싸우고, 증상은 잠깐 떨다가 아무것도 안 하는
로봇입니다.

```bash
sudo ip addr add 192.168.123.99/24 dev $NIC && sudo ip link set $NIC up
ping -c1 192.168.123.161
CONFIGS=$CONFIGS NIC=$NIC bash docker/run.sh motormap --nic $NIC --watch   # 전송 확인
CONFIGS=$CONFIGS NIC=$NIC bash docker/run.sh motormap --nic $NIC           # 맵 자체
CONFIGS=$CONFIGS bash docker/run.sh doctor    # 모터 맵이 `ok  measured: [...]`여야 합니다
```

`--watch`는 보이는 모든 모터를 흘려보냅니다. 아무것도 안 나오면 메시지 타입이 틀린
것입니다 — `venue-configs/a2pro.yaml`에 `msg_type: hg`를 넣고 다시 하십시오. 프롬프트가
나오면 관절을 하나씩 움직이고, 출력된 yaml을 `venue-configs/a2pro.yaml`에 붙여넣습니다.

**게이트 2, 그리고 위험한 쪽.** 틀린 맵은 죽지 않습니다. 로봇은 서고, 스모크 테스트를
통과하고, 처음 도는 순간 접힙니다. 그 행이 `ok`가 되기 전에는 걷게 하지 마십시오.

`ip addr` 줄은 재부팅을 넘기지 **못합니다.** 전원을 껐다 켤 때마다 다시 하십시오.

#### 0:25–0:35 — 첫 기립

받침대에서 내려 바닥에, E-Stop을 든 사람과 함께.

```bash
CONFIGS=$CONFIGS NIC=$NIC bash docker/run.sh prep
```

`ZERO_TORQUE → RAMP`(2초) → `STAND`, 그리고 유지. 어느 관절이든 반대로 구동되면 즉시
중단 — 그건 게인이 아니라 `joint_sign` 문제입니다.

#### 0:35–0:55 — 카메라

```bash
CONFIGS=$CONFIGS bash docker/run.sh calib --capture      # 체커보드 20뷰 정도
CONFIGS=$CONFIGS bash docker/run.sh calib --intrinsics   # -> sensors.yaml 에 붙여넣기
CONFIGS=$CONFIGS bash docker/run.sh calib --extrinsics <image>   # 보드를 바닥에 평평하게
```

이 명령들은 바인드 마운트인 `onboard/logs/calib/`에 씁니다 — 그래서 뷰가 컨테이너보다
오래 남고 다음 명령이 읽을 수 있습니다. 2026-09-09까지는 아니었습니다: 마운트는 `/logs`
인데 도구는 `/app/onboard/logs`에 써서, `--capture`가 컨테이너와 함께 삭제되는 디렉터리를
채우고 `--intrinsics`는 빈 곳을 봤습니다.

피치 오차 **0.3°** 미만을 목표로 하십시오. 2°는 2 m에서 0.31 m의 거리 오차이고, 이는 파지
허용오차 전체보다 큽니다. **카메라가 깊이를 주기는 하는지**도 여기서 확정하고
`camera.depth.source`를 그에 맞추십시오.

#### 0:55–1:05 — 팔

```bash
bash ~/src/piper_sdk/can_activate.sh can0 1000000
CONFIGS=$CONFIGS bash docker/run.sh doctor       # CAN 인터페이스가 `ok` 여야 합니다
```

그 다음 그리퍼를 특성화합니다: 물체 종류마다 닫아 보고 폭과 전류를 기록 — `HoldDetector`가
문턱으로 쓰는 값입니다. `piperx.yaml`의 밴드는 시뮬레이션에서 맞춘 값입니다.

#### 1:05–1:45 — 코스 주행

```bash
CONFIGS=$CONFIGS NIC=$NIC bash docker/run.sh run --backend real --nic $NIC
```

과제1을 먼저, 단독으로. 80점이고, 나머지 160점의 전제조건이고, 실측 기록이 있는 쪽입니다.
매 실행이 호스트에 `onboard/logs/<stamp>/{report.json,ledger.json,run.log}`를 남깁니다.

과제1이 단단하고 시간이 남을 때에 **한해서** `--with-collector`를 붙이십시오. 매 시도 후
`onboard/logs/<stamp>/ledger.json`을 읽으십시오. 각 시도가 어느 단계에서 죽었는지 적혀
있고, `reobs_delta`가 300 mm를 넘으면 로봇이 엉뚱한 물체로 걸어간 것입니다.

#### 1:45–2:00 — 동결

측정한 설정을 이미지 것 위에 복사하고, 한 번 다시 굽고, 신원을 기록합니다:

```bash
cp venue-configs/*.yaml onboard/configs/ && bash docker/build.sh
bash docker/run.sh identity
docker image inspect atec-onboard:latest --format '{{.Id}}'
```

그 빌드의 다이제스트가 실제로 돈 것이고 인용해야 할 값입니다. **설정을 마운트해서 덮어쓴
채 한 주행은 이미지만으로 재현되지 않습니다** — `identity`가 *유효한* 설정에 대한 sha256을
출력하는 이유가 바로 그것이고, 덮어쓰기를 보이지 않는 사실이 아니라 기록된 사실로
만듭니다.

### 경기 당일 — 배정 시각 전 1시간 이상

§IV.5: 검수가 끝나면 코드·알고리즘 파라미터·시스템 설정이 **잠깁니다.** 수정 금지,
파라미터 조정 금지, 펌웨어 재기록 금지, 연결된 디버깅 장비를 통한 기록 금지 — 위반하면
해당 주행이 무효가 되고 중대한 경우 실격입니다. 예비 장비도 같은 코드·설정을 써야 합니다.
그러므로 이 1시간은 확인 전용입니다:

```bash
docker load < atec-onboard.tar.gz      # 새 미니PC나 예비기일 때만
bash docker/run.sh identity            # 봉인한 값과 일치해야 합니다
NIC=$NIC bash docker/run.sh estop      # E-Stop, 다시 한 번
sudo ip addr add 192.168.123.99/24 dev $NIC && sudo ip link set $NIC up
CONFIGS=$CONFIGS NIC=$NIC bash docker/run.sh prep
```

그 다음 휘슬 전 5분 준비 창에서 주행을 띄우고 게이트에서 대기시킵니다. **휘슬 전에 로봇이
움직이면 그 시도는 무효이고, 부정 출발 3회 누적은 실격입니다**(§IV.3). `--gate manual`이
컨트롤러 조합까지 붙잡고 있는 이유이고, `--gate timer`가 리허설 전용인 이유입니다.

### 안 될 때

| 증상 | 먼저 볼 것 |
|---|---|
| LowState가 아예 안 옴 | `sport_mode`가 아직 도는지, 그 다음 `msg_type: hg` |
| 서긴 하는데 첫 회전에서 접힘 | 모터 맵 — 3단계로 돌아가십시오 |
| 어느 관절이 반대로 감 | `a2pro.yaml`의 `joint_sign` |
| 검출 위치가 엉뚱함 | 카메라 피치, 그 다음 깊이가 진짜인지 |
| 집게가 빈 공간을 쥠 | 알려진 97 % 실패입니다 — 코드가 아니라 장부를 읽으십시오 |
| 제어 루프 late/overrun | 실행에서 `--cap-add SYS_NICE`와 `--ulimit rtprio`가 빠졌는지 |
| `ConnectionError: ['can0' Interface __init__ ERROR]` | `can0`가 내려가 있음 — `can_activate.sh`, 또는 다리만 돌리려면 `ARM=0`. 컨테이너가 먼저 확인하고 한 줄로 알려줍니다 |
| `crc_amd64.so: cannot open shared object file` | 2026-09-09 이전에 구운 이미지. `unitree_sdk2py`의 `setup.py`가 프리빌트 CRC 라이브러리를 설치하지 않았습니다. 이제 빌드가 복구하고 로드되는지 단언합니다 |
| 되던 verb가 `NIC: set NIC ...` | `NIC=<iface>`를 주십시오. `run.sh`가 컨테이너로 전달합니다(2026-09-09 이전에는 안 했습니다) |
| 주행 후 ledger.json / report.json 없음 | 호스트의 `onboard/logs/<stamp>/` 아래에 씁니다. 시도가 0건인 주행도 그렇다고 적힌 ledger를 남깁니다 |

## 아직 실기 준비완료는 아닙니다

이미지는 **설치** 문제를 없앱니다. **계측** 문제는 없애지 못합니다 — 컨테이너 안의
`doctor.py`도 모터맵·메시지형·카메라 마운트·카메라 백엔드를 **ASSUMED**로, 내부파라미터와
E-Stop 라인을 **UNCONFIGURED**로 보고합니다. 그건 로봇이 있어야 합니다:

```
1. bash docker/run.sh estop                          # 무엇이 움직이기 전에
2. bash docker/run.sh motormap --nic <iface>         # 첫 번째이고, 건너뛰지 말 것
3. bash docker/run.sh calib --capture
4. NIC=<iface> bash docker/run.sh prep
```

맵이 틀려도 크래시하지 않습니다. 로봇은 서고, 스모크 테스트를 통과하고, 처음 회전할 때
접힙니다.
