# realworld — 실기예선 통합 스택

홍콩 실물 대회(A2 Pro + PiPER)용. **세 작업 영역 중 유일하게 문서화된 코드베이스.**

의도적으로 **두 스택을 한 저장소에** 담았다:

| 스택 | 역할 | 의존성 |
|---|---|---|
| 시뮬레이터 | 알고리즘이 맞다는 걸 증명 | IsaacLab + GPU |
| **온보드** (`gmis_stack/onboard/`) | 로봇에서 **실제로 도는** 코드 | Isaac·CUDA **전부 미사용** |

온보드 타깃은 Beelink 미니PC (Intel N150, 4 E-core, 12GB, **GPU 없음**).

## 성적

Task 1 **80/80**, Task 2 **120/160** (정직한 항법·물체 인식 기준)

## 주요 진입점

- `gmis_stack/onboard/run.py` — 미션 진입점
  - `--backend mock` : 데스크톱 리허설
  - `--backend real --nic enp2s0` : 실제 로봇
  - 백엔드 선택만이 유일한 차이. 로봇은 공용 장비라 체크인 전엔 못 만지기 때문
- `gmis_stack/onboard/tools/offsim_gate.py` — 수락 게이트 (시뮬레이터·GPU·절대경로 금지 확인)
- `gmis_stack/deploy/run_mission.sh` — 정책 교체 가능한 전체 미션

## ⚠️ run_mission.sh 의 `{ }`

스크립트 전체가 중괄호로 감싸여 있다. **지우지 말 것.**
bash는 파일을 바이트 오프셋으로 게으르게 읽어서, 실행 중 파일을 수정하면
오프셋이 밀려 **미션이 두 번 실행된 사고**가 있었다 (`0903_2222_svO`).

## 미니PC 병목

`gmis_stack/GRASP_PERCEPTION_SPEEDUP_PLAN.md` 참조.
파지 정책은 무죄(42→7 MLP, 마이크로초). **인지가 범인.**
핫스팟: `perception.py:1130 _blobs` (순수 파이썬 union-find, GIL 점유),
`perception.py:208 label_map` (23만 픽셀 × 프로토타입).
→ detect_hz 1.2 → 5Hz
