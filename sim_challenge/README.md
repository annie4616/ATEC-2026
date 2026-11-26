# sim_challenge — 시뮬레이션 챌린지 (예선)

IsaacLab 기반 온라인 예선. **93팀 중 10위**로 홍콩 실물 대회 진출.

| 디렉터리 | 내용 |
|---|---|
| `atec_rl_lab_workspace/` | 공식 스타터 repo + 우리 확장. Task A/D/E 환경·학습·평가·제출 |
| `task_e/` | Task E (pick-and-place) 독립 작업. state machine + ACT |
| `taskd_final_controller/` | Task D 최종 확정 컨트롤러 (3회 재현 검증 완료) |

## 진입점

```
atec_rl_lab_workspace/scripts/
├── rsl_rl/        PPO 학습 (train.py, play.py)
├── act/           모방학습 ACT (collect_demos_*, train_task_*, eval_*)
├── steering_head_wz/   Task A 조향 헤드 train/eval 쌍
├── steering_ytrack/    Task A y-추종·배포
├── task_d/        Task D 진단 diag_* 21개
└── common/        환경 확인·녹화 유틸
```

## Task D 최종 해법

`taskd_final_controller/push_controller_cross_FINAL_LOCKED_box050_bail190_v230.py`

6단계 FSM: `BACK_UP → ALIGN_Y → PUSH_Y → ALIGN_X → PUSH_X → CROSS`

핵심 통찰 — **+x로 밀면 박스가 플랫폼 벽에 낀다. 먼저 −y로 빼내야 한다.**
yaw를 0으로 고정해 body-frame 명령이 월드 좌표에 바로 대응되게 했다.

확정값: `BOX_PUSH_DONE_X=0.5`, `PUSH_X_BAIL_RX=-1.90`, `speed=2.30`, `y tol=0.18`
→ 3회 반복 전부 `max_x=+3.51 CROSSED, steps=1327` (완전 결정적)
