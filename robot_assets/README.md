# robot_assets — 로봇 모델

## a2pro_piperx

로봇 플랫폼 변경: **B2 → A2 Pro + Piper X**

URDF→USD 합성으로 20-DoF articulation asset 완성.
핵심은 **FixedJoint 및 arm root API 제거**.

| 스크립트 | 용도 |
|---|---|
| `compose_a2_piperx.py` | 두 URDF를 하나의 articulation으로 합성 |
| `verify_articulation.py` | 관절 구조 검증 |
| `check_arm_stow.py`, `sweep_arm_stow.py` | 팔 접힘 자세 탐색 |
| `check_collision.py` | 자기충돌 확인 |
| `show_standing.py` | 기립 자세 시각화 |

## ⚠️ 관절 매핑

**identity 매핑이다. 순열을 적용하지 말 것.**
과거에 순열을 넣었다가 옆밀림 버그를 만든 적이 있다.

※ USD/URDF 실제 에셋 파일은 용량 때문에 저장소에서 제외됨 (서버 원본 참조)
