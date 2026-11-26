# 물체 집기 실시간화 계획 — 미니PC(Beelink S12 Pro / Intel N100)

## 하드웨어 확정
- **CPU**: Intel N100, **4코어 4스레드** (Alder Lake-N, E-코어만), 16GB DDR4
- **GPU**: Intel UHD 24EU iGPU만 — **연산 가속 실익 없음**
- bench.py가 가정한 N150과 동급. 데스크톱 코어 대비 **4~6배 느림**(E-코어 + 순수파이썬).

## 진단 (코드 근거)
집기 = 인지(perception) + 집기정책(grasp policy).
- **집기정책은 무죄**: `onboard/mission/grasp.py`의 42→7 MLP, 프레임당 μs급.
- **범인은 인지**: 순수 파이썬 파이프라인이 N100에서 못 따라감 → 인지 프레임레이트↓ →
  `objtrack`의 `min_views≥2`(objtrack.py:132)를 못 채워 track이 engageable 안 됨 →
  집기 시도 자체를 못 하거나 낡은 위치로 감.
- 제어 50Hz 끊김은 이미 해결됨(인지 별도 프로세스, procworker.py). **남은 건 인지 처리량.**

### 핫스팟 (프레임당 비용)
1. `course_mission/perception.py:1130 _blobs` — connected-components를 **순수 파이썬
   dict + union-find**로. N100 E-코어에 최악. GIL 오래 점유.
2. `perception.py:208 label_map` — 230k 픽셀 × 프로토타입 수만큼 전체프레임 float 연산.
3. detect_arrays / detect_belt / detect_bins 모두 위 둘을 호출.

## 수정안 (우선순위·리스크 순)

### ① _blobs → cv2.connectedComponentsWithStats (최대효과/최소리스크)
- 순수파이썬 union-find를 C 구현으로. blob 단계 10~50배, GIL 구멍 제거.
- **결과 동등성 유지**: 현재 12px 그리드 양자화 + 8-이웃 병합의 *출력*(각 blob의 (ys,us))만
  같으면 하류 로직 불변. cv2 라벨맵에서 동일 필터(min_pix, max_blobs, min_cell 상당) 재현.
- **cv2 없으면 기존 numpy 경로 fallback** (import 실패 가드).
- 위험: min_cell(스페클 브리징 방지) 의미를 cv2 경로에서 재현해야. → 8-연결 라벨 후
  컴포넌트 크기 필터로 등가 구현.

### ② label_map 벡터화 압축
- 프로토타입을 배열로 스택 → 브로드캐스트 거리 → 한 번의 argmin. float32 유지.
- 중간 효과, 결과 bit-동등에 가까움.

### ③ 인지 detect 경로 다운스케일 640×360 → 320×180
- detect 경로만 절반. 픽셀 1/4 → label_map·blobs 선형 단축.
- MIN_PIX/POST_MIN_PIX 등 픽셀 임계 동반 스케일 필요.
- **A/B 필수**: 원거리 소물체 recall 하락 가능. cfg 플래그로 on/off.

### ④ detect_hz 상향 (①②③ 이후)
- 프레임당 비용이 내려간 만큼 detect_hz 2→3~4로 올려 min_views를 빨리 충족.
- sensors.yaml 한 줄. 현장에서 부하 보며 조정.

## 검증 (미니PC 없이 지금 / 현장에서)
- **지금(desktop)**: bench.py 확장 — 기존 vs ①②③ 각각의 프레임당 ms + **detection 동등성**
  (같은 프레임에서 track 위치/개수 diff). N100_PENALTY 곱해 현장 예측.
- **현장**: 미니PC 붙이면 bench.py 실측 → 브랜치의 cfg 플래그로 ①(항상 on)/③/④만 A/B.

## 작업 규칙
- 팀 공유 main. **새 브랜치 + 사본 원칙**. course_mission/perception.py는
  cv2 경로를 *추가*하고 numpy 경로는 fallback으로 보존(제거 금지).
- 각 최적화는 cfg 플래그 뒤에 둬 현장에서 즉시 롤백 가능.
