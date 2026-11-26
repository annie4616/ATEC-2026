"""
Task B (Garbage Collection) — solution.py 의 "crouch 안정화" 변형본.

⚠️ 중요 — 데모 수집기의 "베이스 핀 고정(write_root_state_to_sim)" 기법은 여기서
   쓸 수 없다.  solution 은 배포 정책이라 predicts(obs)가 돌려주는 관절 액션만
   적용될 뿐, env/scene/robot 핸들이 없어 베이스를 sim 에 직접 고정(teleport)할
   방법이 없다.  따라서 crouch 중 무너짐/가라앉음/물리오류는 *제어*로만 완화한다.

solution.py 대비 바뀐 점 (그래스프 목표각은 그대로 두어 reach 기하는 보존):
  1) crouch / reach 를 더 느리고 부드럽게(ease) — 급격한 전이로 인한 전복 방지.
  2) manip 상태(CROUCH/REACH/HOLD/RECOVER)에서 per-step 액션 rate-limit — 갑작스런
     관절 점프로 인한 PhysX 불안정("물리 오류") 방지.  보행(PATROL/APPROACH)은
     클램프하지 않음(걸음 gait 는 빠른 변화가 필요).
  3) crouch 후 안정화 hold 를 늘림.
  4) 2단계 RECOVER: 팔을 먼저 회수한 뒤(다리는 크라우치 유지) 일어선다 — 팔이
     앞으로 뻗은 채 일어서다 앞으로 고꾸라지는 것 방지.
  5) (선택) 크라우치를 좌우/전후 대칭으로 둘 수 있게 분리 — 비대칭 crouch 는
     몸통을 기울여 COM 이 지지면을 벗어나 전복을 유발한다.

여전히 무너지면: 크라우치를 더 얕게(CALF_* 를 0 에 가깝게, THIGH 작게) 하거나
ramp 를 더 늘려라.  근본적으로 정적 deep-crouch 는 균형기가 없어 한계가 있다.

action 규약 (solution.py 와 동일):
  - leg/arm 모두 JointPositionAction, scale=0.5, use_default_offset=True
  - joint_target = default + scale*action  →  action = (target - default)/scale
"""

import os
import math
import torch
import numpy as np
from typing import Any

# ═══════════════════════════════════════════════════════════════════
# 카메라 검출 모듈 (solution.py 와 동일 — 단일 파일 배포용 인라인)
# ═══════════════════════════════════════════════════════════════════


class CameraIntrinsics:
    def __init__(self, focal_length_mm, horizontal_aperture_mm, width=640, height=480):
        self.fx = focal_length_mm / horizontal_aperture_mm * width
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.width = width
        self.height = height

    @classmethod
    def head(cls):
        return cls(24.0, 20.955)

    @classmethod
    def ee(cls):
        return cls(15.0, 20.955)


HEAD_CAM_POS = np.array([0.4216099977, 0.0250000004, 0.0618509986], dtype=np.float32)
_M = np.array([[0., 0., 1.],
               [-1., 0., 0.],
               [0., -1., 0.]], dtype=np.float32)
_th = np.pi / 6.0
_Ry = np.array([[np.cos(_th), 0., np.sin(_th)],
                [0., 1., 0.],
                [-np.sin(_th), 0., np.cos(_th)]], dtype=np.float32)
HEAD_CAM_R = _Ry @ _M


def head_camera_point_to_base(p_opt):
    return HEAD_CAM_R @ np.asarray(p_opt, dtype=np.float32) + HEAD_CAM_POS


def fit_ground_plane(pts, max_iter=2, ground_band=0.08):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 10:
        return np.array([0.0, 0.0, float(np.median(pts[:, 2]))], dtype=np.float64)
    mask = np.ones(len(pts), dtype=bool)
    coef = np.array([0.0, 0.0, float(np.median(pts[:, 2]))], dtype=np.float64)
    for _ in range(max_iter):
        p = pts[mask]
        if len(p) < 10:
            break
        a_mat = np.c_[p[:, 0], p[:, 1], np.ones(len(p))]
        coef, *_ = np.linalg.lstsq(a_mat, p[:, 2], rcond=None)
        h = height_above_ground(pts, coef)
        mask = np.abs(h) < ground_band
    return coef


def height_above_ground(pts, coef):
    a, b, c = coef
    pts = np.asarray(pts, dtype=np.float64)
    return pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)


def pixel_to_camera_point(u, v, depth, intr: "CameraIntrinsics"):
    z = float(depth)
    x = (u - intr.cx) / intr.fx * z
    y = (v - intr.cy) / intr.fy * z
    return np.array([x, y, z], dtype=np.float32)


def detect_objects_from_camera(obs, intr_head=None, intr_ee=None,
                               depth_min=0.2, depth_max=8.0,
                               ground_height_min=0.02, grid_step=12,
                               min_cluster=2):
    """depth/RGB로 미수거 물체를 찾아 robot-frame 상대 위치 목록을 반환."""
    if intr_head is None:
        intr_head = CameraIntrinsics.head()

    image = obs.get("image") if isinstance(obs, dict) else None
    if not image:
        return []

    depth = image.get("head_depth")
    if depth is None:
        depth = image.get("video_depth")
    if depth is None:
        return []
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth)
    if depth.ndim == 4:
        depth = depth[0, :, :, 0]
    elif depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        return []

    H, W = depth.shape
    valid = (depth > depth_min) & (depth < depth_max) & np.isfinite(depth)
    if not valid.any():
        return []

    samples = []
    for v in range(0, H, grid_step):
        for u in range(0, W, grid_step):
            d = depth[v, u]
            if not (depth_min < d < depth_max) or not np.isfinite(d):
                continue
            p_cam = pixel_to_camera_point(u, v, d, intr_head)
            pb = head_camera_point_to_base(p_cam)
            samples.append((u, v, float(d), pb))
    if not samples:
        return []

    bs = np.array([s[3] for s in samples], dtype=np.float32)
    coef = fit_ground_plane(bs)
    heights = height_above_ground(bs, coef)

    candidates = []
    for i, (u, v, d, pb) in enumerate(samples):
        if heights[i] > ground_height_min:
            candidates.append((u, v, d, pb, float(heights[i])))
    if not candidates:
        return []

    objects = []
    used = [False] * len(candidates)
    for i, (u0, v0, d0, p0, h0) in enumerate(candidates):
        if used[i]:
            continue
        cluster = [candidates[i]]
        used[i] = True
        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue
            uj, vj, dj = candidates[j][0], candidates[j][1], candidates[j][2]
            if abs(uj - u0) < 60 and abs(vj - v0) < 60 and abs(dj - d0) < 0.3:
                cluster.append(candidates[j])
                used[j] = True
        if len(cluster) < min_cluster:
            continue
        cu = np.mean([c[0] for c in cluster])
        cv = np.mean([c[1] for c in cluster])
        cd = np.median([c[2] for c in cluster])
        ch = float(np.median([c[4] for c in cluster]))
        p_cam = pixel_to_camera_point(cu, cv, cd, intr_head)
        pb = head_camera_point_to_base(p_cam)
        pos_robot = (float(pb[0]), float(pb[1]), float(pb[2]))
        dist = math.hypot(pos_robot[0], pos_robot[1])
        objects.append({"pos_robot": pos_robot, "dist": dist,
                        "pixel": (cu, cv), "height": ch})

    objects.sort(key=lambda o: o["dist"])
    return objects


# ═══════════════════════════════════════════════════════════════════
# 색(HSV) 기반 탐지 — depth 지면피팅보다 강건.
#   근거(logs/head_rgb_50.png 실측, Task B 헤드뷰, DR/노이즈 OFF):
#     · 바닥 97.5% 가 sat<0.25 (무채색)  · 노란 물체 hue 51~59, sat 0.5~0.87, val 0.72~0.92
#     · 통은 주황(hue~30) → 노랑 밴드에서 자동 제외
#   설계: 노란 픽셀=물체, 주황 픽셀=통(장애물), 무채색 픽셀=지면(평면피팅용).
#         색으로 '분할', depth 로 '3D 위치', 무채색 지면으로 'height(지면 위 높이)'.
# ═══════════════════════════════════════════════════════════════════


def _to_hwc_uint(arr):
    """obs 의 rgb/이미지 텐서를 HxWxC numpy(float)로 정규화."""
    if arr is None:
        return None
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def rgb_to_hsv_arrays(rgb01):
    """HxWx3 float[0,1] → (hue[0,360), sat[0,1], val[0,1]).  벡터화."""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    mx = rgb01.max(-1)
    mn = rgb01.min(-1)
    df = mx - mn + 1e-9
    hue = np.zeros_like(mx)
    mr, mg, mb = (mx == r), (mx == g), (mx == b)
    hue[mr] = (((g - b) / df) % 6)[mr]
    hue[mg] = (((b - r) / df) + 2)[mg]
    hue[mb] = (((r - g) / df) + 4)[mb]
    hue = (hue * 60.0) % 360.0
    sat = df / (mx + 1e-9)
    return hue, sat, mx


def _unproject_subsampled(mask_s, depth_s, intr, s, depth_min, depth_max):
    """서브샘플(stride s)된 mask/depth 의 True 픽셀을 base 프레임 3D 점으로 역투영.
    실제 픽셀좌표는 u=col*s, v=row*s 로 복원(intrinsics 는 원해상도 기준).
    반환: pts(N,3) base, us(N), vs(N)."""
    rows, cols = np.nonzero(mask_s)
    if len(rows) == 0:
        return np.empty((0, 3)), np.empty(0), np.empty(0)
    d = depth_s[rows, cols]
    good = (d > depth_min) & (d < depth_max) & np.isfinite(d)
    if not good.any():
        return np.empty((0, 3)), np.empty(0), np.empty(0)
    rows, cols, d = rows[good], cols[good], d[good].astype(np.float32)
    u = (cols * s).astype(np.float32)
    v = (rows * s).astype(np.float32)
    z = d
    x = (u - intr.cx) / intr.fx * z
    y = (v - intr.cy) / intr.fy * z
    p_cam = np.stack([x, y, z], axis=1)                       # (N,3) optical
    p_base = p_cam @ HEAD_CAM_R.T + HEAD_CAM_POS             # (N,3) base
    return p_base, u, v


def _cluster_base_xy(pts, us, vs, radius, min_pts):
    """base XY 거리 기준 그리디 군집화 → 군집별 (cx,cy,cz_median,u,v,n).
    pts(N,3), us/vs(N)."""
    clusters = []
    for i in range(len(pts)):
        x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
        placed = False
        for c in clusters:
            if math.hypot(x - c["cx"], y - c["cy"]) < radius:
                c["xs"].append(x); c["ys"].append(y); c["zs"].append(z)
                c["us"].append(float(us[i])); c["vs"].append(float(vs[i]))
                c["cx"] = sum(c["xs"]) / len(c["xs"])
                c["cy"] = sum(c["ys"]) / len(c["ys"])
                placed = True
                break
        if not placed:
            clusters.append({"xs": [x], "ys": [y], "zs": [z],
                             "us": [float(us[i])], "vs": [float(vs[i])],
                             "cx": x, "cy": y})
    out = []
    for c in clusters:
        n = len(c["xs"])
        if n < min_pts:
            continue
        out.append({
            "x": float(np.median(c["xs"])),
            "y": float(np.median(c["ys"])),
            "z": float(np.median(c["zs"])),
            "u": float(np.mean(c["us"])),
            "v": float(np.mean(c["vs"])),
            "n": n,
        })
    return out


def detect_objects_by_color(obs, intr_head=None,
                            hue_lo=42.0, hue_hi=70.0, sat_min=0.40, val_min=0.35,
                            bin_hue_lo=8.0, bin_hue_hi=38.0, bin_sat_min=0.35,
                            floor_sat_max=0.25,
                            depth_min=0.2, depth_max=8.0, grid_step=4,
                            cluster_radius=0.30, min_pixels=5):
    """색으로 물체/통을 분할하고 depth 로 3D 위치를 잡는다.

    반환: (objects, obstacles) — _scan 출력 규약과 동일.
      objects:   list[(rx, ry, dist2d, rz)]
      obstacles: list[(rx, ry, dist2d)]
    색/이미지가 없어 색 탐지가 불가하면 (None, None) → 호출부가 depth 폴백.
    """
    if intr_head is None:
        intr_head = CameraIntrinsics.head()
    image = obs.get("image") if isinstance(obs, dict) else None
    if not image:
        return None, None

    rgb = _to_hwc_uint(image.get("head_rgb"))
    depth = image.get("head_depth")
    if depth is None:
        depth = image.get("video_depth")
    depth = _to_hwc_uint(depth)
    if rgb is None or depth is None:
        return None, None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if rgb.ndim != 3 or rgb.shape[2] < 3 or depth.ndim != 2:
        return None, None
    if rgb.shape[:2] != depth.shape[:2]:
        return None, None

    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.5:               # uint8(0..255) → [0,1]
        rgb = rgb / 255.0

    # ── wall-clock 절감 ──
    #  (1) 풀해상도 대신 grid_step 으로 서브샘플 후 HSV → 연산량 s² 배 감소.
    #      물체는 화면의 ~2% 라 stride 4 에서도 군집에 충분한 점이 남는다.
    #  (2) 지면 평면피팅 제거: color 가 통/물체를 직접 가르므로 height 불필요했고,
    #      가장 큰 floor mask(전체의 ~97%) 역투영을 통째로 생략.
    s = max(1, int(grid_step))
    rgb_s = rgb[::s, ::s]
    depth_s = depth[::s, ::s]
    hue, sat, val = rgb_to_hsv_arrays(rgb_s)
    valid_d = (depth_s > depth_min) & (depth_s < depth_max) & np.isfinite(depth_s)
    yellow = (hue >= hue_lo) & (hue <= hue_hi) & (sat >= sat_min) & (val >= val_min) & valid_d
    orange = (hue >= bin_hue_lo) & (hue <= bin_hue_hi) & (sat >= bin_sat_min) & valid_d
    if not yellow.any() and not orange.any():
        return [], []

    # 물체(노랑)
    objects = []
    y_pts, y_us, y_vs = _unproject_subsampled(yellow, depth_s, intr_head, s, depth_min, depth_max)
    for c in _cluster_base_xy(y_pts, y_us, y_vs, cluster_radius, min_pixels):
        dist = math.hypot(c["x"], c["y"])
        objects.append((c["x"], c["y"], dist, c["z"]))
    objects.sort(key=lambda o: o[2])

    # 통(주황) → 장애물
    obstacles = []
    o_pts, o_us, o_vs = _unproject_subsampled(orange, depth_s, intr_head, s, depth_min, depth_max)
    for c in _cluster_base_xy(o_pts, o_us, o_vs, cluster_radius, max(min_pixels, 8)):
        dist = math.hypot(c["x"], c["y"])
        obstacles.append((c["x"], c["y"], dist))

    return objects, obstacles


# ───────────────────────── 설정 상수 ─────────────────────────
ACTION_SCALE = 0.5            # leg/arm JointPositionAction scale (env 고정값)

APPROACH_DIST = 0.46          # was 0.43 (핀 고정 데모 검증값). 배포에선 베이스가
                              # ~0.45m 에서 stall → 0.43 도달 불가. CROUCH 트리거되도록 완화.
APPROACH_TIMEOUT = 700

# ── CROUCH 전 정지·안정화 (illegal_contact 방지) ──
# 보행 모멘텀이 남은 채 앉으면 base/허벅지가 바닥을 쳐 illegal_contact(임계 1.0N)로
# 에피소드가 종료된다. 제자리 정지시켜 base 속도가 충분히 줄면 CROUCH 로 넘어간다.
SETTLE_MAX_STEPS = 120         # 안정화 타임아웃(이 안에 정지 못 해도 진행)
SETTLE_LIN_THRESH = 0.05      # base 선속도 임계 [m/s]
SETTLE_ANG_THRESH = 0.07      # base 각속도 임계 [rad/s]

# ── APPROACH 클로즈드루프 재고정(re-anchor) ──
# 접근 중 라이브 검출로 target_world 를 보정해 오도메트리 드리프트를 제거한다.
REANCHOR_GATE = 0.6   # 예측위치-검출 매칭 반경[m] (이 안의 검출만 같은 물체로 인정)
REANCHOR_EMA = 0.4    # target_world 보정 강도 (0=고정, 1=검출값 그대로). 노이즈 완화용 EMA

# ── 쓰레기통(장애물) 회피 — illegal_contact 종료 방지 ──
# illegal_contact 는 base/.*_hip/.*_thigh 에 1.0N 만 닿아도 에피소드 '전체'를 종료한다.
# 쓰레기통은 terrain.py 고정 장애물: world(7,0), 외경 반경 1.0m, 벽 높이 0.5m.
# 통 옆 물체로 접근하다 벽에 몸통/허벅지가 닿거나 통 가에서 전복하면 1.0N 으로 사망
# (15/18 에서 종료된 실측 사례). 두 겹으로 막는다:
#   (1) 오도메트리 기준 통 주변 디스크를 keep-out: 타겟/순찰지점 제외 + 진입 시 접근 중단.
#   (2) 카메라로 '키 큰 클러스터'(통 벽)를 장애물로 분류 → 타겟서 제외 + 정면 근접 시 중단.
#       (오도메트리 드리프트와 무관한 안전망. (1) 이 드리프트로 빗나가도 (2) 가 잡는다.)
# 통(=circle 드롭존) 중심(odom). terrain.py 의 trash_bin_x=7 은 "terrain 원점 기준"이고
# terrain 원점 = env origin = 로봇 스폰(world(-10,-10)) 이다. 따라서 통의 world 위치는
# (-10+7, -10+0)=(-3,-10) 으로 objects_in_circle 의 TARGET_CENTER 와 정확히 일치한다
# (통과 채점원이 같은 자리). odom(스폰 기준)으로는 (-3-(-10), -10-(-10))=(7,0).
# ── 실측 확정(log_4): 로봇 true world(-4.40,-10.25)=odom(5.6,0) 에서 target odom(6.0,0.5)
#    로 접근하다 FR_hip 이 통 벽에 165N 으로 충돌 → illegal_contact. 통 중심까지 1.43m.
#    오도메트리는 정확했다(est(5.59,-0.08) ≈ true odom(5.6,-0.25)) → 드리프트가 아니라
#    BIN 좌표 오류가 원인. 이전 값 (17,10) 은 "terrain 원점=월드원점" 오가정에서 나온
#    오류로, 순찰반경(r<=7.2) 밖 빈 공간만 지켜 통 보호가 전혀 안 됐다.
BIN_CENTER_ODOM = (7.0, 0.0)
BIN_RADIUS = 1.0                         # 통 외경 반경[m] (terrain bin_radius=1.0)
# base 중심을 이 반경 밖으로 유지[m]. 보행 정지거리(~0.2m)+hip 오프셋(~0.2m)을 감안해
# 통 벽(1.0m)에서 여유를 둔다. log_4 사망지점은 통 중심 1.12m(=keepout 안)였다.
BIN_KEEPOUT_RADIUS = BIN_RADIUS + 0.8    # = 1.8m
OBJECT_MAX_HEIGHT = 0.30                 # 이보다 높은 클러스터는 물체가 아니라 장애물(통 벽 0.5m)
OBSTACLE_ABORT_DIST = 0.7                # 정면 장애물이 이 안에 들어오면 접근 중단[m]
TILT_ABORT = 0.4                         # projected_gravity xy 크기 임계(≈24°): 기울면 전진 정지

# ── crouch+reach 타이밍 ──
# square-up: CROUCH 강하 전, 캡처한 (비대칭일 수 있는) 다리 포즈를 대칭 default 로
#   먼저 정렬한다. 대칭 강하가 되면 비대칭에서 오던 휘청이 사라져, 아래 느린 보험
#   (CROUCH_INTERP/CROUCH_SETTLE)을 줄여도 안정적 → crouch 전체 시간이 오히려 준다.
SQUARE_UP_STEPS = 50          # 캡처포즈 → 대칭 default 정렬(짧음; walk-stop≈default 라 충분)
CROUCH_INTERP = 220           # (was 380→260) square-up 대칭강하 + crouch 는 전복 원인 아니었음
CROUCH_SETTLE = 30            # (was 50) square-up 으로 강하가 깨끗 → 더 단축
CROUCH_DONE = CROUCH_INTERP + CROUCH_SETTLE
REACH_INTERP = 120            # 팔 뻗기(점수가 여기서 latch 됨) — 정확도라 유지
REACH_DONE = REACH_INTERP + 20   # (was +40) 뻗은 뒤 pad: 점수 이미 등록돼 보험만 → 단축
HOLD_STEPS = 15               # (was 40) 점수는 REACH 중 latch → 거의 순수 보험 → 대폭 단축

# 2단계 RECOVER: 팔 먼저 회수 → 그 다음 일어서기
RECOVER_ARM_STEPS = 60        # 1단계: 팔 회수(다리는 크라우치 유지)
RECOVER_LEG_INTERP = 100      # 2단계: 일어서기
RECOVER_DONE = RECOVER_ARM_STEPS + RECOVER_LEG_INTERP

# per-step 액션 rate-limit (manip 상태 한정).  관절 변화량[rad/step] 한계.
#   action 단위로는 (rad/step)/ACTION_SCALE.
MAX_LEG_DELTA_RAD = 0.02
MAX_ARM_DELTA_RAD = 0.04

SEARCH_WP_TIMEOUT = 230
SEARCH_WP_REACH = 0.7
TARGET_VISIT_RADIUS = 0.6
PATROL_V_MAX = 0.9          # 순찰 전진 속도[m/s] (was 0.6→0.45). 보행 정책 훈련 command
                             # 범위가 lin_vel_x=(-1,1) (커리큘럼 ±0.1→±1.0)이라 0.65 는
                             # 분포 한참 안쪽 — 정책이 충분히 추종. APPROACH(0.45)+SETTLE 가
                             # 앉기 전이를 분리하므로 순찰만 빨라져도 crouch 안정성엔 무영향.

GRASP_TRIGGER_DIST = 1.2       # (was 1.5) 더 가까이서 APPROACH 전환 → 느린 detour 단축,
                              # 빠른 PATROL(0.9) 구간 연장. 나선 pitch 1.2m 라 커버리지 안전.
PATROL_WP_REACH = 0.6
MAP_MERGE_RADIUS = 0.5
# 수거(done)한 물체가 발에 차여 이동해 재검출될 때, 0.5m merge 를 벗어나면 '새 타겟'으로
# 잘못 등록돼 중복 grasp 무한루프가 생긴다(실측: 16/18 후 ~9000스텝을 중복에 낭비).
# done 엔트리는 이 넓은 반경 안의 검출을 흡수해(새 타겟 X) 차인 수거물체를 따라간다.
# 진짜 미수거 물체는 수거물체에서 보통 1m+ 떨어져 있어 영향 적다.
DONE_ABSORB_RADIUS = 1.0
MAP_MIN_HITS = 1
N_OBJECTS = 18

# ── [실험] object tracking (object_map + done-absorb + blacklist 대체) ──
# 주행 중 매 스텝 스캔하므로 프레임-간 association 으로 물체 정체성을 유지한다.
# 차여 움직여도 같은 track 으로 따라가(tight gate) 수거물체는 done track 으로 남고
# (재타겟 X), gate 보다 멀리 떨어진 딴 물체는 별도 track 으로 남는다 → 위치-반경
# dedup 의 "차인 수거물체 vs 옆 미수거물체 구분불가" 딜레마가 사라진다.
TRACK_GATE = 0.40          # 프레임-간 association 반경[m] (빠른 킥도 추종, 0.4m+ 딴물체는 분리)
TRACK_POS_EMA = 0.7        # track 위치 갱신 강도(움직이는 물체 추종 우선, 약간의 평활)
TRACK_MIN_HITS = 1         # 타겟 인정 최소 hit (color 검출이 깨끗해 1 로 충분)
TRACK_PRUNE_MISS = 80      # 1-hit 노이즈 track 정리(이 스캔수 동안 재검출 없으면 삭제)
TARGET_MAX_ATTEMPTS = 2    # 득점 못한 타겟 재시도 한도(genuine miss 1회 재시도 후 포기)

# ── 이미 시도한 위치 blacklist (중복 재집기 방지) ──
# 점수는 물체별 1회 latch 라 재집기는 무의미. 그런데 집은 물체가 물리적으로 남아
# 재검출되고, odom 드리프트/넛지로 done 엔트리(merge 0.5m)에서 벗어난 새 맵 엔트리로
# 등록돼 재타겟됨 → 풀 grasp 사이클(~700스텝)을 낭비. 한 번 시도한 위치를 이 반경 안에서
# 영구 제외해 재방문을 차단한다(merge 0.5m 보다 커야 드리프트/넛지를 흡수).
# 점수 기반 차등 blacklist 반경:
#   HIT (grasp 으로 점수 상승 = 새 물체 찍음) → 작게: 방금 찍은 그 물체만 막아 클러스터
#     이웃이 살아있게 → nearest 선택이 다음 이웃을 연달아 찍는다.
#   MISS (점수 정체 = 이미 찍은 중복/도달실패) → 크게: 그 구역을 회피해 낭비 차단.
ATTEMPT_BL_RADIUS_HIT = 0.55   # (was 0.30) 차여 이동한 수거물체 재집기 방지 위해 키움.
                               # 0.55m+ 떨어진 클러스터 이웃은 여전히 분리 타겟 가능.
ATTEMPT_BL_RADIUS_MISS = 0.80

DETECT_GROUND_HEIGHT_MIN = 0.02
DETECT_GRID_STEP = 12
LOW_OBJECT_Z = 0.10

LEG_DOF, ARM_DOF, FULL_DOF = 12, 8, 20

# ════════════════ grasp 자세 (목표각, rad) ════════════════
# solution.py 와 동일한 reach 목표(그래스프 기하 보존).  다만 안정성을 위해
# 크라우치는 전후 대칭으로 둔다(비대칭이면 몸통이 기울어 COM 이 지지면을 벗어남).
# 더 얕게 하려면 calf 를 0 에 가깝게, thigh 를 작게.                    # TUNE
LEG_CROUCH_TARGET = [0.0, 1.2, -1.8, 0.0, 1.2, -1.8,
                     0.0, 1.2, -1.9, 0.0, 1.2, -1.9]
ARM_REACH_TARGET = [0.0, 2.9, -1.4, 0.0, 0.0, 0.0]

LEG_CROUCH_TARGET_LOW = [0.0, 1.2, -1.8, 0.0, 1.2, -1.8,
                         0.0, 1.2, -1.9, 0.0, 1.2, -1.9]
ARM_REACH_TARGET_LOW = [0.0, 3.0, -1.4, 0.0, 0.0, 0.0]

GRIP_OPEN = [0.04, -0.04]     # arm_joint7, arm_joint8


def _build_spiral_waypoints():
    # r0=0.4: 스폰 바로 근처(중심 구멍)에 떨어진 물체도 덮도록 안쪽에서 시작.
    # r_step=0.15: 한 바퀴(8스텝) 피치 1.2m → 카메라 스캔 밴드(±~0.6m) 안에 인접
    #   가닥이 들어와 틈이 사라진다(seed 20개 검증: gap 0). (was r0=1.0, step=0.2 →
    #   loop 1.6m + 중심 r<1.0 구멍 → 중심·틈 물체 누락.)
    import math as _m
    wps = []
    r = 0.4
    ang = 0.0
    while r <= 7.2:
        x, y = r * _m.cos(ang), r * _m.sin(ang)
        # 통 keep-out 안의 순찰지점은 건너뛴다(통으로 걸어들어가 전복 방지).
        if _m.hypot(x - BIN_CENTER_ODOM[0], y - BIN_CENTER_ODOM[1]) >= BIN_KEEPOUT_RADIUS:
            wps.append((x, y))
        ang += _m.pi / 4.0
        r += 0.15
    return wps


SPIRAL_WAYPOINTS = _build_spiral_waypoints()

S_PATROL = "PATROL"
S_APPROACH = "APPROACH"
S_SETTLE = "SETTLE"
S_CROUCH = "CROUCH"
S_REACH = "REACH"
S_HOLD = "HOLD"
S_RECOVER = "RECOVER"
S_DONE = "DONE"

# rate-limit 를 적용할 manip 상태들 (보행은 제외)
_MANIP_STATES = (S_CROUCH, S_REACH, S_HOLD, S_RECOVER)


class AlgSolution:
    def __init__(self):
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.intr_head = CameraIntrinsics.head()

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        self.leg_joint_indices = list(range(LEG_DOF))
        self.arm_joint_indices = list(range(LEG_DOF, FULL_DOF))
        self.train_to_env_action_scale = torch.tensor(
            [0.25, 0.5, 0.5] * 4, device=self.device, dtype=torch.float32).view(1, -1)
        self.env_to_train_action_scale = torch.tensor(
            [4.0, 2.0, 2.0] * 4, device=self.device, dtype=torch.float32).view(1, -1)

        DEFAULT_LEG = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5,
                       -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]
        DEFAULT_ARM = [0.0] * 8

        self._default_leg = torch.tensor(DEFAULT_LEG, device=self.device, dtype=torch.float32)
        self._default_arm = torch.tensor(DEFAULT_ARM, device=self.device, dtype=torch.float32)

        self.leg_crouch_act = self._leg_target_to_act(LEG_CROUCH_TARGET)
        self.leg_crouch_low_act = self._leg_target_to_act(LEG_CROUCH_TARGET_LOW)
        self.arm_reach_act = self._arm_target_to_act(ARM_REACH_TARGET + GRIP_OPEN)
        self.arm_reach_low_act = self._arm_target_to_act(ARM_REACH_TARGET_LOW + GRIP_OPEN)
        self.arm_home_act = self._arm_target_to_act([0.0] * 6 + GRIP_OPEN)
        self.leg_zero_act = torch.zeros((1, LEG_DOF), device=self.device, dtype=torch.float32)

        # rate-limit 한계를 action 단위로 변환
        self.max_leg_delta = MAX_LEG_DELTA_RAD / ACTION_SCALE
        self.max_arm_delta = MAX_ARM_DELTA_RAD / ACTION_SCALE

        self.reset()

    def _leg_target_to_act(self, target_list):
        t = torch.tensor(target_list, device=self.device, dtype=torch.float32)
        return ((t - self._default_leg) / ACTION_SCALE).view(1, -1)

    def _arm_target_to_act(self, target_list):
        t = torch.tensor(target_list, device=self.device, dtype=torch.float32)
        return ((t - self._default_arm) / ACTION_SCALE).view(1, -1)

    def reset(self, **kwargs):
        self.state = S_PATROL
        self.state_timer = 0
        self.target_world = None
        self.crouch_start_act = None
        self.cur_leg_crouch_act = self.leg_crouch_act
        self.cur_arm_reach_act = self.arm_reach_act
        self.prev_score = 0.0

        self.tracks: list[dict] = []          # persistent object tracks
        self._target_track = None             # 현재 추적 중인 타겟 track (dict 참조)
        self._score_events_at_commit = 0      # grasp 커밋 시점 score_events (HIT/MISS 판정)

        self.patrol_idx = 0
        self.patrol_loops = 0
        self._wp_timer = 0                 # 현재 나선 웨이포인트 체류 스텝(타임아웃용)

        self.est_x = 0.0
        self.est_y = 0.0
        self.est_yaw = 0.0
        self.dt = 0.02

        # 완료 판정용. current_score 는 grasp 순간에만 계단 상승하므로(이 task 는 연속보상
        # 없음) '상승 전이 횟수' = grasp 수 (current_score 스케일과 무관). 18 이면 완료.
        self._score_events = 0
        self._loops_at_last_event = 0     # 마지막 득점 시점의 patrol_loops (소진 판정용)

        # rate-limit 용 직전 명령(액션). manip 상태 진입 시 walk 액션에서 부드럽게 이어진다.
        self.prev_leg_cmd = None
        self.prev_arm_cmd = None

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    # ───────────────────── rate-limit ─────────────────────
    @staticmethod
    def _rate_limit(prev: torch.Tensor, target: torch.Tensor, max_delta: float) -> torch.Tensor:
        if prev is None:
            return target.clone()
        return prev + torch.clamp(target - prev, -max_delta, max_delta)

    # ───────────────────── 보행 policy ─────────────────────
    def _extract_policy_obs(self, obs, vel_cmd: torch.Tensor) -> torch.Tensor:
        p = obs["proprio"].to(self.device)
        base_ang_vel = p[:, 3:6]
        projected_gravity = p[:, 9:12]
        joint_pos_leg = p[:, 12:12 + LEG_DOF]
        joint_vel_leg = p[:, 32:32 + LEG_DOF]
        actions_env_leg = p[:, 52:52 + LEG_DOF]
        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(p.dtype)
        return torch.cat([
            base_ang_vel * 0.25,
            projected_gravity,
            vel_cmd,
            joint_pos_leg,
            joint_vel_leg * 0.05,
            actions_train_leg,
        ], dim=-1)

    def _walk_leg_action(self, obs, vel_cmd: torch.Tensor) -> torch.Tensor:
        policy_obs = self._extract_policy_obs(obs, vel_cmd)
        with torch.inference_mode():
            a_train = self.policy(policy_obs)
        if a_train.ndim == 1:
            a_train = a_train.unsqueeze(0)
        return a_train * self.train_to_env_action_scale

    def _assemble(self, leg_action: torch.Tensor, arm_action: torch.Tensor) -> torch.Tensor:
        a_env = torch.zeros((1, FULL_DOF), device=self.device, dtype=torch.float32)
        a_env[:, self.leg_joint_indices] = leg_action
        a_env[:, self.arm_joint_indices] = arm_action
        return a_env

    # ───────────────────── odom / 좌표 변환 ─────────────────────
    def _update_odometry(self, obs):
        p = obs["proprio"][0]
        vx, vy = float(p[0]), float(p[1])
        wz = float(p[5])
        self.est_x += (vx * math.cos(self.est_yaw) - vy * math.sin(self.est_yaw)) * self.dt
        self.est_y += (vx * math.sin(self.est_yaw) + vy * math.cos(self.est_yaw)) * self.dt
        self.est_yaw += wz * self.dt

    def _robot_to_world(self, dx: float, dy: float) -> tuple[float, float]:
        c, s = math.cos(self.est_yaw), math.sin(self.est_yaw)
        return (self.est_x + dx * c - dy * s, self.est_y + dx * s + dy * c)

    def _world_to_robot(self, wx: float, wy: float) -> tuple[float, float]:
        dx, dy = wx - self.est_x, wy - self.est_y
        c, s = math.cos(-self.est_yaw), math.sin(-self.est_yaw)
        return (dx * c - dy * s, dx * s + dy * c)

    # ───────────────────── 검출 ─────────────────────
    def _scan(self, obs):
        """카메라 1회 검출 → (objects, obstacles) 로 분리.

        1순위: 색(HSV) 분할 — 노랑=물체, 주황=통(장애물). 바닥 무채색이라 강건.
        폴백: 색/RGB 가 없으면 depth 지면피팅(detect_objects_from_camera) 사용,
              이때 height > OBJECT_MAX_HEIGHT(예: 0.5m 통 벽)을 장애물로 분류.
        objects: (rx, ry, dist2d, rz),  obstacles: (rx, ry, dist2d)
        """
        objects, obstacles = detect_objects_by_color(
            obs, intr_head=self.intr_head,
            grid_step=4,            # depth(12)보다 촘촘히 — 색 mask 가 희소해 비용 적음
        )
        if objects is not None:                  # 색 탐지 성공
            return objects, obstacles

        # ── depth 폴백 ──
        raw = detect_objects_from_camera(
            obs, intr_head=self.intr_head,
            ground_height_min=DETECT_GROUND_HEIGHT_MIN,
            grid_step=DETECT_GRID_STEP,
        )
        objects, obstacles = [], []
        for o in raw:
            rx, ry, rz = o["pos_robot"]
            if o["height"] > OBJECT_MAX_HEIGHT:
                obstacles.append((rx, ry, o["dist"]))
            else:
                objects.append((rx, ry, o["dist"], rz))
        return objects, obstacles

    @staticmethod
    def _nearest_ahead(items, half_angle=0.7):
        """로봇 정면(±half_angle rad) 부채꼴에서 가장 가까운 항목의 거리. 없으면 None."""
        best = None
        for it in items:
            rx, ry = it[0], it[1]
            if rx <= 0.0 or abs(math.atan2(ry, rx)) > half_angle:
                continue
            d = math.hypot(rx, ry)
            if best is None or d < best:
                best = d
        return best

    def _in_bin_keepout(self, x, y) -> bool:
        return math.hypot(x - BIN_CENTER_ODOM[0],
                          y - BIN_CENTER_ODOM[1]) < BIN_KEEPOUT_RADIUS

    # ───────────────────── 물체 맵 ─────────────────────
    def _track_update(self, world_dets):
        """world_dets: [(wx, wy, z), ...]. 프레임-간 greedy association 으로 track 갱신.
        매칭된 track 은 위치 EMA 갱신(차임 추종) + hit++; 미매칭 검출은 새 track;
        미매칭 track 은 miss++ 후 노이즈(1-hit 장기 미관측) 정리."""
        for t in self.tracks:
            t["_m"] = False
        a = TRACK_POS_EMA
        for (wx, wy, z) in world_dets:
            best, bestd = None, TRACK_GATE
            for t in self.tracks:
                if t["_m"]:
                    continue
                d = math.hypot(wx - t["x"], wy - t["y"])
                if d < bestd:
                    bestd, best = d, t
            if best is not None:
                best["x"] = (1 - a) * best["x"] + a * wx
                best["y"] = (1 - a) * best["y"] + a * wy
                best["z"] = (1 - a) * best["z"] + a * z
                best["hits"] += 1
                best["miss"] = 0
                best["_m"] = True
            else:
                self.tracks.append({"x": wx, "y": wy, "z": z, "hits": 1,
                                    "miss": 0, "done": False, "attempts": 0, "_m": True})
        survivors = []
        for t in self.tracks:
            if not t["_m"]:
                t["miss"] += 1
            # 1-hit 노이즈가 오래 안 보이면 삭제(done/다중hit track 은 영구 보존)
            if t["hits"] <= 1 and t["miss"] > TRACK_PRUNE_MISS and not t["done"]:
                continue
            survivors.append(t)
        self.tracks = survivors

    def _next_target(self):
        cands = [t for t in self.tracks
                 if t["hits"] >= TRACK_MIN_HITS and not t["done"]
                 and not self._in_bin_keepout(t["x"], t["y"])]
        if not cands:
            return None
        cx, cy = self.est_x, self.est_y
        return min(cands, key=lambda t: math.hypot(t["x"] - cx, t["y"] - cy))

    # ───────────────────── 속도 명령 ─────────────────────
    def _vel_cmd_toward(self, heading_err: float, dist: float,
                        v_max=0.45, w_max=0.8, slow_radius=1.2) -> torch.Tensor:
        wz = max(-w_max, min(w_max, 1.5 * heading_err))
        align = math.cos(heading_err)
        vx = v_max * max(0.0, align) * (min(dist, slow_radius) / slow_radius)
        return torch.tensor([[vx, 0.0, wz]], device=self.device, dtype=torch.float32)

    def _vel_cmd_to_odom_point(self, tx: float, ty: float, v_max=PATROL_V_MAX) -> torch.Tensor:
        dx = tx - self.est_x
        dy = ty - self.est_y
        dist = math.hypot(dx, dy)
        c, s = math.cos(-self.est_yaw), math.sin(-self.est_yaw)
        dx_r = dx * c - dy * s
        dy_r = dx * s + dy * c
        heading_err = math.atan2(dy_r, dx_r)
        cmd = self._vel_cmd_toward(heading_err, dist, v_max=v_max, w_max=0.6)
        cmd[0, 2] += 0.15 * math.sin(self.state_timer * 0.05)
        return cmd

    def _spiral_patrol(self) -> torch.Tensor:
        tx, ty = SPIRAL_WAYPOINTS[self.patrol_idx]
        dist = math.hypot(tx - self.est_x, ty - self.est_y)
        self._wp_timer += 1
        # 도달(0.6m) 했거나, 타임아웃 시 다음 웨이포인트로. 타임아웃이 없으면 odom 드리프트로
        # '못 가는 웨이포인트'에 영원히 갇혀 제자리 맴돌고(실측: 16/18 후 tail 내내 중앙 선회)
        # 나선이 바깥 구석으로 안 나가 마지막 물체를 못 찾고 patrol_loops 도 안 늘어 정지도 못 함.
        if dist < PATROL_WP_REACH or self._wp_timer > SEARCH_WP_TIMEOUT:
            self.patrol_idx += 1
            self._wp_timer = 0
            if self.patrol_idx >= len(SPIRAL_WAYPOINTS):
                self.patrol_idx = 0
                self.patrol_loops += 1
            tx, ty = SPIRAL_WAYPOINTS[self.patrol_idx]
        return self._vel_cmd_to_odom_point(tx, ty)

    # ───────────────────── 메인 루프 ─────────────────────
    def predicts(self, obs, current_score):
        self._update_odometry(obs)
        self.state_timer += 1
        # 점수 상승 전이 = grasp 1회 (스케일 무관). 완료/소진 판정에 사용.
        if current_score > self.prev_score + 1e-6:
            self.prev_score = current_score
            self._score_events += 1
            self._loops_at_last_event = self.patrol_loops

        stop = torch.tensor([[0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
        leg_action = self._walk_leg_action(obs, stop)
        arm_action = self.arm_home_act

        if self.state == S_PATROL:
            objs, _obstacles = self._scan(obs)
            self._track_update([(*self._robot_to_world(d[0], d[1]), d[3]) for d in objs])

            # ── 완료 → 정지(S_DONE): 18개 다 스침 (스케일 무관 카운트) ──
            if self._score_events >= N_OBJECTS:
                self._enter(S_DONE)
                return self._finalize(self._walk_leg_action(obs, stop), self.arm_home_act)

            target = self._next_target()
            # ── 소진 → 정지(S_DONE): 못 스치는 물체(통에 차여 들어감 등) 대비 ──
            # 마지막 득점 이후 전 구역 2바퀴 재탐색해도 잡을 track 이 없으면 도달불가로 보고 정지.
            # (tracking 이 차인 미수거물체를 별도 track 으로 유지하므로 blacklist 재검증 불필요.)
            if target is None and (self.patrol_loops - self._loops_at_last_event) >= 2:
                self._enter(S_DONE)
                return self._finalize(self._walk_leg_action(obs, stop), self.arm_home_act)
            if target is not None:
                d_to = math.hypot(target["x"] - self.est_x, target["y"] - self.est_y)
                if d_to < GRASP_TRIGGER_DIST:
                    self._target_track = target
                    self.target_world = (target["x"], target["y"])
                    if target["z"] < LOW_OBJECT_Z:
                        self.cur_leg_crouch_act = self.leg_crouch_low_act
                        self.cur_arm_reach_act = self.arm_reach_low_act
                    else:
                        self.cur_leg_crouch_act = self.leg_crouch_act
                        self.cur_arm_reach_act = self.arm_reach_act
                    self._enter(S_APPROACH)
                    leg_action = self._walk_leg_action(obs, stop)
                    return self._finalize(leg_action, arm_action)

            # ── 이동 (전복 안전망) ──
            # 기울면(전복 조짐) 정지해 정책이 균형을 회복하게 한다. RECOVER 직후
            # greedy 급선회로 넘어져 base 가 바닥을 쳐 illegal_contact(266N)로 죽은
            # 사례가 있어, APPROACH 와 동일한 tilt-abort 를 PATROL 에도 둔다.
            p = obs["proprio"][0]
            tilt = math.hypot(float(p[9]), float(p[10]))
            if tilt > TILT_ABORT:
                leg_action = self._walk_leg_action(obs, stop)
            elif target is not None:
                # greedy 수집: 알려진 미수거 물체로 직진(나선 대신). 도중 더 가까운
                # 물체가 검출되면 다음 스텝 _next_target 이 그쪽으로 전환.
                vel_cmd = self._vel_cmd_to_odom_point(target["x"], target["y"])
                leg_action = self._walk_leg_action(obs, vel_cmd)
            else:
                # 알려진 미수거 물체가 없을 때만 나선으로 '미지 영역' 탐색.
                vel_cmd = self._spiral_patrol()
                leg_action = self._walk_leg_action(obs, vel_cmd)

        elif self.state == S_APPROACH:
            dets, obstacles = self._scan(obs)
            self._track_update([(*self._robot_to_world(d[0], d[1]), d[3]) for d in dets])
            tt = self._target_track
            if tt is None:                      # 안전장치(타겟 유실) → 순찰 복귀
                self._enter(S_PATROL)
                leg_action = self._walk_leg_action(obs, stop)
            else:
                # target_world = 추적 track 의 최신 위치 → 매 스텝 라이브 갱신(드리프트 자동 제거)
                self.target_world = (tt["x"], tt["y"])
                rx, ry = self._world_to_robot(tt["x"], tt["y"])
                dist = math.hypot(rx, ry)
                heading = math.atan2(ry, rx)

                # ── 안전: 통(장애물) 회피로 접근 포기(illegal_contact 사망 방지) ──
                obs_ahead = self._nearest_ahead(obstacles)
                if (obs_ahead is not None and obs_ahead < OBSTACLE_ABORT_DIST) \
                        or self._in_bin_keepout(self.est_x, self.est_y) \
                        or self._in_bin_keepout(tt["x"], tt["y"]):
                    tt["done"] = True           # 통 옆 도달불가 → 이 track 포기
                    self._target_track = None
                    self.target_world = None
                    self._enter(S_PATROL)
                    leg_action = self._walk_leg_action(obs, stop)
                elif dist < APPROACH_DIST:
                    # 도달=grasp 커밋. done 판정은 RECOVER 종료 때 점수-델타로(성공=done,
                    # 실패=재시도 1회 후 done). tracking 이 정체성을 유지하니 blacklist 불필요.
                    self._score_events_at_commit = self._score_events
                    self._enter(S_SETTLE)
                    leg_action = self._walk_leg_action(obs, stop)
                elif self.state_timer > APPROACH_TIMEOUT:
                    tt["attempts"] += 1
                    if tt["attempts"] >= TARGET_MAX_ATTEMPTS:
                        tt["done"] = True
                    self._target_track = None
                    self.target_world = None
                    self._enter(S_PATROL)
                    leg_action = self._walk_leg_action(obs, stop)
                else:
                    p = obs["proprio"][0]
                    tilt = math.hypot(float(p[9]), float(p[10]))
                    if tilt > TILT_ABORT:
                        leg_action = self._walk_leg_action(obs, stop)
                    else:
                        vel_cmd = self._vel_cmd_toward(heading, dist, v_max=0.6)
                        leg_action = self._walk_leg_action(obs, vel_cmd)

        elif self.state == S_SETTLE:
            # 보행 정책으로 제자리 정지 → base 모멘텀을 죽인 뒤 CROUCH.
            # 정적 deep-crouch 는 균형기가 없어, 속도가 남은 채 앉으면 base/허벅지가
            # 바닥을 쳐 illegal_contact(1.0N) 로 종료된다.
            leg_action = self._walk_leg_action(obs, stop)
            arm_action = self.arm_home_act
            p = obs["proprio"][0]
            speed = math.hypot(float(p[0]), float(p[1]))
            wz = abs(float(p[5]))
            if (speed < SETTLE_LIN_THRESH and wz < SETTLE_ANG_THRESH) \
                    or self.state_timer >= SETTLE_MAX_STEPS:
                self._enter(S_CROUCH)

        elif self.state == S_CROUCH:
            if self.crouch_start_act is None:
                p = obs["proprio"].to(self.device)
                self.crouch_start_act = (p[:, 12:12 + LEG_DOF] / ACTION_SCALE).to(torch.float32)
            arm_action = self.arm_home_act
            if self.state_timer < SQUARE_UP_STEPS:
                # phase 1 (square-up): 캡처한 (비대칭일 수 있는) 포즈 → 대칭 default.
                # 이후 강하를 좌우대칭으로 만들어 비대칭에서 오던 휘청을 제거한다.
                a = self._ease(min(1.0, self.state_timer / SQUARE_UP_STEPS))
                leg_action = (1 - a) * self.crouch_start_act + a * self.leg_zero_act
            else:
                # phase 2 (descent): 대칭 default → crouch 타겟. 대칭 강하라 빠르게 OK.
                t = self.state_timer - SQUARE_UP_STEPS
                a = self._ease(min(1.0, t / CROUCH_INTERP))
                leg_action = (1 - a) * self.leg_zero_act + a * self.cur_leg_crouch_act
            if self.state_timer >= SQUARE_UP_STEPS + CROUCH_DONE:
                self._enter(S_REACH)

        elif self.state == S_REACH:
            leg_action = self.cur_leg_crouch_act
            a = self._ease(min(1.0, self.state_timer / REACH_INTERP))
            arm_action = (1 - a) * self.arm_home_act + a * self.cur_arm_reach_act
            if self.state_timer >= REACH_DONE:
                self._enter(S_HOLD)

        elif self.state == S_HOLD:
            leg_action = self.cur_leg_crouch_act
            arm_action = self.cur_arm_reach_act
            if self.state_timer >= HOLD_STEPS:
                self._enter(S_RECOVER)

        elif self.state == S_RECOVER:
            # 2단계: (1) 팔 먼저 회수(다리 크라우치 유지) → (2) 일어서기
            if self.state_timer < RECOVER_ARM_STEPS:
                a = self._ease(min(1.0, self.state_timer / RECOVER_ARM_STEPS))
                leg_action = self.cur_leg_crouch_act
                arm_action = (1 - a) * self.cur_arm_reach_act + a * self.arm_home_act
            else:
                t = self.state_timer - RECOVER_ARM_STEPS
                a = self._ease(min(1.0, t / RECOVER_LEG_INTERP))
                leg_action = (1 - a) * self.cur_leg_crouch_act + a * self.leg_zero_act
                arm_action = self.arm_home_act
            if self.state_timer >= RECOVER_DONE:
                # grasp 결과로 타겟 track 의 done 판정:
                #   점수↑(수거 성공) → done. tracking 이 차여 이동해도 같은 done track 으로
                #   따라가므로 phantom 재타겟이 없다. 점수 정체(miss) → 1회 재시도 후 done.
                tt = self._target_track
                if tt is not None:
                    if self._score_events > self._score_events_at_commit:
                        tt["done"] = True
                    else:
                        tt["attempts"] += 1
                        if tt["attempts"] >= TARGET_MAX_ATTEMPTS:
                            tt["done"] = True
                    self._target_track = None
                self.target_world = None
                self.crouch_start_act = None
                self._enter(S_PATROL)

        elif self.state == S_DONE:
            leg_action = self._walk_leg_action(obs, stop)
            arm_action = self.arm_home_act

        return self._finalize(leg_action, arm_action)

    # ── 부드러운 보간(ease-in-out) ──
    @staticmethod
    def _ease(a: float) -> float:
        # smoothstep: 시작/끝에서 속도 0 → 급가속 없음
        return a * a * (3.0 - 2.0 * a)

    def _finalize(self, leg_action, arm_action):
        """manip 상태에서는 per-step rate-limit 로 부드럽게, 보행은 그대로."""
        if self.state in _MANIP_STATES:
            leg_action = self._rate_limit(self.prev_leg_cmd, leg_action, self.max_leg_delta)
            arm_action = self._rate_limit(self.prev_arm_cmd, arm_action, self.max_arm_delta)
        # 다음 스텝 rate-limit 기준 갱신(보행 액션도 저장 → manip 진입 시 매끄럽게 연결)
        self.prev_leg_cmd = leg_action.detach().clone()
        self.prev_arm_cmd = arm_action.detach().clone()
        action_env = self._assemble(leg_action, arm_action)
        return {"action": action_env.cpu().numpy().tolist(), "giveup": False}

    def _enter(self, new_state):
        self.state = new_state
        self.state_timer = 0
        if new_state == S_CROUCH:
            self.crouch_start_act = None
