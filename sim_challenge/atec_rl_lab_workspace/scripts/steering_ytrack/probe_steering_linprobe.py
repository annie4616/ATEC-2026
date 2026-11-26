"""FALLACY-1 QUANTIFICATION: can a pretrained encoder's features predict the teacher wz?

Takes the steering_data.npz from probe_taska_steering_cues.py and, for each student
sensor (head RGB via DINOv2, head depth via DeFM, lidar height-scan raw), fits a
simple ridge linear probe features -> teacher_wz with a train/test split. A high test
R^2 means the steering cue the kyaw teacher uses IS present in that sensor, so an IL
student could learn to steer from it without world yaw. Low R^2 across all sensors
would mean the cue is NOT observable and IL is doomed before we collect anything.

Run WITHOUT Isaac (plain python), needs torch + the pretrained encoders (cached):
  PYTHONPATH=/hdd0/sohyun python scripts/probe_steering_linprobe.py \
      --data /hdd0/sohyun/atec_2026_tesk_e/taska_steering_probe/steering_data.npz
"""
import argparse
import numpy as np
import torch


def ridge_r2(X, y, alpha=1.0, split=0.75, pca_dim=64):
    """Fit ridge on a train split, report test R^2 (and train for reference).

    Optionally reduce features to `pca_dim` PCA components FIRST (fit on train only)
    to curb overfitting when feature dim >> #samples — otherwise a high-dim encoder
    memorizes the train wz and test R^2 is meaningless. Also split by TIME (not random)
    so temporally-adjacent near-duplicate frames don't leak train->test."""
    n = len(y)
    ntr = int(n * split)
    tr, te = np.arange(ntr), np.arange(ntr, n)   # temporal split (no leakage)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    # PCA (fit on train) to keep the probe honest when d >> n
    if pca_dim and Xtr.shape[1] > pca_dim:
        mu0 = Xtr.mean(0)
        U, S, Vt = np.linalg.svd(Xtr - mu0, full_matrices=False)
        comp = Vt[:pca_dim]
        Xtr = (Xtr - mu0) @ comp.T
        Xte = (Xte - mu0) @ comp.T
    # standardize
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    # closed-form ridge
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d)
    w = np.linalg.solve(A, Xtr.T @ ytr)
    b = ytr.mean() - (Xtr.mean(0) @ w)
    def r2(Xs, ys):
        pred = Xs @ w + b
        ss_res = ((ys - pred) ** 2).sum()
        ss_tot = ((ys - ys.mean()) ** 2).sum() + 1e-9
        return 1 - ss_res / ss_tot
    return r2(Xte, yte), r2(Xtr, ytr)


def dinov2_features(rgb, device):
    """rgb: (N,H,W,3) uint8 -> (N,D) DINOv2 CLS features."""
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    feats = []
    with torch.inference_mode():
        for i in range(0, len(rgb), 32):
            batch = rgb[i:i + 32]
            inp = proc(images=list(batch), return_tensors="pt").to(device)
            out = model(**inp)
            feats.append(out.last_hidden_state[:, 0].cpu().numpy())  # CLS
    return np.concatenate(feats, 0)


def defm_features(depth, device, weight="defm_efficientnet_b0.pth"):
    """depth: (N,H,W) float -> (N,D) DeFM features. Downloads weight if needed."""
    from huggingface_hub import hf_hub_download
    import torchvision.models as tvm
    # DeFM uses a 3-channel log-compressed depth representation; approximate by
    # log-normalizing and stacking to 3 channels for an ImageNet-shaped backbone.
    wpath = hf_hub_download("leggedrobotics/defm", weight)
    sd = torch.load(wpath, map_location="cpu")
    body = tvm.efficientnet_b0(weights=None)
    # try to load matching keys (best-effort; DeFM backbone == torchvision efficientnet)
    try:
        state = sd.get("model", sd.get("state_dict", sd))
        state = {k.replace("backbone.", "").replace("encoder.", ""): v for k, v in state.items()}
        missing, unexpected = body.load_state_dict(state, strict=False)
        print(f"  DeFM load: {len(missing)} missing, {len(unexpected)} unexpected keys")
    except Exception as e:
        print(f"  DeFM load warn: {str(e)[:120]}")
    body.classifier = torch.nn.Identity()
    body = body.to(device).eval()
    def prep(d):
        d = np.asarray(d, dtype=np.float32)
        d = np.where(np.isfinite(d), d, 0.0)
        d = np.log1p(np.clip(d, 0, 100))
        d = (d - d.mean()) / (d.std() + 1e-6)
        return d
    feats = []
    with torch.inference_mode():
        for i in range(0, len(depth), 32):
            batch = np.stack([prep(x) for x in depth[i:i + 32]])  # (b,H,W)
            t = torch.from_numpy(batch).unsqueeze(1).repeat(1, 3, 1, 1)  # (b,3,H,W)
            t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            feats.append(body(t.to(device)).cpu().numpy())
    return np.concatenate(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    d = np.load(args.data)
    wz = d["wz"].astype(np.float32)
    print(f"[data] {len(wz)} frames, wz range [{wz.min():+.2f},{wz.max():+.2f}] std={wz.std():.3f}")
    print(f"  rgb {d['rgb'].shape} depth {d['depth'].shape} lidar {d['lidar'].shape}\n")

    results = {}

    # baseline: predict mean (R^2=0 by construction) — sanity
    print("=== Linear probe: sensor features -> teacher wz (test R^2) ===")

    # lidar raw (flatten) — cheapest, geometry, sim2real robust
    if d["lidar"].size > 0:
        Xl = d["lidar"].reshape(len(wz), -1)
        te, tr = ridge_r2(Xl, wz, alpha=10.0)
        results["lidar_raw"] = te
        print(f"  lidar height-scan (raw)   : test R^2={te:+.3f}  (train {tr:+.3f})")

    # depth via DeFM
    try:
        Xd = defm_features(d["depth"], args.device)
        te, tr = ridge_r2(Xd, wz, alpha=10.0)
        results["depth_defm"] = te
        print(f"  head depth (DeFM)         : test R^2={te:+.3f}  (train {tr:+.3f})")
    except Exception as e:
        print(f"  head depth (DeFM)         : SKIP ({str(e)[:100]})")

    # rgb via DINOv2
    try:
        Xr = dinov2_features(d["rgb"], args.device)
        te, tr = ridge_r2(Xr, wz, alpha=10.0)
        results["rgb_dinov2"] = te
        print(f"  head RGB (DINOv2)         : test R^2={te:+.3f}  (train {tr:+.3f})")
    except Exception as e:
        print(f"  head RGB (DINOv2)         : SKIP ({str(e)[:100]})")

    print("\n=== VERDICT (fallacy-1) ===")
    best = max(results.values()) if results else -1
    if best > 0.5:
        print(f"  PASS: steering cue IS present (best test R^2={best:+.3f}). IL viable.")
    elif best > 0.2:
        print(f"  WEAK: partial cue (best R^2={best:+.3f}). IL possible but may need temporal context / better encoder.")
    else:
        print(f"  FAIL: cue NOT observable (best R^2={best:+.3f}). Sensors lack heading info as-is.")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v:+.3f}")


if __name__ == "__main__":
    main()
