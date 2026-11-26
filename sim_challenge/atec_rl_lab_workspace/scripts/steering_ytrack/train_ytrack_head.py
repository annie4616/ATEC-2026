"""Train a CROSS-TRACK y regressor: sensors -> absolute lateral position y.

Motivation: the camera sees the red boundary, so absolute y is directly observable EVERY
frame with NO integration drift (unlike odom). We regress y, then a PD lane-keeping loop
(wz = -k*y_hat - kd*yaw_rate) restores the ORIGINAL privileged-y teacher that completed the
course -- but driven by the ESTIMATED y instead of the privileged world y.

Two deployable variants (prepared side by side; pick at real-world time):
  --mode rgb   : DINOv2 CLS (camera) -> y     (precise ~7cm in-sim, but appearance-dependent)
  --mode lidar : raw LiDAR (geometry) -> y    (coarser ~23cm, but appearance-INVARIANT = sim2real)
  --mode both  : RGB + LiDAR -> y

  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/hdd0/sohyun \
    python scripts/train_ytrack_head.py --mode rgb   --data *.npz --out .../ytrack_rgb
    python scripts/train_ytrack_head.py --mode lidar --data *.npz --out .../ytrack_lidar
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn


def load_data(paths):
    rgb, lidar, y = [], [], []
    for p in paths:
        d = np.load(p)
        n = len(d["wz"])
        rgb.append(d["rgb"])
        lidar.append(d["lidar"].reshape(n, -1))
        y.append(d["xy"][:, 1].astype(np.float32))   # absolute cross-track y = regression target
    return np.concatenate(rgb, 0), np.concatenate(lidar, 0), np.concatenate(y, 0)


@torch.inference_mode()
def dinov2_features(rgb, device, bs=32):
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    feats = []
    for i in range(0, len(rgb), bs):
        inp = proc(images=list(rgb[i:i + bs]), return_tensors="pt").to(device)
        feats.append(model(**inp).last_hidden_state[:, 0].cpu().numpy())
    return np.concatenate(feats, 0)


class YHead(nn.Module):
    """sensor feature -> scalar cross-track y. Branch-per-sensor when mode=both."""
    def __init__(self, rgb_dim, lidar_dim, mode, hidden=256):
        super().__init__()
        self.mode = mode
        if mode in ("rgb", "both"):
            self.rgb_mlp = nn.Sequential(nn.Linear(rgb_dim, hidden), nn.GELU(), nn.Dropout(0.2),
                                         nn.Linear(hidden, hidden), nn.GELU())
        if mode in ("lidar", "both"):
            self.lidar_mlp = nn.Sequential(nn.Linear(lidar_dim, hidden), nn.GELU(), nn.Dropout(0.2),
                                           nn.Linear(hidden, hidden), nn.GELU())
        fuse_in = hidden * (2 if mode == "both" else 1)
        self.head = nn.Sequential(nn.Linear(fuse_in, hidden), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, 1))

    def forward(self, rgb_feat, lidar):
        parts = []
        if self.mode in ("rgb", "both"):
            parts.append(self.rgb_mlp(rgb_feat))
        if self.mode in ("lidar", "both"):
            parts.append(self.lidar_mlp(lidar))
        return self.head(torch.cat(parts, -1)).squeeze(-1)


def r2_np(p, t):
    return 1 - ((t - p) ** 2).sum() / (((t - t.mean()) ** 2).sum() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["rgb", "lidar", "both"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    rgb, lidar, y = load_data(args.data)
    print(f"[data] {len(y)} frames  y std={y.std():.3f} range[{y.min():+.2f},{y.max():+.2f}]  mode={args.mode}", flush=True)

    if args.mode in ("rgb", "both"):
        print("[feat] extracting DINOv2 ...", flush=True)
        rgbf = dinov2_features(rgb, dev)
    else:
        rgbf = np.zeros((len(y), 1), np.float32)   # unused placeholder
    rf_mu, rf_sd = rgbf.mean(0), rgbf.std(0) + 1e-6
    li_mu, li_sd = lidar.mean(0), lidar.std(0) + 1e-6
    rgbf = (rgbf - rf_mu) / rf_sd
    lidar = (lidar - li_mu) / li_sd

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(y)); nv = int(len(y) * args.val_frac)
    va, tr = idx[:nv], idx[nv:]

    def t(a): return torch.tensor(a, dtype=torch.float32, device=dev)
    Rtr, Ltr, Ytr = t(rgbf[tr]), t(lidar[tr]), t(y[tr])
    Rva, Lva, Yva = t(rgbf[va]), t(lidar[va]), t(y[va])

    model = YHead(rgbf.shape[1], lidar.shape[1], args.mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.SmoothL1Loss()

    best = {"r2": -1e9}
    bs = 256
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            loss = lossf(model(Rtr[b], Ltr[b]), Ytr[b]); loss.backward(); opt.step()
        sched.step()
        if ep % 20 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.inference_mode():
                pv = model(Rva, Lva).cpu().numpy()
            yv = Yva.cpu().numpy(); rr = r2_np(pv, yv); mae = np.abs(pv - yv).mean()
            if rr > best["r2"]:
                best = {"r2": float(rr), "mae": float(mae), "ep": ep}
                torch.save({"model": model.state_dict(), "mode": args.mode,
                            "rf_mu": rf_mu, "rf_sd": rf_sd, "li_mu": li_mu, "li_sd": li_sd,
                            "rgb_dim": rgbf.shape[1], "lidar_dim": lidar.shape[1],
                            "arch": "ytrack_head"}, os.path.join(args.out, "ytrack_head.pt"))
            print(f"[ep {ep:3d}] val R2={rr:+.3f} MAE={mae:.3f}m (best R2={best['r2']:+.3f}@{best['ep']})", flush=True)

    print(f"\n[BEST] R2={best['r2']:+.3f} MAE={best['mae']:.3f}m @ep{best['ep']}  mode={args.mode}", flush=True)
    print(f"  saved -> {os.path.join(args.out, 'ytrack_head.pt')}", flush=True)


if __name__ == "__main__":
    main()
