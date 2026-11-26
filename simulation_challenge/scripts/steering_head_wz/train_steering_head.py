"""STEERING IL: train a wz head on frozen DINOv2 (RGB) + raw LiDAR features.

The kyaw/lane-keeping teacher produced (sensor, wz) pairs; the linear probe already
showed the steering cue is present (LiDAR sign-acc 0.83, RGB 0.76). Here we train a
proper NONLINEAR head (frozen self-supervised backbone -> MLP -> wz) which should
beat the linear probe and give a deployable steering policy that reads the red course
boundary from the camera / lidar instead of privileged world yaw.

Backbone is FROZEN (DINOv2-small) for sim2real robustness — only the small head learns,
so the appearance gap of the real outdoor course is absorbed by the self-supervised
features. LiDAR (geometry, most sim2real-robust) goes in raw through its own MLP.

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/hdd0/sohyun \
    python scripts/train_steering_head.py \
      --data /hdd0/sohyun/atec_2026_tesk_e/taska_redstrip_ytrack/steering_data.npz \
             /hdd0/sohyun/atec_2026_tesk_e/taska_redstrip_jitter/steering_data.npz \
      --out /hdd0/sohyun/atec_2026_tesk_e/steering_head
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn


def load_data(paths):
    rgb, lidar, wz = [], [], []
    for p in paths:
        d = np.load(p)
        rgb.append(d["rgb"])
        lidar.append(d["lidar"].reshape(len(d["wz"]), -1))
        wz.append(d["wz"].astype(np.float32))
    return (np.concatenate(rgb, 0),
            np.concatenate(lidar, 0),
            np.concatenate(wz, 0))


@torch.inference_mode()
def dinov2_features(rgb, device, bs=32):
    """rgb (N,H,W,3) uint8 -> (N,384) frozen DINOv2 CLS features (cached backbone)."""
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    feats = []
    for i in range(0, len(rgb), bs):
        inp = proc(images=list(rgb[i:i + bs]), return_tensors="pt").to(device)
        feats.append(model(**inp).last_hidden_state[:, 0].cpu().numpy())
    return np.concatenate(feats, 0)


class SteeringHead(nn.Module):
    """frozen RGB feat (384) + raw lidar (L) -> wz. Two branches, late fusion."""
    def __init__(self, rgb_dim, lidar_dim, hidden=256):
        super().__init__()
        self.rgb_mlp = nn.Sequential(
            nn.Linear(rgb_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.lidar_mlp = nn.Sequential(
            nn.Linear(lidar_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, rgb_feat, lidar):
        r = self.rgb_mlp(rgb_feat)
        l = self.lidar_mlp(lidar)
        return self.fuse(torch.cat([r, l], -1)).squeeze(-1)


def r2_np(pred, y):
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-9
    return 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    rgb, lidar, wz = load_data(args.data)
    print(f"[data] {len(wz)} frames  wz std={wz.std():.3f} range[{wz.min():+.2f},{wz.max():+.2f}]", flush=True)

    print("[feat] extracting frozen DINOv2 features ...", flush=True)
    rgb_feat = dinov2_features(rgb, dev)          # (N,384)
    print(f"[feat] rgb_feat {rgb_feat.shape}, lidar {lidar.shape}", flush=True)

    # standardize inputs (store stats for deployment)
    rf_mu, rf_sd = rgb_feat.mean(0), rgb_feat.std(0) + 1e-6
    li_mu, li_sd = lidar.mean(0), lidar.std(0) + 1e-6
    rgb_feat = (rgb_feat - rf_mu) / rf_sd
    lidar = (lidar - li_mu) / li_sd

    # random split (cue-present regime; temporal split extrapolates terrain, misleading)
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(wz))
    nval = int(len(wz) * args.val_frac)
    va, tr = idx[:nval], idx[nval:]

    def to_t(a):
        return torch.tensor(a, dtype=torch.float32, device=dev)
    Rtr, Ltr, Ytr = to_t(rgb_feat[tr]), to_t(lidar[tr]), to_t(wz[tr])
    Rva, Lva, Yva = to_t(rgb_feat[va]), to_t(lidar[va]), to_t(wz[va])

    model = SteeringHead(rgb_feat.shape[1], lidar.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.SmoothL1Loss()

    best = {"r2": -1e9}
    bs = 256
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            pred = model(Rtr[b], Ltr[b])
            loss = lossf(pred, Ytr[b])
            loss.backward()
            opt.step()
        sched.step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.inference_mode():
                pv = model(Rva, Lva).cpu().numpy()
            yv = Yva.cpu().numpy()
            r2 = r2_np(pv, yv)
            sign = (np.sign(pv) == np.sign(yv)).mean()
            if r2 > best["r2"]:
                best = {"r2": float(r2), "sign": float(sign), "ep": ep}
                torch.save({
                    "model": model.state_dict(),
                    "rf_mu": rf_mu, "rf_sd": rf_sd, "li_mu": li_mu, "li_sd": li_sd,
                    "rgb_dim": rgb_feat.shape[1], "lidar_dim": lidar.shape[1],
                }, os.path.join(args.out, "steering_head.pt"))
            print(f"[ep {ep:3d}] val R2={r2:+.3f} sign={sign:.3f}  (best R2={best['r2']:+.3f}@{best['ep']})", flush=True)

    print(f"\n[BEST] val R2={best['r2']:+.3f} sign-acc={best['sign']:.3f} @ep{best['ep']}", flush=True)
    print(f"  linear-probe baseline was: LiDAR R2=+0.496/0.832, RGB R2=+0.338/0.756", flush=True)
    print(f"  saved -> {os.path.join(args.out, 'steering_head.pt')}", flush=True)


if __name__ == "__main__":
    main()
