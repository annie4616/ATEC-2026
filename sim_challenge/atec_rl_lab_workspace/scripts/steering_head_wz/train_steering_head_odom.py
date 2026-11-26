"""STEERING IL + ODOMETRY: the proven 2-branch head (frozen DINOv2 CLS + LiDAR) PLUS a
third branch fed a dead-reckoned pose estimate (x, y, heading) integrated from eval-legal
proprioception with periodic reset.

Motivation: the camera/LiDAR head is reactive and cannot see LATERAL DRIFT (y) building up
on slope/stairs -- that is exactly where the robot leaves the course (|y|->12, falls off).
Dead-reckoning (integrate body velocity + yaw rate, reset every N steps) gives a cheap,
eval-legal, short-horizon estimate of "how far I've drifted sideways / turned since the last
reset", which we feed to the head so it can correct drift it cannot see in a single frame.
The periodic reset keeps it order-invariant (no absolute course position memorized).

  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/hdd0/sohyun \
    python scripts/train_steering_head_odom.py --data A.npz B.npz \
      --out .../steering_head_odom
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn


def load_data(paths):
    rgb, lidar, wz, odom = [], [], [], []
    for p in paths:
        d = np.load(p)
        if "odom" not in d.files:
            raise SystemExit(f"{p} has no 'odom' key -- re-collect with the odom-logging "
                             f"dagger_rollout_taska.py before training the odom head.")
        n = len(d["wz"])
        rgb.append(d["rgb"])
        lidar.append(d["lidar"].reshape(n, -1))
        wz.append(d["wz"].astype(np.float32))
        odom.append(d["odom"].astype(np.float32))          # (n,3) = (x,y,heading)
    return (np.concatenate(rgb, 0), np.concatenate(lidar, 0),
            np.concatenate(wz, 0), np.concatenate(odom, 0))


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


class SteeringHeadOdom(nn.Module):
    """frozen RGB CLS (384) + raw lidar (L) + dead-reckoned odom (3) -> wz. Three branches,
    late fusion. Same RGB/LiDAR branches as the proven head; odom gets a small branch."""
    def __init__(self, rgb_dim, lidar_dim, odom_dim=3, hidden=256):
        super().__init__()
        self.rgb_mlp = nn.Sequential(
            nn.Linear(rgb_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU())
        self.lidar_mlp = nn.Sequential(
            nn.Linear(lidar_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU())
        self.odom_mlp = nn.Sequential(
            nn.Linear(odom_dim, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2 + 64, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1))

    def forward(self, rgb_feat, lidar, odom):
        r = self.rgb_mlp(rgb_feat)
        l = self.lidar_mlp(lidar)
        o = self.odom_mlp(odom)
        return self.fuse(torch.cat([r, l, o], -1)).squeeze(-1)


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

    rgb, lidar, wz, odom = load_data(args.data)
    print(f"[data] {len(wz)} frames  wz std={wz.std():.3f}  odom dim={odom.shape[1]} "
          f"per-col range={[f'[{odom[:,i].min():+.2f},{odom[:,i].max():+.2f}]' for i in range(odom.shape[1])]}",
          flush=True)

    print("[feat] extracting frozen DINOv2 features ...", flush=True)
    rgb_feat = dinov2_features(rgb, dev)

    rf_mu, rf_sd = rgb_feat.mean(0), rgb_feat.std(0) + 1e-6
    li_mu, li_sd = lidar.mean(0), lidar.std(0) + 1e-6
    od_mu, od_sd = odom.mean(0), odom.std(0) + 1e-6
    rgb_feat = (rgb_feat - rf_mu) / rf_sd
    lidar = (lidar - li_mu) / li_sd
    odom = (odom - od_mu) / od_sd

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(wz))
    nval = int(len(wz) * args.val_frac)
    va, tr = idx[:nval], idx[nval:]

    def to_t(a): return torch.tensor(a, dtype=torch.float32, device=dev)
    Rtr, Ltr, Otr, Ytr = to_t(rgb_feat[tr]), to_t(lidar[tr]), to_t(odom[tr]), to_t(wz[tr])
    Rva, Lva, Ova, Yva = to_t(rgb_feat[va]), to_t(lidar[va]), to_t(odom[va]), to_t(wz[va])

    odom_dim = odom.shape[1]
    model = SteeringHeadOdom(rgb_feat.shape[1], lidar.shape[1], odom_dim=odom_dim).to(dev)
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
            pred = model(Rtr[b], Ltr[b], Otr[b])
            loss = lossf(pred, Ytr[b])
            loss.backward()
            opt.step()
        sched.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.inference_mode():
                pv = model(Rva, Lva, Ova).cpu().numpy()
            yv = Yva.cpu().numpy()
            r2 = r2_np(pv, yv); sign = (np.sign(pv) == np.sign(yv)).mean()
            if r2 > best["r2"]:
                best = {"r2": float(r2), "sign": float(sign), "ep": ep}
                torch.save({
                    "model": model.state_dict(),
                    "rf_mu": rf_mu, "rf_sd": rf_sd, "li_mu": li_mu, "li_sd": li_sd,
                    "od_mu": od_mu, "od_sd": od_sd,
                    "rgb_dim": rgb_feat.shape[1], "lidar_dim": lidar.shape[1], "odom_dim": odom_dim,
                    "arch": "steering_head_odom",
                }, os.path.join(args.out, "steering_head.pt"))
            print(f"[ep {ep:3d}] val R2={r2:+.3f} sign={sign:.3f} (best {best['r2']:+.3f}@{best['ep']})", flush=True)

    print(f"\n[BEST] val R2={best['r2']:+.3f} sign-acc={best['sign']:.3f} @ep{best['ep']}", flush=True)
    print(f"  saved -> {os.path.join(args.out, 'steering_head.pt')}", flush=True)


if __name__ == "__main__":
    main()
