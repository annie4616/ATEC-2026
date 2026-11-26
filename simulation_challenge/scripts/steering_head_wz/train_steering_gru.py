"""RECURRENT steering head: frozen DINOv2 PATCH tokens + LiDAR 1D-conv -> GRU -> wz.

Motivation (from the failure analysis + literature survey):
  - The single-frame MLP head (train_steering_head.py) is REACTIVE: it sees only the
    current frame, so it detects a large lateral offset y only AFTER it has grown, and by
    then the slope/stairs have already pushed the robot out of the course (y blows up to
    +-12 -> falls off). It cannot see dy/dt (the onset of being pushed).
  - Fixes, each grounded in a paper:
      * GRU recurrent head  -> temporal belief state estimates the DRIFT TREND, steering
        pre-emptively before y grows (ANYmal Miki 2022 GRU belief; Extreme Parkour /
        Robot Parkour 2023 GRU student).
      * DINOv2 PATCH tokens (not CLS) -> steering is a "which side is the boundary" spatial
        question; the 384-d CLS pools away left/right structure (Patch Policy 2607.18236).
      * LiDAR 1D-conv over the 360 angular axis (16 channels), circular padding -> keeps
        angular locality / direction of approaching terrain (vs a flat 5760 MLP).

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/hdd0/sohyun \
    python scripts/train_steering_gru.py --data A.npz B.npz --out .../steering_head_gru
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn


# ----------------------------- data -----------------------------
def load_episodes(paths, reset_dx=-20.0):
    """Load npz files, split each into contiguous episodes at x-resets (teleport/respawn
    make x jump backwards). Returns list of dicts per episode with rgb/lidar/wz arrays."""
    eps = []
    for p in paths:
        d = np.load(p)
        rgb = d["rgb"]                                   # (N,H,W,3) uint8
        lidar = d["lidar"].reshape(len(d["wz"]), -1)     # (N,5760)
        wz = d["wz"].astype(np.float32)
        x = d["xy"][:, 0]
        # episode boundaries: index where x jumps back by > |reset_dx|
        cuts = [0] + list(np.where(np.diff(x) < reset_dx)[0] + 1) + [len(x)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a >= 4:                               # need a few frames for a sequence
                eps.append({"rgb": rgb[a:b], "lidar": lidar[a:b], "wz": wz[a:b]})
    return eps


@torch.inference_mode()
def dinov2_patch_features(rgb, device, bs=16):
    """rgb (N,H,W,3) uint8 -> (N, 256, 384) frozen DINOv2 PATCH tokens (CLS dropped)."""
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    feats = []
    for i in range(0, len(rgb), bs):
        inp = proc(images=list(rgb[i:i + bs]), return_tensors="pt").to(device)
        # last_hidden_state (B, 1+256, 384); [:,1:] = patch tokens
        feats.append(model(**inp).last_hidden_state[:, 1:].cpu().numpy().astype(np.float32))
    return np.concatenate(feats, 0)                       # (N,256,384)


# ----------------------------- model -----------------------------
class PatchPool(nn.Module):
    """256 patch tokens (16x16 grid) x 384 -> a compact vector that PRESERVES horizontal
    (left/right) structure. We pool over rows (vertical) but keep columns (horizontal) via
    a 1D conv along the 16-wide column axis, then flatten. This keeps 'boundary is on the
    left vs right' information a steering head needs."""
    def __init__(self, dim=384, out=256, grid=16):
        super().__init__()
        self.grid = grid
        self.proj = nn.Linear(dim, 64)
        # after row-mean: (B,64,16 cols) -> conv along columns
        self.col_conv = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
        )
        self.out = nn.Linear(64 * grid, out)

    def forward(self, patches):                            # (B,256,384)
        B = patches.shape[0]
        h = self.proj(patches)                             # (B,256,64)
        g = h.view(B, self.grid, self.grid, 64)            # (B, rows, cols, 64)
        g = g.mean(dim=1)                                  # pool rows -> (B, cols, 64)
        g = g.transpose(1, 2)                              # (B,64,cols)
        g = self.col_conv(g)                               # (B,64,cols)
        return self.out(g.reshape(B, -1))                  # (B,out)


class LidarConv(nn.Module):
    """LiDAR 5760 = 16 channels x 360 angular bins -> 1D conv along the 360 axis with
    circular padding (angle is periodic). Keeps direction-of-terrain structure."""
    def __init__(self, ch=16, bins=360, out=128):
        super().__init__()
        self.ch, self.bins = ch, bins
        self.net = nn.Sequential(
            nn.Conv1d(ch, 32, 5, stride=3, padding=2, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1, padding_mode="circular"), nn.GELU(),
        )
        self.out = nn.LazyLinear(out)

    def forward(self, lidar):                              # (B,5760)
        B = lidar.shape[0]
        x = lidar.view(B, self.ch, self.bins)              # (B,16,360)
        x = self.net(x)
        return self.out(x.reshape(B, -1))                  # (B,out)


class GRUSteer(nn.Module):
    """per-frame feat = [PatchPool(rgb_patches), LidarConv(lidar)] -> GRU -> wz.
    NOTE: no past action fed in (avoids causal confusion in recurrent BC)."""
    def __init__(self, rgb_out=256, lidar_out=128, gru_hidden=256, gru_layers=1):
        super().__init__()
        self.patch = PatchPool(out=rgb_out)
        self.lidar = LidarConv(out=lidar_out)
        self.gru = nn.GRU(rgb_out + lidar_out, gru_hidden, gru_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(gru_hidden, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, patches_seq, lidar_seq, h0=None):
        # patches_seq (B,T,256,384), lidar_seq (B,T,5760)
        B, T = patches_seq.shape[:2]
        p = self.patch(patches_seq.reshape(B * T, *patches_seq.shape[2:])).view(B, T, -1)
        l = self.lidar(lidar_seq.reshape(B * T, -1)).view(B, T, -1)
        feat = torch.cat([p, l], -1)                       # (B,T,rgb_out+lidar_out)
        out, hn = self.gru(feat, h0)                       # (B,T,H)
        wz = self.head(out).squeeze(-1)                    # (B,T)
        return wz, hn


def r2_np(pred, y):
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-9
    return 1 - ss_res / ss_tot


# ----------------------------- train -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq_len", type=int, default=32, help="subsequence length for BPTT")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--warmup", type=int, default=8, help="warmup steps not counted in loss (h0=0 bias)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=str, default="", help="npz cache of patch feats (skip DINOv2 if exists)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    print("[data] loading episodes ...", flush=True)
    eps = load_episodes(args.data)
    print(f"[data] {len(eps)} episodes, total frames {sum(len(e['wz']) for e in eps)}", flush=True)

    # extract patch features per episode (cache to disk since DINOv2 is the slow part)
    if args.cache and os.path.exists(args.cache):
        print(f"[feat] loading cached patch feats {args.cache}", flush=True)
        cache = np.load(args.cache, allow_pickle=True)
        patch_list = list(cache["patch"]); lidar_list = list(cache["lidar"]); wz_list = list(cache["wz"])
    else:
        print("[feat] extracting frozen DINOv2 patch tokens (slow, one-time) ...", flush=True)
        patch_list, lidar_list, wz_list = [], [], []
        for i, e in enumerate(eps):
            patch_list.append(dinov2_patch_features(e["rgb"], dev))   # (Ti,256,384)
            lidar_list.append(e["lidar"].astype(np.float32))
            wz_list.append(e["wz"])
            if i % 5 == 0:
                print(f"  ep {i+1}/{len(eps)}", flush=True)
        if args.cache:
            np.savez(args.cache,
                     patch=np.array(patch_list, dtype=object),
                     lidar=np.array(lidar_list, dtype=object),
                     wz=np.array(wz_list, dtype=object))
            print(f"[feat] cached -> {args.cache}", flush=True)

    # normalization stats (over all frames)
    all_lidar = np.concatenate(lidar_list, 0)
    li_mu, li_sd = all_lidar.mean(0), all_lidar.std(0) + 1e-6
    # patch feats: normalize per-channel over the 384 dim (flatten patches+frames)
    all_patch = np.concatenate([p.reshape(-1, 384) for p in patch_list], 0)
    pf_mu, pf_sd = all_patch.mean(0), all_patch.std(0) + 1e-6
    del all_lidar, all_patch

    # build (episode, start) sliding windows
    rng = np.random.RandomState(args.seed)
    ep_idx = rng.permutation(len(eps))
    nval = max(1, int(len(eps) * args.val_frac))
    val_eps, tr_eps = set(ep_idx[:nval].tolist()), set(ep_idx[nval:].tolist())

    def windows(ep_set):
        w = []
        for ei in ep_set:
            T = len(wz_list[ei])
            if T <= args.seq_len:
                w.append((ei, 0, T))
            else:
                for s in range(0, T - args.seq_len + 1, args.seq_len // 2):
                    w.append((ei, s, s + args.seq_len))
        return w
    tr_w, va_w = windows(tr_eps), windows(val_eps)
    print(f"[data] train windows {len(tr_w)}, val windows {len(va_w)}", flush=True)

    def batch(wins):
        L = args.seq_len
        P = np.zeros((len(wins), L, 256, 384), np.float32)
        Ld = np.zeros((len(wins), L, lidar_list[0].shape[1]), np.float32)
        Y = np.zeros((len(wins), L), np.float32)
        M = np.zeros((len(wins), L), np.float32)
        for k, (ei, s, e) in enumerate(wins):
            n = e - s
            P[k, :n] = (patch_list[ei][s:e] - pf_mu) / pf_sd
            Ld[k, :n] = (lidar_list[ei][s:e] - li_mu) / li_sd
            Y[k, :n] = wz_list[ei][s:e]
            M[k, :n] = 1.0
        return (torch.tensor(P, device=dev), torch.tensor(Ld, device=dev),
                torch.tensor(Y, device=dev), torch.tensor(M, device=dev))

    model = GRUSteer().to(dev)
    # init LazyLinear
    with torch.no_grad():
        _ = model(torch.zeros(1, 2, 256, 384, device=dev), torch.zeros(1, 2, lidar_list[0].shape[1], device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    huber = nn.SmoothL1Loss(reduction="none")

    best = {"r2": -1e9}
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(tr_w)
        for i in range(0, len(tr_w), args.bs):
            wins = tr_w[i:i + args.bs]
            P, Ld, Y, M = batch(wins)
            opt.zero_grad()
            pred, _ = model(P, Ld)
            m = M.clone()
            m[:, :args.warmup] = 0.0                        # ignore warmup (h0=0 bias)
            loss = (huber(pred, Y) * m).sum() / (m.sum() + 1e-6)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if ep % 5 == 0 or ep == args.epochs - 1:
            model.eval()
            preds, ys = [], []
            with torch.inference_mode():
                for i in range(0, len(va_w), args.bs):
                    wins = va_w[i:i + args.bs]
                    P, Ld, Y, M = batch(wins)
                    pred, _ = model(P, Ld)
                    mm = M.bool()
                    mm[:, :args.warmup] = False
                    preds.append(pred[mm].cpu().numpy()); ys.append(Y[mm].cpu().numpy())
            pv, yv = np.concatenate(preds), np.concatenate(ys)
            r2 = r2_np(pv, yv); sign = (np.sign(pv) == np.sign(yv)).mean()
            if r2 > best["r2"]:
                best = {"r2": float(r2), "sign": float(sign), "ep": ep}
                torch.save({
                    "model": model.state_dict(),
                    "pf_mu": pf_mu, "pf_sd": pf_sd, "li_mu": li_mu, "li_sd": li_sd,
                    "arch": "gru_patch_lidarconv",
                }, os.path.join(args.out, "steering_head.pt"))
            print(f"[ep {ep:3d}] val R2={r2:+.3f} sign={sign:.3f} (best {best['r2']:+.3f}@{best['ep']})", flush=True)

    print(f"\n[BEST] val R2={best['r2']:+.3f} sign-acc={best['sign']:.3f} @ep{best['ep']}", flush=True)
    print(f"  saved -> {os.path.join(args.out, 'steering_head.pt')}", flush=True)


if __name__ == "__main__":
    main()
