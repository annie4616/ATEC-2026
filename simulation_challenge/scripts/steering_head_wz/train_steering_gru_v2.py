"""RECURRENT steering head V2: proven CLS(384)+LiDAR front-end (the MLP that hit
sign-acc 0.935 / 3-of-5 completion) + a GRU temporal layer on top.

Why V2 (vs train_steering_gru.py):
  The first GRU attempt replaced the DINOv2 CLS token with a hand-rolled PatchPool
  (row-mean pooling + col-conv). That THREW AWAY the well-optimized CLS summary and
  offline R2 collapsed 0.935 -> 0.514, and closed-loop wz oscillated (0/5 completed).
  V2 isolates the one question we actually want answered — "does a temporal belief
  state help pre-empt the slope/stairs drift?" — by keeping the exact proven per-frame
  front-end (rgb_mlp + lidar_mlp, late fusion) and only adding recurrence:

      frame_feat_t = [rgb_mlp(CLS_t), lidar_mlp(lidar_t)]   (proven, 0.935)
      out_t, h_t   = GRU(frame_feat_t, h_{t-1})             (NEW: belief over drift)
      wz_t         = head(out_t)

  No past action is fed in (avoids causal confusion in recurrent BC, De Haan 2019).

Data: CONTINUOUS episodes only (spawn at x~-141, single episode). Teleport-collected
  mid-course clips are EXCLUDED — they inject fake context jumps that poison a recurrent
  belief state (they were fine for the memoryless MLP, harmful here).

  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/hdd0/sohyun \
    python scripts/train_steering_gru_v2.py --data A.npz B.npz \
      --out .../steering_head_gru_v2 --cache .../gru_v2_cls_cache.npz
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn


# ----------------------------- data -----------------------------
def load_episodes(paths, reset_dx=-20.0):
    """Split each npz into contiguous episodes at x-resets. For continuous files this is
    a single episode; the split is a safety net against any stray teleport."""
    eps = []
    for p in paths:
        d = np.load(p)
        rgb = d["rgb"]
        lidar = d["lidar"].reshape(len(d["wz"]), -1).astype(np.float32)
        wz = d["wz"].astype(np.float32)
        x = d["xy"][:, 0]
        cuts = [0] + list(np.where(np.diff(x) < reset_dx)[0] + 1) + [len(x)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a >= 4:
                eps.append({"rgb": rgb[a:b], "lidar": lidar[a:b], "wz": wz[a:b]})
    return eps


@torch.inference_mode()
def dinov2_cls_features(rgb, device, bs=32):
    """rgb (N,H,W,3) uint8 -> (N,384) frozen DINOv2 CLS features (same as the MLP head)."""
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    feats = []
    for i in range(0, len(rgb), bs):
        inp = proc(images=list(rgb[i:i + bs]), return_tensors="pt").to(device)
        feats.append(model(**inp).last_hidden_state[:, 0].cpu().numpy().astype(np.float32))
    return np.concatenate(feats, 0)


# ----------------------------- model -----------------------------
class GRUSteerV2(nn.Module):
    """Proven two-branch late-fusion front-end + GRU temporal layer."""
    def __init__(self, rgb_dim=384, lidar_dim=5760, hidden=256,
                 gru_hidden=256, gru_layers=1):
        super().__init__()
        self.rgb_mlp = nn.Sequential(
            nn.Linear(rgb_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.lidar_mlp = nn.Sequential(
            nn.Linear(lidar_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.gru = nn.GRU(hidden * 2, gru_hidden, gru_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, rgb_seq, lidar_seq, h0=None):
        # rgb_seq (B,T,384), lidar_seq (B,T,5760)
        r = self.rgb_mlp(rgb_seq)                # (B,T,hidden)
        l = self.lidar_mlp(lidar_seq)            # (B,T,hidden)
        feat = torch.cat([r, l], -1)             # (B,T,2*hidden)
        out, hn = self.gru(feat, h0)             # (B,T,gru_hidden)
        wz = self.head(out).squeeze(-1)          # (B,T)
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
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=str, default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    print("[data] loading episodes ...", flush=True)
    eps = load_episodes(args.data)
    print(f"[data] {len(eps)} episodes, {sum(len(e['wz']) for e in eps)} frames", flush=True)

    if args.cache and os.path.exists(args.cache):
        print(f"[feat] loading cached CLS feats {args.cache}", flush=True)
        cache = np.load(args.cache, allow_pickle=True)
        cls_list = list(cache["cls"]); lidar_list = list(cache["lidar"]); wz_list = list(cache["wz"])
    else:
        print("[feat] extracting frozen DINOv2 CLS features (one-time) ...", flush=True)
        cls_list, lidar_list, wz_list = [], [], []
        for i, e in enumerate(eps):
            cls_list.append(dinov2_cls_features(e["rgb"], dev))
            lidar_list.append(e["lidar"])
            wz_list.append(e["wz"])
            if i % 5 == 0:
                print(f"  ep {i+1}/{len(eps)}", flush=True)
        if args.cache:
            np.savez(args.cache,
                     cls=np.array(cls_list, dtype=object),
                     lidar=np.array(lidar_list, dtype=object),
                     wz=np.array(wz_list, dtype=object))
            print(f"[feat] cached -> {args.cache}", flush=True)

    # normalization (over all frames)
    all_cls = np.concatenate(cls_list, 0)
    rf_mu, rf_sd = all_cls.mean(0), all_cls.std(0) + 1e-6
    all_lidar = np.concatenate(lidar_list, 0)
    li_mu, li_sd = all_lidar.mean(0), all_lidar.std(0) + 1e-6
    del all_cls, all_lidar

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

    ldim = lidar_list[0].shape[1]

    def batch(wins):
        L = args.seq_len
        R = np.zeros((len(wins), L, 384), np.float32)
        Ld = np.zeros((len(wins), L, ldim), np.float32)
        Y = np.zeros((len(wins), L), np.float32)
        M = np.zeros((len(wins), L), np.float32)
        for k, (ei, s, e) in enumerate(wins):
            n = e - s
            R[k, :n] = (cls_list[ei][s:e] - rf_mu) / rf_sd
            Ld[k, :n] = (lidar_list[ei][s:e] - li_mu) / li_sd
            Y[k, :n] = wz_list[ei][s:e]
            M[k, :n] = 1.0
        return (torch.tensor(R, device=dev), torch.tensor(Ld, device=dev),
                torch.tensor(Y, device=dev), torch.tensor(M, device=dev))

    model = GRUSteerV2(lidar_dim=ldim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    huber = nn.SmoothL1Loss(reduction="none")

    best = {"r2": -1e9}
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(tr_w)
        for i in range(0, len(tr_w), args.bs):
            wins = tr_w[i:i + args.bs]
            R, Ld, Y, M = batch(wins)
            opt.zero_grad()
            pred, _ = model(R, Ld)
            m = M.clone(); m[:, :args.warmup] = 0.0
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
                    R, Ld, Y, M = batch(wins)
                    pred, _ = model(R, Ld)
                    mm = M.bool(); mm[:, :args.warmup] = False
                    preds.append(pred[mm].cpu().numpy()); ys.append(Y[mm].cpu().numpy())
            pv, yv = np.concatenate(preds), np.concatenate(ys)
            r2 = r2_np(pv, yv); sign = (np.sign(pv) == np.sign(yv)).mean()
            if r2 > best["r2"]:
                best = {"r2": float(r2), "sign": float(sign), "ep": ep}
                torch.save({
                    "model": model.state_dict(),
                    "rf_mu": rf_mu, "rf_sd": rf_sd, "li_mu": li_mu, "li_sd": li_sd,
                    "lidar_dim": ldim, "arch": "gru_cls_lidarmlp",
                }, os.path.join(args.out, "steering_head.pt"))
            print(f"[ep {ep:3d}] val R2={r2:+.3f} sign={sign:.3f} (best {best['r2']:+.3f}@{best['ep']})", flush=True)

    print(f"\n[BEST] val R2={best['r2']:+.3f} sign-acc={best['sign']:.3f} @ep{best['ep']}", flush=True)
    print(f"  saved -> {os.path.join(args.out, 'steering_head.pt')}", flush=True)


if __name__ == "__main__":
    main()
