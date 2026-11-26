"""SHORT-HORIZON steering head: proven CLS(384)+LiDAR front-end + a K-frame STACK.

Why a frame stack instead of a GRU (the pivot):
  A full-episode GRU carries a belief state that CAN memorize absolute course position
  ("I've walked N steps, stairs come now"). On this single fixed course that shortcut
  helps offline but makes the policy ORDER-DEPENDENT — if the real course reorders the
  terrain, the memorized clock is wrong and it fails. (Empirically the GRU also didn't
  even help: episode-split R2 stayed ~0.37.)

  The only temporal signal we actually need is the DRIFT ONSET (dy/dt) — a SHORT window
  (~0.2 s). A K-frame stack gives exactly that: the head sees the last K frames of
  [CLS, lidar] and can read the motion trend, but has NO capacity to store where it is
  in the course. It is order-invariant / reactive, so it is robust to real-world terrain
  reordering — which is the stated top priority.

  Bonus: with no long belief state, teleport-collected mid-course clips are NO LONGER
  harmful, so ALL data (incl. the extra stairs/slope coverage) is usable again. Stacks
  are built WITHIN an episode only (never across a teleport cut).

  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/hdd0/sohyun \
    python scripts/train_steering_stack.py --data *.npz \
      --out .../steering_head_stack --cache .../stack_cls_cache_all.npz --K 4
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn


def load_episodes(paths, reset_dx=-20.0):
    eps = []
    for p in paths:
        try:
            d = np.load(p)
        except Exception as e:
            print(f"[skip] {p}: {e}", flush=True); continue
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
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    feats = []
    for i in range(0, len(rgb), bs):
        inp = proc(images=list(rgb[i:i + bs]), return_tensors="pt").to(device)
        feats.append(model(**inp).last_hidden_state[:, 0].cpu().numpy().astype(np.float32))
    return np.concatenate(feats, 0)


class FrameStackHead(nn.Module):
    """Per-frame proven front-end (shared across the K frames) then a small temporal
    encoder over the K axis -> wz. K is small (~4), so there is no course-position memory."""
    def __init__(self, K=4, rgb_dim=384, lidar_dim=5760, hidden=256):
        super().__init__()
        self.K = K
        self.rgb_mlp = nn.Sequential(
            nn.Linear(rgb_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.lidar_mlp = nn.Sequential(
            nn.Linear(lidar_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        # temporal encoder over the K frames of the fused (2*hidden) feature.
        # 1D conv on the time axis reads the motion trend (dy/dt) locally.
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, rgb_stack, lidar_stack):
        # rgb_stack (B,K,384), lidar_stack (B,K,5760)
        r = self.rgb_mlp(rgb_stack)                 # (B,K,hidden)
        l = self.lidar_mlp(lidar_stack)             # (B,K,hidden)
        f = torch.cat([r, l], -1)                   # (B,K,2*hidden)
        t = self.temporal(f.transpose(1, 2))        # (B,hidden,K)
        t = t[:, :, -1]                             # take the newest frame's summary
        return self.head(t).squeeze(-1)             # (B,)


def r2_np(pred, y):
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-9
    return 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--K", type=int, default=4, help="frames in the stack (short: no course memory)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=str, default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    K = args.K

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
            lidar_list.append(e["lidar"]); wz_list.append(e["wz"])
            if i % 10 == 0:
                print(f"  ep {i+1}/{len(eps)}", flush=True)
        if args.cache:
            np.savez(args.cache,
                     cls=np.array(cls_list, dtype=object),
                     lidar=np.array(lidar_list, dtype=object),
                     wz=np.array(wz_list, dtype=object))
            print(f"[feat] cached -> {args.cache}", flush=True)

    all_cls = np.concatenate(cls_list, 0)
    rf_mu, rf_sd = all_cls.mean(0), all_cls.std(0) + 1e-6
    all_lidar = np.concatenate(lidar_list, 0)
    li_mu, li_sd = all_lidar.mean(0), all_lidar.std(0) + 1e-6
    ldim = lidar_list[0].shape[1]
    del all_cls, all_lidar

    # build stacks WITHIN each episode: sample t uses frames [t-K+1 .. t] (clamped at start)
    rng = np.random.RandomState(args.seed)
    ep_idx = rng.permutation(len(eps))
    nval = max(1, int(len(eps) * args.val_frac))
    val_eps = set(ep_idx[:nval].tolist())

    def build(ep_set):
        R, L, Y = [], [], []
        for ei in ep_set:
            cls = (cls_list[ei] - rf_mu) / rf_sd
            lid = (lidar_list[ei] - li_mu) / li_sd
            wz = wz_list[ei]
            T = len(wz)
            for t in range(T):
                lo = t - K + 1
                idx = [max(0, j) for j in range(lo, t + 1)]   # clamp-pad at episode start
                R.append(cls[idx]); L.append(lid[idx]); Y.append(wz[t])
        return (np.asarray(R, np.float32), np.asarray(L, np.float32), np.asarray(Y, np.float32))

    tr_set = [i for i in range(len(eps)) if i not in val_eps]
    Rtr, Ltr, Ytr = build(tr_set)
    Rva, Lva, Yva = build(list(val_eps))
    print(f"[data] train stacks {len(Ytr)}, val stacks {len(Yva)}  (K={K})", flush=True)

    def to_t(a): return torch.tensor(a, device=dev)
    Rtr_t, Ltr_t, Ytr_t = to_t(Rtr), to_t(Ltr), to_t(Ytr)
    Rva_t, Lva_t, Yva_t = to_t(Rva), to_t(Lva), to_t(Yva)

    model = FrameStackHead(K=K, lidar_dim=ldim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.SmoothL1Loss()

    best = {"r2": -1e9}
    N = len(Ytr)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, args.bs):
            b = perm[i:i + args.bs]
            opt.zero_grad()
            pred = model(Rtr_t[b], Ltr_t[b])
            loss = lossf(pred, Ytr_t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.inference_mode():
                pv = model(Rva_t, Lva_t).cpu().numpy()
            yv = Yva
            r2 = r2_np(pv, yv); sign = (np.sign(pv) == np.sign(yv)).mean()
            if r2 > best["r2"]:
                best = {"r2": float(r2), "sign": float(sign), "ep": ep}
                torch.save({
                    "model": model.state_dict(),
                    "rf_mu": rf_mu, "rf_sd": rf_sd, "li_mu": li_mu, "li_sd": li_sd,
                    "lidar_dim": ldim, "K": K, "arch": "framestack_cls_lidarmlp",
                }, os.path.join(args.out, "steering_head.pt"))
            print(f"[ep {ep:3d}] val R2={r2:+.3f} sign={sign:.3f} (best {best['r2']:+.3f}@{best['ep']})", flush=True)

    print(f"\n[BEST] val R2={best['r2']:+.3f} sign-acc={best['sign']:.3f} @ep{best['ep']}", flush=True)
    print(f"  (episode-held-out split; K={K} frame stack, order-invariant)", flush=True)
    print(f"  saved -> {os.path.join(args.out, 'steering_head.pt')}", flush=True)


if __name__ == "__main__":
    main()
