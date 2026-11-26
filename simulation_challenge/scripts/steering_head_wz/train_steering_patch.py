"""STEERING IL (patch-token ablation): train a wz head on frozen DINOv2 RGB + raw LiDAR.

Motivation: the deployed PD8 head uses only the DINOv2 CLS token (1x384), a GLOBAL
image summary that BLURS where the red boundary strip sits left/right. Steering is a
left/right spatial decision, so we test PATCH tokens (16x16x384) with the left/right
(column) axis preserved via a 1D conv -- the head then sees "red is in the left columns
-> steer right" directly. This mirrors the CLS-vs-patch ablation in the roadmap.

Two RGB front-ends selectable via --rgb_mode:
  cls   : last_hidden_state[:,0]           (1,384)   -- reproduces PD8 exactly
  patch : last_hidden_state[:,1:] 16x16    -> row-mean -> (16col,384) -> col 1D-conv -> D

Color/photometric augmentation (--color_aug): the red strip is a COLOR cue, and sim has
a single fixed light. To avoid overfitting to "this exact red under this exact light",
we jitter brightness/contrast/saturation/hue BEFORE the frozen DINOv2, so the head learns
"a red-ish boundary somewhere on the left" rather than a specific pixel value -> better
sim2real light transfer. Because augmented features can't be cached once, we pre-extract
--aug_mult augmented copies per frame (plus the clean one) and train on the union.

  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/hdd0/sohyun \
    python scripts/steering_head_wz/train_steering_patch.py \
      --data <PD8 npz list> --rgb_mode patch --color_aug --aug_mult 2 \
      --out .../steering_head_patch --cache .../patch_cache_patch_aug.npz
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


def _color_jitter(pix, gen):
    """pix: (B,3,H,W) float in DINOv2-normalized space is hard; jitter is applied to the
    RAW uint8->float[0,1] image BEFORE the processor normalization. Called on raw images."""
    # not used directly; jitter is applied via torchvision transforms on PIL/uint8 below.
    raise NotImplementedError


@torch.inference_mode()
def dinov2_features(rgb, device, rgb_mode, color_aug=False, aug_mult=0, seed=0, bs=32):
    """rgb (N,H,W,3) uint8 -> features.
      rgb_mode cls   -> (M,384)
      rgb_mode patch -> (M,16,16,384)   (row,col,dim)  [caller pools]
    If color_aug: emits the clean copy PLUS aug_mult jittered copies (M=N*(1+aug_mult)),
    and returns (feats, wz_index) so labels can be tiled. Otherwise M=N.
    """
    from transformers import AutoModel, AutoImageProcessor
    import torchvision.transforms as T
    from PIL import Image
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

    # photometric jitter on the raw image (before DINOv2 normalization). hue is the key
    # axis for a colored boundary strip; keep it modest so red stays red-ish.
    jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)
    g = torch.Generator().manual_seed(seed)

    n_copies = 1 + (aug_mult if color_aug else 0)
    feats_all, idx_all = [], []
    for copy in range(n_copies):
        do_aug = color_aug and copy > 0
        for i in range(0, len(rgb), bs):
            batch = rgb[i:i + bs]
            imgs = [Image.fromarray(im) for im in batch]
            if do_aug:
                # deterministic-ish per copy via the shared generator
                torch.manual_seed(seed * 1000 + copy * 97 + i)
                imgs = [jitter(im) for im in imgs]
            inp = proc(images=imgs, return_tensors="pt").to(device)
            out = model(**inp).last_hidden_state          # (b,257,384)
            if rgb_mode == "cls":
                f = out[:, 0].cpu().numpy()               # (b,384)
            else:
                b = out.shape[0]
                p = out[:, 1:].reshape(b, 16, 16, 384)    # (b,row,col,384)
                # ROW-MEAN here (fixed op, not learned) so the cache is (N,16col,384)
                # ~1.3GB instead of (N,16,16,384) ~500GB. Vertical info is intentionally
                # pooled; steering needs the horizontal (column) axis.
                f = p.mean(dim=1).cpu().numpy()           # (b,16col,384)
            feats_all.append(f)
            idx_all.append(np.arange(i, i + len(batch)))
    feats = np.concatenate(feats_all, 0)
    idx = np.concatenate(idx_all, 0)
    return feats, idx


class SteeringHeadCLS(nn.Module):
    def __init__(self, rgb_dim, lidar_dim, hidden=256):
        super().__init__()
        self.rgb_mlp = nn.Sequential(
            nn.Linear(rgb_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU())
        self.lidar_mlp = nn.Sequential(
            nn.Linear(lidar_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1))

    def forward(self, rgb_feat, lidar):
        r = self.rgb_mlp(rgb_feat)
        l = self.lidar_mlp(lidar)
        return self.fuse(torch.cat([r, l], -1)).squeeze(-1)


class SteeringHeadPatch(nn.Module):
    """RGB patch grid (16row,16col,384): average over rows (vertical), then a 1D conv
    along the COLUMN (left/right) axis so the head reads WHERE the boundary sits
    horizontally. Late-fuse with the raw-lidar branch, same as CLS head."""
    def __init__(self, patch_dim, lidar_dim, ncol=16, hidden=256, rgb_out=256):
        super().__init__()
        self.patch_dim = patch_dim
        # per-column channel reduce 384 -> 64, then conv along columns
        self.col_proj = nn.Sequential(nn.Linear(patch_dim, 64), nn.GELU())
        self.col_conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.GELU(),   # keep 16 cols
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.GELU())
        self.rgb_head = nn.Sequential(
            nn.Linear(128 * ncol, rgb_out), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(rgb_out, hidden), nn.GELU())
        self.lidar_mlp = nn.Sequential(
            nn.Linear(lidar_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1))

    def forward(self, col, lidar):
        # col: (B,16col,384) -- rows already pooled at extraction time
        col = self.col_proj(col)                     # (B,16,64)
        col = col.transpose(1, 2)                    # (B,64,16)
        col = self.col_conv(col)                     # (B,128,16)
        r = self.rgb_head(col.flatten(1))            # (B,hidden)
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
    ap.add_argument("--rgb_mode", choices=["cls", "patch"], default="patch")
    ap.add_argument("--color_aug", action="store_true")
    ap.add_argument("--aug_mult", type=int, default=2, help="extra jittered copies/frame")
    ap.add_argument("--cache", default="", help="npz cache for extracted features")
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
    print(f"[data] {len(wz)} frames  wz std={wz.std():.3f} "
          f"range[{wz.min():+.2f},{wz.max():+.2f}]  rgb_mode={args.rgb_mode} "
          f"color_aug={args.color_aug} aug_mult={args.aug_mult}", flush=True)

    # --- extract (or load cached) RGB features ---
    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        rgb_feat, feat_idx = z["feat"], z["idx"]
        print(f"[feat] loaded cache {args.cache} feat={rgb_feat.shape}", flush=True)
    else:
        print("[feat] extracting frozen DINOv2 features ...", flush=True)
        rgb_feat, feat_idx = dinov2_features(
            rgb, dev, args.rgb_mode, color_aug=args.color_aug,
            aug_mult=args.aug_mult, seed=args.seed)
        if args.cache:
            np.savez(args.cache, feat=rgb_feat, idx=feat_idx)
            print(f"[feat] cached -> {args.cache}", flush=True)
    print(f"[feat] rgb_feat {rgb_feat.shape}  (idx {feat_idx.shape})", flush=True)

    # tile lidar/wz by feat_idx (handles augmented copies)
    lidar = lidar[feat_idx]
    wz = wz[feat_idx]

    # standardize inputs (store stats for deployment)
    if args.rgb_mode == "cls":
        rf_mu, rf_sd = rgb_feat.mean(0), rgb_feat.std(0) + 1e-6
    else:
        # normalize per-dim over (row,col) pooled stats: compute over all patches
        flat = rgb_feat.reshape(-1, rgb_feat.shape[-1])
        rf_mu, rf_sd = flat.mean(0), flat.std(0) + 1e-6
    rgb_feat = (rgb_feat - rf_mu) / rf_sd
    li_mu, li_sd = lidar.mean(0), lidar.std(0) + 1e-6
    lidar = (lidar - li_mu) / li_sd

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(wz))
    nval = int(len(wz) * args.val_frac)
    va, tr = idx[:nval], idx[nval:]

    def to_t(a):
        return torch.tensor(a, dtype=torch.float32, device=dev)
    Rtr, Ltr, Ytr = to_t(rgb_feat[tr]), to_t(lidar[tr]), to_t(wz[tr])
    Rva, Lva, Yva = to_t(rgb_feat[va]), to_t(lidar[va]), to_t(wz[va])

    if args.rgb_mode == "cls":
        model = SteeringHeadCLS(rgb_feat.shape[-1], lidar.shape[1]).to(dev)
    else:
        model = SteeringHeadPatch(rgb_feat.shape[-1], lidar.shape[1]).to(dev)
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
                    "rgb_mode": args.rgb_mode,
                    "rf_mu": rf_mu, "rf_sd": rf_sd, "li_mu": li_mu, "li_sd": li_sd,
                    "rgb_dim": rgb_feat.shape[-1], "lidar_dim": lidar.shape[1],
                }, os.path.join(args.out, "steering_head.pt"))
            print(f"[ep {ep:3d}] val R2={r2:+.3f} sign={sign:.3f}  "
                  f"(best R2={best['r2']:+.3f}@{best['ep']})", flush=True)

    print(f"\n[BEST] rgb_mode={args.rgb_mode} color_aug={args.color_aug} "
          f"val R2={best['r2']:+.3f} sign-acc={best['sign']:.3f} @ep{best['ep']}", flush=True)
    print(f"  saved -> {os.path.join(args.out, 'steering_head.pt')}", flush=True)


if __name__ == "__main__":
    main()
