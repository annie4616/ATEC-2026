"""Demo-level quality filter for Task E trajectories.

The input HDF5 already holds only *successful* demos (collected with
--only_success). This script applies a STRICTER, demo-level filter to keep only
the smoothest / most stable trajectories for fine-tuning.

Because object end-positions are NOT stored in the HDF5, quality is judged from
the trajectory itself:

  - qvel_max  : max |joint velocity| over the demo. High values = jerky / unstable
                arm motion. This is the most discriminative signal in the data.
  - ee_jerk_max : max per-step end-effector displacement (sudden EE jumps).
  - tail_std  : std of EE position over the last `tail` steps (place jitter).
  - length    : total steps (very long demos = hesitation / re-tries).

A demo is KEPT only if every enabled metric is below its threshold. Pass
--dry_run first to see the distribution and how many demos survive.

Usage:
    # inspect distribution + survival count, no write
    python scripts/act/filter_demos_quality.py \
        --input  /hdd0/sohyun/atec_2026_tesk_e/task_e/trajectory_filtered.hdf5 \
        --dry_run

    # write the strict subset
    python scripts/act/filter_demos_quality.py \
        --input  /hdd0/sohyun/atec_2026_tesk_e/task_e/trajectory_filtered.hdf5 \
        --output /hdd0/sohyun/atec_2026_tesk_e/task_e/trajectory_strict.hdf5 \
        --qvel_max 2.75 --tail 30
"""

import argparse
import h5py
import numpy as np


def demo_metrics(grp, tail):
    """Compute per-demo quality metrics from a trajectory group."""
    act = grp["actions"][:]
    ee = grp["ee_pos"][:]
    qvel = grp["qvel"][:]
    T = len(act)

    qvel_max = float(np.abs(qvel).max())
    ee_step = np.linalg.norm(np.diff(ee, axis=0), axis=1)
    ee_jerk_max = float(ee_step.max()) if len(ee_step) else 0.0
    tail_n = min(tail, T)
    tail_std = float(ee[-tail_n:].std(axis=0).max())
    return {
        "length": T,
        "qvel_max": qvel_max,
        "ee_jerk_max": ee_jerk_max,
        "tail_std": tail_std,
    }


def keep_demo(m, args):
    """Return (keep_bool, reason) for one demo's metrics."""
    if args.qvel_max is not None and m["qvel_max"] > args.qvel_max:
        return False, f"qvel_max {m['qvel_max']:.2f}>{args.qvel_max}"
    if args.ee_jerk_max is not None and m["ee_jerk_max"] > args.ee_jerk_max:
        return False, f"ee_jerk {m['ee_jerk_max']:.4f}>{args.ee_jerk_max}"
    if args.tail_std is not None and m["tail_std"] > args.tail_std:
        return False, f"tail_std {m['tail_std']:.4f}>{args.tail_std}"
    if args.max_length is not None and m["length"] > args.max_length:
        return False, f"length {m['length']}>{args.max_length}"
    return True, "ok"


def main():
    p = argparse.ArgumentParser(description="Strict demo-level quality filter.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--tail", type=int, default=30,
                   help="Trailing steps used for place-jitter (tail_std).")
    p.add_argument("--qvel_max", type=float, default=2.75,
                   help="Reject demos whose max|joint vel| exceeds this (None=off).")
    p.add_argument("--ee_jerk_max", type=float, default=None,
                   help="Reject demos whose max EE step exceeds this (None=off).")
    p.add_argument("--tail_std", type=float, default=None,
                   help="Reject demos whose place jitter exceeds this (None=off).")
    p.add_argument("--max_length", type=int, default=None,
                   help="Reject demos longer than this many steps (None=off).")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--no_compress", action="store_true",
                   help="Skip gzip compression for a much faster write.")
    args = p.parse_args()
    comp = None if args.no_compress else "gzip"

    with h5py.File(args.input, "r") as fin:
        keys = sorted(fin.keys(), key=lambda k: int(k.split("_")[1]))
        metrics = {k: demo_metrics(fin[k], args.tail) for k in keys}

        # distribution summary
        for name in ("length", "qvel_max", "ee_jerk_max", "tail_std"):
            v = np.array([metrics[k][name] for k in keys])
            print(f"{name:12s} min={v.min():.4f} p25={np.percentile(v,25):.4f} "
                  f"med={np.median(v):.4f} p75={np.percentile(v,75):.4f} "
                  f"p90={np.percentile(v,90):.4f} max={v.max():.4f}")

        kept, dropped = [], []
        for k in keys:
            ok, reason = keep_demo(metrics[k], args)
            (kept if ok else dropped).append((k, reason))

        print(f"\nKEEP {len(kept)}/{len(keys)}  (DROP {len(dropped)})")
        if dropped:
            print("dropped:")
            for k, r in dropped:
                print(f"  {k}: {r}")

        if args.dry_run or args.output is None:
            print("\n(dry run — no output written)" if args.dry_run
                  else "\n(no --output given — nothing written)")
            return

        with h5py.File(args.output, "w") as fout:
            for new_i, (k, _) in enumerate(kept):
                grp_in = fin[k]
                grp_out = fout.create_group(f"traj_{new_i}")
                for ds in grp_in.keys():
                    if ds == "images":
                        ig = grp_out.create_group("images")
                        for ik in grp_in["images"].keys():
                            ig.create_dataset(ik, data=grp_in[f"images/{ik}"][:],
                                              compression=comp)
                    elif isinstance(grp_in[ds], h5py.Dataset):
                        ig = grp_in[ds][:]
                        grp_out.create_dataset(ds, data=ig, compression=comp)
        print(f"\nSaved {len(kept)} demos → {args.output}")


if __name__ == "__main__":
    main()
