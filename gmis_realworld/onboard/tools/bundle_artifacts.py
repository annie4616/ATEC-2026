#!/usr/bin/env python3
"""Copy the five policy artifacts into onboard/artifacts/ and write a sha256 manifest.

WHY THIS EXISTS
---------------
The grasp champion that produced the 120/160 honest run lives at
`exp_runs/runs/tailB3_2026-09-07_09-59-57/model_1999.pt`, and `exp_runs/runs/` is in
`.gitignore`.  So the single most valuable weight file in the project is **not in the
repository**, which means it is not in the code escrow and it is not on the robot.
Everything else is scattered across `from_eunha/`, `eunha_rl/s075/...`, `s0821/banks_ik/`
with paths that only exist on this server.

This turns those five paths into one self-contained directory with content hashes, so the
mini-PC gets a bundle instead of a tree, and so a mismatch between "what we escrowed" and
"what ran" is detectable instead of silent.

Run:  python onboard/tools/bundle_artifacts.py [--config onboard/configs/policies.yaml]
Check: python onboard/tools/bundle_artifacts.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ONBOARD = Path(__file__).resolve().parents[1]
REPO = ONBOARD.parent
ARTIFACTS = ONBOARD / "artifacts"

# role -> (repo-relative source, destination name).  A source may be a directory.
SOURCES: dict[str, tuple[str, str]] = {
    "walk":       ("from_eunha/0824/model_22998.pt",                                  "walk_actor.pt"),
    "adaptation": ("from_eunha/0824/adaptation.pt",                                   "adaptation.pt"),
    "crouch":     ("eunha_rl/s075/crouch_runs/crouch0.24_2026-08-11_01-35-23/model_2999.pt", "crouch_actor.pt"),
    "grasp":      ("exp_runs/runs/tailB3_2026-09-07_09-59-57/model_1999.pt",          "grasp_actor.pt"),
    "bank":       ("eunha_rl/s075/banks/pregrasp_bank_0.24.npz",                      "pregrasp_bank_0.24.npz"),
    "banks_ik":   ("eunha_rl/s0821/banks_ik",                                         "banks_ik"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_digest(root: Path) -> tuple[str, int, int]:
    """Order-independent digest of a directory: sha256 over sorted (relpath, filehash)."""
    h = hashlib.sha256()
    n = 0
    size = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(sha256(p).encode())
        n += 1
        size += p.stat().st_size
    return h.hexdigest(), n, size


def entry_for(dest: Path) -> dict:
    if dest.is_dir():
        digest, n, size = dir_digest(dest)
        return {"kind": "dir", "sha256": digest, "files": n, "bytes": size}
    return {"kind": "file", "sha256": sha256(dest), "bytes": dest.stat().st_size}


def build(strict: bool) -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"schema": 1, "entries": {}}
    missing: list[str] = []

    for role, (rel, name) in SOURCES.items():
        src = REPO / rel
        dst = ARTIFACTS / name
        if not src.exists():
            missing.append(f"{role}: {rel}")
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        e = entry_for(dst)
        e["source"] = rel
        manifest["entries"][role] = e
        print(f"  {role:11s} {name:26s} {e['sha256'][:16]}  {e.get('bytes', 0)/1e6:7.2f} MB")

    if missing:
        print("\nMISSING SOURCES (not bundled):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        if strict:
            return 1

    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(e.get("bytes", 0) for e in manifest["entries"].values())
    print(f"\nbundled {len(manifest['entries'])}/{len(SOURCES)} artifacts, {total/1e6:.1f} MB"
          f" -> {ARTIFACTS.relative_to(REPO)}/manifest.json")
    return 0


def verify() -> int:
    mpath = ARTIFACTS / "manifest.json"
    if not mpath.exists():
        print("no manifest.json — run without --verify first", file=sys.stderr)
        return 2
    manifest = json.loads(mpath.read_text())
    bad = 0
    for role, want in manifest["entries"].items():
        name = SOURCES[role][1] if role in SOURCES else role
        dst = ARTIFACTS / name
        if not dst.exists():
            print(f"MISSING  {role:11s} {name}", file=sys.stderr)
            bad += 1
            continue
        got = entry_for(dst)
        ok = got["sha256"] == want["sha256"]
        print(f"{'OK      ' if ok else 'MISMATCH'} {role:11s} {name}")
        bad += 0 if ok else 1
    print("ARTIFACTS OK" if not bad else f"ARTIFACTS BAD ({bad})")
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--strict", action="store_true", help="fail if any source is missing")
    a = ap.parse_args()
    raise SystemExit(verify() if a.verify else build(a.strict))
