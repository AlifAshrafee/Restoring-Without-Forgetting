#!/usr/bin/env python3
"""
Merge five named degradation domains into one joint (all-in-one) set.
Source structure:
    <root>/<split>/<domain>/{input,target}_crops/<name>
Output structure:
    <root>/<split>/joint_degradations/{input,target}_crops/<domain>_<name>
"""

import argparse
import os
import shutil
from pathlib import Path

DOMAINS = ["D1_denoise", "D2_deblur", "D3_derain", "D4_dehaze", "D5_lowlight"]
SPLITS = ["train", "val"]
OUT_NAME = "joint_degradations"
IN_SUB, TG_SUB = "input_crops", "target_crops"


def link(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:  # copy
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./Datasets/", 
                    type=str, help="Root directory containing the datasets.")
    ap.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    args = ap.parse_args()
    root = Path(args.root).expanduser()

    for split in SPLITS:
        out_in = root / split / OUT_NAME / IN_SUB
        out_tg = root / split / OUT_NAME / TG_SUB
        out_in.mkdir(parents=True, exist_ok=True)
        out_tg.mkdir(parents=True, exist_ok=True)

        print(f"--- {split} ({args.link_mode}) ---")
        total = 0
        for domain in DOMAINS:
            in_src = root / split / domain / IN_SUB
            tg_src = root / split / domain / TG_SUB
            n = 0
            with os.scandir(in_src) as it:
                for e in it:
                    if not e.is_file():
                        continue
                    new = f"{domain}_{e.name}"
                    link(Path(e.path), out_in / new, args.link_mode)
                    link(tg_src / e.name, out_tg / new, args.link_mode)
                    n += 1
            total += n
            print(f"{domain}: {n} pairs")
        print(f"TOTAL {total} pairs\n")


if __name__ == "__main__":
    main()