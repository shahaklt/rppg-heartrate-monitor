"""Organize a downloaded UBFC-Phys dataset into the layout the pipeline expects.

UBFC-Phys (Sabour et al., IEEE Trans. Affective Computing 2021) ships one file per
participant per task:
    vid_s{n}_T{k}.avi / vidp{n}_T{k}.avi    RGB webcam video (~35 fps)
    bvp_s{n}_T{k}.csv / bvpp{n}_T{k}.csv     Empatica E4 BVP (64 Hz)
    eda_... / ...                            (EDA, ignored here)
Tasks: T1 rest, T2 speech, T3 arithmetic (NOT neutral/stress/amusement).

This script is tolerant about exact filenames: it globs for .avi/.csv under the
raw folder and parses (subject, task) with a configurable regex, then MOVES
(default) the files into:
    data/ubfc_phys/subj_{n}/T{k}.avi
    data/ubfc_phys/subj_{n}/T{k}.csv
Move avoids duplicating ~50 GB; pass --copy to keep the raw files.

Run:
    python -m data.download_ubfc_phys --raw data/raw
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
from collections import defaultdict
from glob import glob

from config_loader import load_config

_CFG = load_config()
# matches vidp12_T2 / vid_s12_T2 / bvpp3_T1 / bvp_s3_T1 (case-insensitive)
DEFAULT_REGEX = r"(?:vid|bvp)\D*?(\d+)[_-]?T(\d)"


def _extract_zips(raw_dir: str):
    zips = glob(os.path.join(raw_dir, "**", "*.zip"), recursive=True)
    for z in zips:
        dest = os.path.join(raw_dir, os.path.splitext(os.path.basename(z))[0])
        if os.path.isdir(dest):
            continue
        print(f"  extracting {os.path.basename(z)} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)


def _discover(raw_dir: str, pattern: str):
    rx = re.compile(pattern, re.IGNORECASE)
    found = defaultdict(dict)   # subject -> {('T1','video'): path, ...}
    for ext, kind in ((".avi", "video"), (".mp4", "video"),
                      (".csv", "bvp"), (".txt", "bvp")):
        for f in glob(os.path.join(raw_dir, "**", f"*{ext}"), recursive=True):
            name = os.path.basename(f)
            if kind == "bvp" and "bvp" not in name.lower():
                continue   # skip eda/respiration csvs
            m = rx.search(name)
            if not m:
                continue
            subj, task = m.group(1), f"T{m.group(2)}"
            found[int(subj)][(task, kind)] = f
    return found


def organize(raw_dir: str, out_root: str = None, pattern: str = DEFAULT_REGEX,
             move: bool = True):
    out_root = out_root or _CFG.path("data_root")
    os.makedirs(out_root, exist_ok=True)
    _extract_zips(raw_dir)
    found = _discover(raw_dir, pattern)
    if not found:
        raise SystemExit(
            f"No UBFC-Phys files matched under {raw_dir}.\n"
            f"Checked pattern: {pattern}\n"
            "Pass --pattern with a regex capturing (subject)(task) from filenames.")

    op = shutil.move if move else shutil.copy2
    manifest = {"source": "ubfc_phys", "subjects": []}
    for subj in sorted(found):
        sid = f"subj_{subj}"
        subj_dir = os.path.join(out_root, sid)
        os.makedirs(subj_dir, exist_ok=True)
        entry = {"id": sid, "tasks": {}}
        for (task, kind), src in sorted(found[subj].items()):
            ext = ".avi" if kind == "video" else ".csv"
            dst = os.path.join(subj_dir, f"{task}{ext}")
            if os.path.abspath(src) != os.path.abspath(dst):
                op(src, dst)
            entry["tasks"].setdefault(task, {})[kind] = os.path.basename(dst)
        manifest["subjects"].append(entry)
        print(f"  {sid}: tasks {sorted(entry['tasks'])}")

    man_path = _CFG.path("manifest")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nOrganized {len(manifest['subjects'])} subjects -> {out_root}")
    print(f"Manifest: {man_path}")
    print("Next: python -m data.verify_data")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help="folder with downloaded/extracted UBFC-Phys files (or zips)")
    ap.add_argument("--pattern", default=DEFAULT_REGEX,
                    help="regex capturing (subject)(task) from filenames")
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of move (keeps raw, doubles disk use)")
    args = ap.parse_args()
    organize(args.raw, pattern=args.pattern, move=not args.copy)
