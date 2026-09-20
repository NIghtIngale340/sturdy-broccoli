#!/usr/bin/env python3
"""Verify that every recorded artifact still has the bytes it was recorded with.

RDR-011. `results/*_provenance.json` and the SHA-256 fields in
`results/*_bpw_manifest.json` were previously written once and never checked,
so a rebuilt or overwritten artifact could not be distinguished from the one
the results were computed on. This script is the check, and Gate 0 runs it.

Usage:
    python3 scripts/verify_provenance.py
    python3 scripts/verify_provenance.py --provenance results/sprint0_provenance.json
    python3 scripts/verify_provenance.py --write   # re-record current bytes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quant_utils import fingerprint_source_dir, sha256_file


def check_provenance(path: Path) -> tuple[int, int, list[str]]:
    """Returns (checked, missing, mismatches) for one provenance file."""
    entries = json.loads(path.read_text())
    checked = missing = 0
    bad: list[str] = []
    for rel, meta in sorted(entries.items()):
        f = ROOT / rel
        if not f.exists():
            print(f"  [missing ] {rel}")
            missing += 1
            continue
        size_ok = f.stat().st_size == meta["bytes"]
        digest = sha256_file(f)
        hash_ok = digest == meta["sha256"]
        checked += 1
        if size_ok and hash_ok:
            print(f"  [ ok     ] {rel}")
        else:
            bad.append(rel)
            print(f"  [MISMATCH] {rel}")
            if not size_ok:
                print(f"             bytes  recorded {meta['bytes']} "
                      f"actual {f.stat().st_size}")
            if not hash_ok:
                print(f"             sha256 recorded {meta['sha256']}")
                print(f"             sha256 actual   {digest}")
    return checked, missing, bad


def check_manifest(path: Path) -> tuple[int, int, list[str]]:
    """Verify the per-file SHA-256 and the source fingerprint in a BPW manifest."""
    data = json.loads(path.read_text())
    checked = missing = 0
    bad: list[str] = []

    src = data.get("source_sha256")
    merged = data.get("merged_dir")
    if src and merged:
        d = ROOT / merged
        if not d.is_dir():
            print(f"  [missing ] source dir {merged}")
            missing += 1
        else:
            actual = fingerprint_source_dir(d)
            if actual == src:
                print(f"  [ ok     ] source fingerprint {merged}")
                checked += 1
            else:
                bad.append(f"{path.name}:source")
                print(f"  [MISMATCH] source fingerprint {merged}")
                print(f"             recorded {src}")
                print(f"             actual   {actual}")
    else:
        print(f"  [ warn   ] {path.name} has no source_sha256 (written before RDR-011)")

    for m in data.get("measurements", []):
        rel = m["path"]
        f = ROOT / rel
        recorded = m.get("sha256")
        if recorded is None:
            print(f"  [ warn   ] {rel} has no recorded sha256")
            continue
        if not f.exists():
            print(f"  [missing ] {rel}")
            missing += 1
            continue
        digest = sha256_file(f)
        checked += 1
        if digest == recorded:
            print(f"  [ ok     ] {rel}")
        else:
            bad.append(rel)
            print(f"  [MISMATCH] {rel}")
            print(f"             recorded {recorded}")
            print(f"             actual   {digest}")
    return checked, missing, bad


def write_provenance(path: Path, targets: list[str]) -> None:
    out = {}
    for rel in sorted(targets):
        f = ROOT / rel
        if not f.exists():
            print(f"  skipping absent {rel}")
            continue
        out[rel] = {"sha256": sha256_file(f), "bytes": f.stat().st_size}
        print(f"  recorded {rel}")
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path} ({len(out)} artifacts).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provenance", nargs="*", default=None,
                    help="default: results/*_provenance.json")
    ap.add_argument("--manifest", nargs="*", default=None,
                    help="default: results/*_bpw_manifest.json")
    ap.add_argument("--write", metavar="PROVENANCE_JSON", default=None,
                    help="re-record the current bytes of the files already listed "
                         "in this provenance file, then exit")
    args = ap.parse_args()

    if args.write:
        target = ROOT / args.write
        existing = json.loads(target.read_text()) if target.exists() else {}
        if not existing:
            print(f"ERROR: {args.write} lists no artifacts to re-record.")
            return 2
        write_provenance(target, list(existing))
        return 0

    prov = ([ROOT / p for p in args.provenance] if args.provenance
            else sorted((ROOT / "results").glob("*_provenance.json")))
    mans = ([ROOT / p for p in args.manifest] if args.manifest
            else sorted((ROOT / "results").glob("*_bpw_manifest.json")))

    if not prov and not mans:
        print("ERROR: nothing to verify - no provenance files or BPW manifests found.")
        return 2

    total_checked = total_missing = 0
    all_bad: list[str] = []

    for p in prov:
        print(f"\nprovenance: {p.relative_to(ROOT)}")
        c, m, bad = check_provenance(p)
        total_checked += c; total_missing += m; all_bad += bad

    for p in mans:
        print(f"\nmanifest: {p.relative_to(ROOT)}")
        c, m, bad = check_manifest(p)
        total_checked += c; total_missing += m; all_bad += bad

    print("\n" + "=" * 72)
    status = "FAIL" if all_bad else ("INCOMPLETE" if total_missing else "PASS")
    print(f" PROVENANCE: {status}   "
          f"({total_checked} verified, {total_missing} missing, {len(all_bad)} mismatched)")
    print("=" * 72)
    if all_bad:
        print(" Recorded results were computed on DIFFERENT bytes than are on disk.")
        print(" Do not cite any result touching:")
        for b in all_bad:
            print(f"   {b}")
        return 1
    if total_missing:
        print(" Some recorded artifacts are absent (they are git-ignored, so this is")
        print(" expected on a fresh clone). Nothing present has changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
