#!/usr/bin/env python
"""
Find metaRotation_*.nc files that contain only a single file_starttime
entry.

Such files are the symptom of the bug where createMetaRotation() wrote an
output file for a day with no level1detect data at all, seeded only with
the carried-over rotation estimate from the previous day/config (see
VISSSlib.matching.createMetaRotation). They contain no information
actually derived from that day and are candidates for deletion/rerun.

Usage
-----
    python find_single_point_metaRotation.py /projekt1/ag_maahn/data_obs_nobackup/hyytiala
    python find_single_point_metaRotation.py --delete /path/to/metaRotation
"""

import argparse
import sys
from pathlib import Path

import xarray as xr


def find_single_point_files(roots, pattern="metaRotation_*.nc", threshold=1):
    matches = []
    broken = []
    nScanned = 0
    for root in roots:
        for fname in sorted(Path(root).rglob(pattern)):
            nScanned += 1
            try:
                with xr.open_dataset(fname) as ds:
                    n = ds.sizes.get("file_starttime")
            except Exception as e:
                broken.append((fname, str(e)))
                continue
            if n is None:
                broken.append((fname, "no file_starttime dimension"))
            elif n <= threshold:
                matches.append((fname, n))
    return matches, broken, nScanned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots", nargs="+", help="Directories to search recursively"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Flag files with file_starttime size <= threshold (default: 1)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete matching files instead of just listing them",
    )
    args = parser.parse_args()

    matches, broken, nScanned = find_single_point_files(
        args.roots, threshold=args.threshold
    )

    for fname, n in matches:
        print(f"{n}\t{fname}")

    if broken:
        print(f"\n{len(broken)} file(s) could not be opened:", file=sys.stderr)
        for fname, err in broken:
            print(f"  {fname}: {err}", file=sys.stderr)

    print(
        f"\n{len(matches)} of {nScanned} scanned metaRotation files have "
        f"<= {args.threshold} timestamp(s)",
        file=sys.stderr,
    )

    if args.delete:
        for fname, _ in matches:
            fname.unlink()
        print(f"deleted {len(matches)} file(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
