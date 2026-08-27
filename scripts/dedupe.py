"""Find images repeated across YOLO dataset exports.

    python scripts/dedupe.py datasets/ph-fine datasets/import-a datasets/import-b
    python scripts/dedupe.py datasets/* --apply

Public weapon datasets descend from a handful of the same corpora, so imports
overlap heavily. Run this over the union BEFORE splitting train/val -- the same
image landing on both sides inflates every metric.

# ponytail: exact dHash equality only -- catches rescales and recompression, misses
# crops and heavy edits. If those show up, switch to a Hamming radius over a BK-tree
# rather than an O(n^2) scan of every pair.

Earlier directories win: list the curated set first and duplicates are deleted
from the imports, never from it.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(path, size=8):
    """64-bit difference hash: survives rescaling and recompression, which is how
    the same source image differs between two datasets."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = value << 1 | int(bit)
    return value


def label_for(image):
    label = Path(str(image.parent).replace("images", "labels")) / (image.stem + ".txt")
    return label if label.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", type=Path)
    ap.add_argument("--apply", action="store_true", help="delete duplicates; otherwise dry run")
    args = ap.parse_args()

    seen, groups, unreadable = {}, defaultdict(list), []
    for root in args.datasets:
        for image in sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES):
            value = dhash(image)
            if value is None:
                unreadable.append(image)
            elif value in seen:
                groups[seen[value]].append(image)
            else:
                seen[value] = image

    duplicates = [d for dups in groups.values() for d in dups]
    for keeper, dups in groups.items():
        print(f"{keeper}  <- {len(dups)} duplicate(s)")
        for dup in dups:
            print(f"    {dup}")
            if args.apply:
                label = label_for(dup)
                dup.unlink()
                if label:
                    label.unlink()

    print(f"\n{len(seen)} unique, {len(duplicates)} duplicate, {len(unreadable)} unreadable")
    for image in unreadable:
        print(f"  unreadable: {image}", file=sys.stderr)
    if duplicates and not args.apply:
        print("nothing deleted; rerun with --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
