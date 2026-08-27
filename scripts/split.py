"""Re-split deduplicated datasets into one clean train/val, and write data.yaml.

    python scripts/split.py datasets/gun-knife-thesis datasets/knife-detection-hgvy2
    python scripts/train.py --data data.yaml

Run this AFTER `dedupe.py --apply`. The published splits cannot be reused: dedupe
deletes across split boundaries, so what is left is lopsided, and the original
splits were contaminated anyway (both of these datasets contain images that appear
in their own train and valid at once).

Images stay where they are. YOLO accepts a text file of image paths, so the split
is two lists -- nothing is copied or moved, and provenance survives.

Assignment is by SHA1 of the filename, so the split is identical every run and a
re-import lands the same image on the same side.
"""
import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_for(image):
    return Path(str(image.parent).replace("images", "labels")) / (image.stem + ".txt")


def is_val(image, fraction):
    digest = hashlib.sha1(image.name.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < fraction


def class_counts(images, names):
    counts = Counter()
    for image in images:
        label = label_for(image)
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    counts[names[int(line.split()[0])]] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=ROOT / "data.yaml")
    args = ap.parse_args()

    schema = yaml.safe_load((ROOT / "classes.yaml").read_text(encoding="utf-8"))
    names = [c["name"] for c in sorted(schema["classes"], key=lambda c: c["id"])]

    images = sorted(p for root in args.datasets for p in root.rglob("*") if p.suffix.lower() in SUFFIXES)
    if not images:
        raise SystemExit("no images found -- did the import run?")
    unlabeled = [p for p in images if not label_for(p).exists()]
    train = [p for p in images if not is_val(p, args.val_fraction)]
    val = [p for p in images if is_val(p, args.val_fraction)]

    for split, paths in (("train", train), ("val", val)):
        listing = ROOT / f"{split}.txt"
        listing.write_text("".join(f"{p.resolve()}\n" for p in paths), encoding="utf-8")
        counts = class_counts(paths, names)
        print(f"{split}: {len(paths)} images, {sum(counts.values())} boxes")
        for name, count in counts.most_common():
            print(f"    {name:20} {count}")

    args.out.write_text(
        yaml.safe_dump({"train": str(ROOT / "train.txt"), "val": str(ROOT / "val.txt"),
                        "names": names, "nc": len(names)}, sort_keys=False), encoding="utf-8")
    print(f"\n{args.out} written")
    if unlabeled:
        print(f"{len(unlabeled)} images have no label file -- they train as background negatives")
    return 0


if __name__ == "__main__":
    sys.exit(main())
