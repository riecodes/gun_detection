"""Train on one or more remapped datasets.

    python scripts/train.py datasets/gun-knife-thesis datasets/knife-detection-hgvy2
    python scripts/train.py datasets/* --epochs 100 --batch 8

Every dataset must already be through `remap_classes.py --map --apply`, so they
all share our class ids. Ultralytics accepts a list of image directories, so the
datasets are pointed at directly -- nothing is copied or merged on disk.
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def split_dirs(datasets, split):
    dirs = [d for root in datasets for d in root.rglob(f"{split}/images") if d.is_dir()]
    return [str(d) for d in sorted(dirs)]


def write_data_yaml(datasets, out):
    schema = yaml.safe_load((ROOT / "classes.yaml").read_text(encoding="utf-8"))
    names = [c["name"] for c in sorted(schema["classes"], key=lambda c: c["id"])]
    train, val = split_dirs(datasets, "train"), split_dirs(datasets, "valid") or split_dirs(datasets, "val")
    if not train:
        raise SystemExit("no */train/images found -- are these unzipped YOLO exports?")
    if not val:
        raise SystemExit("no */valid/images found -- every dataset needs a validation split")
    data = {"train": train, "val": val, "names": names, "nc": len(names)}
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"{out}: {len(train)} train dir(s), {len(val)} val dir(s), {len(names)} classes")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", type=Path)
    ap.add_argument("--data", type=Path, help="use an existing data.yaml (what scripts/split.py writes)")
    ap.add_argument("--model", default="yolov8n.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8, help="8 fits a 4GB card at 640px")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default="ph-weapons")
    ap.add_argument("--dry-run", action="store_true", help="write the data.yaml and stop")
    args = ap.parse_args()

    if args.data:
        data = args.data
    elif args.datasets:
        data = write_data_yaml(args.datasets, ROOT / "data.yaml")
    else:
        ap.error("give dataset directories or --data data.yaml")
    if args.dry_run:
        return 0

    from ultralytics import YOLO

    YOLO(args.model).train(
        data=str(data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        cache=False,  # ponytail: 4GB card, RAM caching of a 20k-image set will thrash; enable if you get more VRAM/RAM
    )
    print("weights: runs/detect/" + args.name + "/weights/best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
