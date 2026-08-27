"""Remap the classes of a YOLO dataset export, in place.

    python scripts/remap_classes.py DIR --map aliases   # foreign class names -> our schema
    python scripts/remap_classes.py DIR --to tier       # our schema -> one class per group
    python scripts/remap_classes.py DIR --to binary     # our schema -> person / weapon
    python scripts/remap_classes.py --table             # regenerate the README class table

Dry run by default; pass --apply to write. DIR is an unzipped YOLO export: a
data.yaml plus */labels/*.txt.

Fine -> coarse is free and repeatable, coarse -> fine needs relabeling by hand,
so the fine labels stay the stored asset and every coarse set is generated.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

SCHEMA = Path(__file__).resolve().parent.parent / "classes.yaml"
DROP = "\0drop"  # sentinel: box is deleted rather than remapped


def normalize(name):
    """'Screw Driver', 'screw-driver' and 'SCREWDRIVER' are the same alias."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_schema(path=SCHEMA):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_names(data_yaml):
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def alias_index(schema):
    index = {}
    for cls in schema["classes"]:
        for alias in [cls["name"], *cls.get("aliases", [])]:
            key = normalize(alias)
            if index.get(key, cls["id"]) != cls["id"]:
                raise SystemExit(f"classes.yaml: alias {alias!r} claimed by two classes")
            index[key] = cls["id"]
    return index


def build_mapping(source_names, schema, mode, keep_person):
    """source name -> (new name, new id), or DROP, or None when unmapped."""
    if mode == "aliases":
        index, refused = alias_index(schema), {normalize(n) for n in schema["unmapped_review"]}
        by_id = {c["id"]: c for c in schema["classes"]}
        out = {}
        for name in source_names:
            cid = index.get(normalize(name))
            if cid == 0 and not keep_person:
                out[name] = DROP  # partial person labels poison the background class
            elif cid is not None:
                out[name] = (by_id[cid]["name"], cid)
            elif normalize(name) in refused:
                out[name] = DROP
            else:
                out[name] = None
        return out

    by_name = {c["name"]: c for c in schema["classes"]}
    if mode == "tier":
        targets = schema["groups"]
    else:
        targets = ["person", "weapon"]
    out = {}
    for name in source_names:
        cls = by_name.get(name)
        if cls is None:
            out[name] = None
        elif mode == "tier":
            out[name] = (cls["group"], targets.index(cls["group"]))
        else:
            label = "person" if cls["group"] == "context" else "weapon"
            out[name] = (label, targets.index(label))
    return out, targets


def rewrite(root, source_names, mapping, targets, apply):
    before, after, dropped = Counter(), Counter(), 0
    for label in root.rglob("labels/*.txt"):
        kept = []
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            head, _, rest = line.partition(" ")
            name = source_names[int(head)]
            before[name] += 1
            target = mapping[name]
            if target is DROP or target is None:
                dropped += 1
                continue
            after[target[0]] += 1
            kept.append(f"{target[1]} {rest}")
        if apply:
            label.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    if apply:
        data_yaml = root / "data.yaml"
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        data["names"] = list(targets)
        data["nc"] = len(targets)
        data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return before, after, dropped


def markdown_table(schema):
    rows = ["| id | class | group | Filipino | notes |", "|---:|---|---|---|---|"]
    for cls in schema["classes"]:
        rows.append(
            f"| {cls['id']} | `{cls['name']}` | {cls['group']} | "
            f"{cls.get('filipino', '')} | {cls.get('notes', '')} |"
        )
    return "\n".join(rows)


def write_readme(schema, readme=SCHEMA.parent / "README.md"):
    """Replace the block between the taxonomy markers so the table cannot drift."""
    start, end = "<!-- taxonomy:start -->", "<!-- taxonomy:end -->"
    text = readme.read_text(encoding="utf-8")
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    body = "\n".join([start, markdown_table(schema), end])
    readme.write_text(head + body + tail, encoding="utf-8")
    print(f"updated {readme}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, nargs="?")
    ap.add_argument("--table", action="store_true", help="print the class table as markdown")
    ap.add_argument("--write-readme", action="store_true", help="write that table into README.md")
    ap.add_argument("--map", dest="mode", const="aliases", action="store_const")
    ap.add_argument("--to", dest="mode", choices=["tier", "binary"])
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry run")
    ap.add_argument("--keep-person", action="store_true", help="keep person boxes (only for exhaustively labeled sets)")
    ap.add_argument("--drop-unmapped", action="store_true", help="delete boxes whose class has no mapping")
    args = ap.parse_args()
    if args.table or args.write_readme:
        schema = load_schema()
        write_readme(schema) if args.write_readme else print(markdown_table(schema))
        return 0
    if not args.dataset:
        ap.error("give a dataset directory")
    if not args.mode:
        ap.error("pass --map aliases or --to tier|binary")

    root = args.dataset
    if not (root / "data.yaml").exists():
        raise SystemExit(f"{root}: no data.yaml -- point this at an unzipped YOLO export")

    schema, source_names = load_schema(), read_names(root / "data.yaml")
    result = build_mapping(source_names, schema, args.mode, args.keep_person)
    mapping, targets = result if isinstance(result, tuple) else (result, None)
    if targets is None:
        by_id = {c["id"]: c["name"] for c in schema["classes"]}
        targets = [by_id[i] for i in sorted(by_id)]

    unmapped = [n for n, t in mapping.items() if t is None]
    if unmapped and not args.drop_unmapped:
        print("unmapped class names -- add an alias to classes.yaml or rerun with --drop-unmapped:", file=sys.stderr)
        for name in unmapped:
            print(f"  {name}", file=sys.stderr)
        return 1

    before, after, dropped = rewrite(root, source_names, mapping, targets, args.apply)
    print(f"{'applied' if args.apply else 'dry run'}: {root}")
    for name, count in before.most_common():
        target = mapping[name]
        print(f"  {name:24} {count:6}  ->  {'(dropped)' if target in (DROP, None) else target[0]}")
    print(f"  {'kept':24} {sum(after.values()):6}   dropped {dropped}")
    if not args.apply:
        print("nothing written; rerun with --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
