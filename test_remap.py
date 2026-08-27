"""Round-trip check for scripts/remap_classes.py. Run: python test_remap.py"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REMAP = [sys.executable, str(Path(__file__).parent / "scripts" / "remap_classes.py")]


def fixture(root):
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["Handgun", "Knife", "weapon", "Person"], "nc": 4}))
    (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
    (root / "train" / "labels" / "b.txt").write_text("2 0.4 0.4 0.2 0.2\n3 0.6 0.6 0.3 0.3\n")


def boxes(root):
    return [line for f in root.rglob("labels/*.txt") for line in f.read_text().splitlines() if line.strip()]


def run(*args):
    out = subprocess.run(REMAP + list(args), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def check_schema():
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    from remap_classes import alias_index, load_schema

    schema = load_schema()
    ids = [c["id"] for c in schema["classes"]]
    assert ids == sorted(set(ids)) == list(range(len(ids))), "ids must be unique, dense and ordered"
    alias_index(schema)  # raises if one alias is claimed by two classes


def main():
    check_schema()
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / "ds"
        fixture(root)

        assert "nothing written" in run(str(root), "--map"), "dry run must not write"
        assert len(boxes(root)) == 4, "dry run changed labels"

        run(str(root), "--map", "--apply")
        names = yaml.safe_load((root / "data.yaml").read_text())["names"]
        # Handgun -> pistol(1), Knife -> knife_unknown(23); weapon is refused, Person is dropped.
        assert [b.split()[0] for b in boxes(root)] == ["1", "23"], boxes(root)
        assert names[1] == "pistol" and names[23] == "knife_unknown"

        run(str(root), "--to", "binary", "--apply")
        assert {b.split()[0] for b in boxes(root)} == {"1"}, boxes(root)
        assert len(boxes(root)) == 2, "binary merge must not change box count"
        assert yaml.safe_load((root / "data.yaml").read_text())["names"] == ["person", "weapon"]

        # An unknown class name has to fail loudly rather than be guessed at.
        other = tmp / "ds2"
        fixture(other)
        (other / "data.yaml").write_text(yaml.safe_dump({"names": ["blorp"], "nc": 1}))
        assert subprocess.run(REMAP + [str(other), "--map"], capture_output=True).returncode == 1

        print("test_remap ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
