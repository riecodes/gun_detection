"""Download a Roboflow Universe dataset and remap it onto our class schema.

    pip install roboflow
    export ROBOFLOW_API_KEY=...          # PowerShell: $env:ROBOFLOW_API_KEY="..."
    python scripts/import_dataset.py crime-detection-zbmr9/gun-knife-thesis --version 1

Downloads to datasets/<project>/ and runs remap_classes.py --map over it, so the
export lands already speaking our ids. Nothing is applied until the mapping covers
every class name in the export -- an unmapped name stops the import instead of
being guessed at.

Record the dataset's licence in docs/DATASET_SOURCES.md before you import it.
Universe licences run from CC BY 4.0 to unstated, and the project page is the only
place the licence is shown.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_dotenv(path=ROOT / ".env"):
    """Tiny .env reader -- one key, not worth a dependency."""
    if not path.exists():
        return {}
    pairs = (line.partition("=") for line in path.read_text(encoding="utf-8").splitlines())
    return {k.strip(): v.strip().strip("\"'") for k, _, v in pairs if k.strip() and not k.startswith("#")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug", help="workspace/project from the Universe URL")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--apply", action="store_true", help="write the remap; otherwise dry run")
    ap.add_argument("--keep-person", action="store_true", help="only for exhaustively person-labeled sets")
    args = ap.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY") or read_dotenv().get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("set ROBOFLOW_API_KEY (any free Roboflow account provides one)")
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("pip install roboflow")

    workspace, _, project = args.slug.partition("/")
    out = ROOT / "datasets" / project
    print(f"downloading {args.slug} v{args.version} -> {out}")
    Roboflow(api_key=key).workspace(workspace).project(project).version(args.version).download(
        "yolov8", location=str(out)
    )

    cmd = [sys.executable, str(ROOT / "scripts" / "remap_classes.py"), str(out), "--map"]
    if args.apply:
        cmd.append("--apply")
    if args.keep_person:
        cmd.append("--keep-person")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
