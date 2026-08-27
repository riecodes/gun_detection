"""Verify the training stack imports. Run: python scripts/check_env.py

This machine has repeatedly lost individual files out of installed packages --
`matplotlib/colors.py`, `torchvision/transforms/transforms.py`, `psutil/_psutil_windows.pyd`
have each gone missing while their neighbours stayed intact. The symptom is always a
ModuleNotFoundError for a submodule of a package pip reports as installed, and the fix
is always `pip install --force-reinstall --no-deps <package>`.

Run this before a long training job rather than discovering it 40 minutes in.
"""
import importlib
import sys

MODULES = [
    "numpy", "cv2", "PIL", "matplotlib.pyplot", "yaml", "tqdm", "requests",
    "mpmath", "sympy", "sympy.utilities.iterables", "pandas", "psutil", "scipy",
    "torch", "torchvision", "torchvision.transforms.transforms", "torch._dynamo",
    "ultralytics",
]


def main():
    broken = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            broken.append((name, f"{type(exc).__name__}: {exc}"))

    for name, error in broken:
        print(f"FAIL {name}\n     {error}", file=sys.stderr)

    if broken:
        roots = sorted({name.split(".")[0] for name, _ in broken})
        print(f"\n{len(broken)} broken. Repair with:", file=sys.stderr)
        print(f"  python -m pip install --force-reinstall --no-deps --no-cache-dir {' '.join(roots)}", file=sys.stderr)
        print("  (torchvision needs --index-url https://download.pytorch.org/whl/cu118 to keep the CUDA build)", file=sys.stderr)
        return 1

    import torch

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"{len(MODULES)} modules ok | torch {torch.__version__} | {device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
