#!/bin/sh
# torch wheels bundle their own libomp.dylib and LightGBM links Homebrew's. Two OpenMP runtimes in
# one process hang or segfault on macOS, so point torch at the Homebrew copy. Idempotent; rerun
# after any `uv sync` that reinstalls torch.
set -e
cd "$(dirname "$0")/.."
# Some process on this machine marks .venv hidden; CPython then skips the editable-install .pth.
chflags -R nohidden .venv
BREW_OMP="$(brew --prefix libomp)/lib/libomp.dylib"
TORCH_OMP="$(uv run --no-sync python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib" / "libomp.dylib")')"
[ -f "$BREW_OMP" ] || { echo "brew install libomp first"; exit 1; }
if [ -L "$TORCH_OMP" ]; then echo "already linked: $TORCH_OMP"; exit 0; fi
mv "$TORCH_OMP" "$TORCH_OMP.bundled"
ln -s "$BREW_OMP" "$TORCH_OMP"
echo "linked $TORCH_OMP -> $BREW_OMP"
