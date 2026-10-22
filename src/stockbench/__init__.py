import pathlib
import platform
import warnings

if platform.system() == "Darwin":  # see scripts/fix_macos_venv.sh
    import torch

    _omp = pathlib.Path(torch.__file__).parent / "lib" / "libomp.dylib"
    if _omp.exists() and not _omp.is_symlink():
        warnings.warn(
            "torch ships its own libomp.dylib; with LightGBM in the same process torch CPU kernels "
            "hang or crash. Run scripts/fix_macos_venv.sh.",
            RuntimeWarning,
            stacklevel=1,
        )
