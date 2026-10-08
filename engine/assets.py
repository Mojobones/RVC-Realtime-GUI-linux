"""Inference assets that are not stored in Git (see README: Inference assets).

Every model needs the HuBERT/ContentVec feature extractor; the RMVPE pitch
detector needs its own weights.  FCPE and PM ship with their Python packages.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
HUBERT_DIR = os.path.join(ASSETS_DIR, "hubert_base")
HUBERT_WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")
RMVPE_PATH = os.path.join(ASSETS_DIR, "rmvpe", "rmvpe.pt")


def missing_assets(f0method=None, assets_dir=ASSETS_DIR):
    """Return the missing asset files, relative to the project folder.

    With ``f0method=None`` every asset is checked; otherwise only the ones the
    given pitch detector needs.
    """
    hubert_dir = os.path.join(assets_dir, "hubert_base")
    missing = []
    if not os.path.isfile(os.path.join(hubert_dir, "config.json")):
        missing.append("assets/hubert_base/config.json")
    if not any(os.path.isfile(os.path.join(hubert_dir, name)) for name in HUBERT_WEIGHT_FILES):
        missing.append("assets/hubert_base/pytorch_model.bin")
    if f0method in (None, "rmvpe") and not os.path.isfile(
        os.path.join(assets_dir, "rmvpe", "rmvpe.pt")
    ):
        missing.append("assets/rmvpe/rmvpe.pt")
    return missing
