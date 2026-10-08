"""Copy dropped .pth/.index files into the models folder.

Each ``.pth`` becomes its own model folder named after the file.  ``.index``
files dropped together with a single ``.pth`` go into its folder; with several
``.pth`` files, an index goes with the model whose name it contains.
"""

import os
import shutil
import uuid
from pathlib import Path


class ModelImportError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def plan_import(paths):
    """Return ``[(model_file, [index_files])]`` for the dropped paths."""
    files = [Path(path) for path in paths]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ModelImportError("import_failed", f"File not found: {missing[0]}")
    models = [path for path in files if path.suffix.casefold() == ".pth"]
    indexes = [path for path in files if path.suffix.casefold() == ".index"]
    if not models:
        raise ModelImportError(
            "import_no_model_file",
            "Drop an RVC model's .pth file (its .index can be dropped with it).",
        )
    if len(models) == 1:
        return [(models[0], indexes)]
    plan = []
    for model in models:
        stem = model.stem.casefold()
        plan.append((model, [index for index in indexes if stem in index.stem.casefold()]))
    return plan


def unique_folder_name(models_root, name):
    candidate = name
    number = 2
    while (Path(models_root) / candidate).exists():
        candidate = f"{name} ({number})"
        number += 1
    return candidate


def import_models(paths, models_root):
    """Copy the dropped models in and return the new model (folder) names.

    Files are copied into a hidden temporary folder that is renamed into
    place only when complete, so an interrupted copy never shows up as a
    half-imported model.
    """
    plan = plan_import(paths)
    os.makedirs(models_root, exist_ok=True)
    names = []
    for model_file, index_files in plan:
        name = unique_folder_name(models_root, model_file.stem.strip() or "Model")
        # os.mkdir honours the umask (tempfile.mkdtemp would force 0700).
        staging = os.path.join(models_root, f".import-{uuid.uuid4().hex}")
        try:
            os.mkdir(staging)
            for source in (model_file, *index_files):
                shutil.copy2(source, os.path.join(staging, source.name))
            os.rename(staging, os.path.join(models_root, name))
        except OSError as error:
            shutil.rmtree(staging, ignore_errors=True)
            raise ModelImportError("import_failed", str(error)) from error
        names.append(name)
    return names
