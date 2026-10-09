import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelEntry:
    name: str
    directory: Path
    model_path: Path
    index_path: Path | None
    #: Size of the model file in bytes and its modification time (Unix).
    size_bytes: int = 0
    modified: float = 0.0


def discover_models(models_root: str | Path) -> list[ModelEntry]:
    """Discover one RVC model per direct child directory.

    A directory is listed when it contains at least one ``.pth`` file.
    When multiple files exist, names are sorted case-insensitively and the
    first file is selected.  An ``added_*.index`` file is preferred over
    other index files.
    """
    root = Path(models_root)
    if not root.is_dir():
        return []

    entries = []
    for directory in sorted(
        # Hidden folders are skipped, including in-progress imports (.import-*).
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name.casefold(),
    ):
        model_files = sorted(
            directory.glob("*.pth"), key=lambda path: path.name.casefold()
        )
        if not model_files:
            continue

        index_files = sorted(
            directory.glob("*.index"),
            key=lambda path: (
                not path.name.casefold().startswith("added_"),
                path.name.casefold(),
            ),
        )
        try:
            stat = model_files[0].stat()
        except OSError:
            continue
        entries.append(
            ModelEntry(
                name=directory.name,
                directory=directory.resolve(),
                model_path=model_files[0].resolve(),
                index_path=index_files[0].resolve() if index_files else None,
                size_bytes=stat.st_size,
                modified=stat.st_mtime,
            )
        )
    return entries


#: Characters a model folder name cannot contain (also invalid on NTFS/FAT).
INVALID_NAME_CHARACTERS = '/\\:*?"<>|'


def model_name_problem(name):
    """Why ``name`` cannot be a model folder name, or ``None`` if it can."""
    if not name or not name.strip():
        return "empty"
    if name != name.strip() or name.endswith("."):
        return "edges"
    if name.startswith("."):
        return "hidden"
    if any(character in INVALID_NAME_CHARACTERS or ord(character) < 32 for character in name):
        return "characters"
    if len(name.encode("utf-8")) > 200:
        return "too_long"
    return None


def rename_model_folder(directory, new_name):
    """Rename a model folder within its parent; returns the new path.

    Case-only renames (``voice`` -> ``Voice``) go through a temporary name,
    which case-insensitive filesystems such as NTFS need.
    """
    directory = Path(directory)
    target = directory.parent / new_name
    same_folder = target.exists() and os.path.samefile(target, directory)
    if target.exists() and not same_folder:
        raise FileExistsError(new_name)
    if same_folder and directory.name != new_name:
        temporary = directory.parent / f".rename-{os.getpid()}-{directory.name}"
        os.rename(directory, temporary)
        directory = temporary
    os.rename(directory, target)
    return target


class TrashError(Exception):
    pass


def move_to_trash(path):
    """Move a model folder to the desktop Trash (recoverable), via ``gio``."""
    gio = shutil.which("gio")
    if gio is None:
        raise TrashError("gio was not found, so the folder cannot be moved to the Trash")
    completed = subprocess.run(
        [gio, "trash", "--", os.fspath(path)], capture_output=True, text=True, timeout=60
    )
    if completed.returncode != 0:
        raise TrashError(completed.stderr.strip() or f"gio trash failed ({completed.returncode})")
