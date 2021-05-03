"""Standard library toolkit"""
from datetime import datetime
from pathlib import Path


def date_versioned_dir(dst: Path) -> Path:
    """Make directory with name of current date in destination directory and return it"""
    versioned_dir = dst / str(datetime.now().date())
    versioned_dir.mkdir(exist_ok=True, parents=True)
    return versioned_dir
