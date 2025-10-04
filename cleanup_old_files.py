import sys
from datetime import datetime, timedelta
from pathlib import Path

# Directories to clean (relative to this script's location)
CLEANUP_DIRS = [
    "analysis_results",
    "uploads",
    "validations",
    "vector_indices",
]

# Files older than this (in days) will be deleted
DAYS_OLD = 7


def delete_old_files(base_path: Path, days_old: int = 7):
    cutoff = datetime.now() - timedelta(days=days_old)
    for rel_dir in CLEANUP_DIRS:
        dir_path = base_path / rel_dir
        if not dir_path.exists() or not dir_path.is_dir():
            continue
        for file in dir_path.iterdir():
            if file.is_file():
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime < cutoff:
                    print(f"Deleting {file} (last modified: {mtime})")
                    try:
                        file.unlink()
                    except Exception as e:
                        print(f"Failed to delete {file}: {e}", file=sys.stderr)


if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    delete_old_files(script_dir, DAYS_OLD)
