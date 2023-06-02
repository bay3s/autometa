from subprocess import check_output, CalledProcessError
from functools import lru_cache
import os.path


@lru_cache(maxsize=1)
def repository_root() -> str:
    """
    Returns the absolute path of the repository root.

    Returns:
      str
    """
    try:
        base = check_output(["git", "rev-parse", "--show-toplevel"])
    except CalledProcessError:
        raise IOError("Current working directory is not a git repository")

    return base.decode("utf-8").strip()


def absolute_path(relative_path: str) -> str:
    """
    Returns the absolute path for a path given relative to the root of the git repository.

    Args:
      relative_path (str): Relative path of the file / folder in the repo.

    Returns:
        str
    """
    return os.path.join(repository_root(), relative_path)
