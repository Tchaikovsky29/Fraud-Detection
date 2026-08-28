import lakefs
from src.constants import LAKEFS_REPO_NAME, FEATURES_BRANCH_NAME
import os

repo = lakefs.Repository(LAKEFS_REPO_NAME)
branch = repo.branch(FEATURES_BRANCH_NAME)

def reset_branch():
    """Reset the branch to the latest commit on the main branch."""
    branch.reset_changes()

def upload(remote_path: str, data):
    """Upload data to a lakeFS repository and commit it."""
    branch.object(remote_path).upload(data, pre_sign=False)

def commit(message: str):
    """Optional helper: commit whatever Spark just staged on a branch."""
    ref = branch.commit(message=message)
    print(f"Committed -> {ref.get_commit().id}: {message}")

def download(branch_name: str, remote_path: str):
    """Download data from a lakeFS repository."""
    return repo.branch(branch_name).object(remote_path).download()

def fetch_batch():
    objects = list(branch.objects(prefix="raw/"))
    batch_files = sorted(
        [o.path for o in objects if "batch-" in o.path],
        key=lambda p: int(p.split("batch-")[1].split(".")[0])
    )
    return batch_files

def get_object_checksum(path: str) -> str:
    """
    Returns lakeFS's content-derived checksum for an object -- works even
    on staged (uncommitted) writes.
    """
    path_parts = path.split("/")
    relative_path = os.path.join(*path_parts[2:])  # strip repo + branch

    objects = list(branch.objects(prefix=relative_path))
    part_files = [o for o in objects if o.path.endswith(".parquet")]

    if not part_files:
        raise FileNotFoundError(f"No parquet part file found under prefix '{relative_path}'")
    if len(part_files) > 1:
        raise ValueError(
            f"Expected exactly one part file under '{relative_path}' (coalesce(1)), "
            f"found {len(part_files)}: {[p.path for p in part_files]}"
        )

    stat = branch.object(part_files[0].path).stat()
    return stat.checksum