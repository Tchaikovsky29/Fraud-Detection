from typing import NamedTuple
from kfp.dsl import component

@component(
        base_image="tchaikovsky29/fraud-detection-base-env:latest"
        )
def resolve_commit_component() -> NamedTuple(
    "CommitOutput", [("commit_id", str),
                     ("data_mode", str)]):
    import lakefs
    from src.constants import LAKEFS_REPO_NAME, MAIN_BRANCH_NAME, DATA_MODE
    repo = lakefs.Repository(LAKEFS_REPO_NAME)
    commit_id = repo.branch(MAIN_BRANCH_NAME).head.get_commit().id
    CommitOutput = NamedTuple("CommitOutput", [("commit_id", str), ("data_mode", str)])
    return CommitOutput(commit_id=commit_id, data_mode=DATA_MODE)