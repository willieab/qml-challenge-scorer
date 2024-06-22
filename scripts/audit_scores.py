import argparse
import logging
from pathlib import Path

import git
import pandas as pd

from .score_submission import score_submission

path_to_repo = "../ionq-skku-vision-challenge/"

def audit_scores(path_to_repo, max_attempts=10):
    repo = git.Repo(path_to_repo)
    repo.remotes.origin.fetch()

    total_submissions = 0
    recorded = set(pd.read_csv("site/results.csv").Commit)
    for idx, branch in enumerate(repo.refs):
        if branch.name == "origin/main" or not branch.is_remote():
            continue

        # Checkout models in remote branch
        repo.refs[idx].checkout()
        models = repo.commit("HEAD").tree / "trained_models"

        # Traverse submissions
        for model in models.traverse():
            cmt_sha = Path(model.name).stem.split("_")[-1]
            total_submissions += 1

            # Grade every submission with no score on record
            if cmt_sha[:6] not in recorded:
                try:
                    score_submission(path_to_repo, cmt_sha, max_attempts)
                except Exception as e:
                    logging.warn(e)
    return total_submissions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score all submissions.")
    parser.add_argument("--submission_repo", default=".", help="Path to submissions repo.")
    parser.add_argument("--max_retries", type=int, default=10, help="Max attempts at pushing updated results.")
    args = parser.parse_args()

    audit_scores(args.submission_repo, args.max_retries)
