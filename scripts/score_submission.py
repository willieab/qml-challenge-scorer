#
# Copyright (c)2024. IonQ, Inc. All rights reserved.
#
import argparse
from pathlib import Path
import subprocess
import sys
from time import time

import git
import torch

from ionqvision.modules import BinaryMNISTClassifier, MODELS_DIR

RESULTS_FILE = "site/results.csv"

def grade_submission(path):
    # Fetch model and validation set
    model = BinaryMNISTClassifier.load_model(path)
    with open("scripts/validation_set.pkl", "rb") as f:
        validation = torch.load(f)

    # Compute score and generate report
    acc = model.compute_accuracy(validation)
    qc = model.quantum_layer.layer_qc
    return [100*acc, qc.num_gates()["2q"], qc.num_parameters]


def attempt_record_results(repo, report):
    # Pull changes in remote, if any (from concurrent runs)
    origin = repo.remotes.origin
    origin.pull()

    # Time results.csv remains locked, which is potentially dangerous bc
    # another runner may push changes in the meantime
    t0 = time()
    with open(RESULTS_FILE, "a") as resfile:
        print(",".join(map(str, report)), file=resfile)

    # Push changes
    repo.git.add(RESULTS_FILE)
    repo.index.commit(f"Scoring {report[1][:6]} from {report[0]}")
    origin.push().raise_if_error()
    print(f"Locked results.csv for {time() - t0:.03f}s...")


def attempt_retrieve_submission(submission_repo, commit):
    # If no trained model exists for this commit, exit gracefully
    path = list(Path(submission_repo).joinpath(MODELS_DIR).glob(f"model_*_{commit}.zip"))
    if not len(path) == 1:
        raise ValueError(f"No trained model for commit {commit} found...")
    return path[0]


def get_metadata(path_to_repo, commit_sha):
    commit = git.Repo(path_to_repo).commit(commit_sha)
    team = " ".join(submission.name.split("_")[1:-1]).replace("-", " ")
    return [team.title(), commit.hexsha[:6], commit.committed_date]


def record_results(results, max_attempts=10):
    repo = git.Repo(search_parent_directories=True)
    suffix = {1: "st", 2: "nd", 3: "rd"}
    for k in range(max_attempts):
        try:
            if k > 0:
                print(f"Attempting push for the {k}{suffix.get(k, 'th')} time...")
            attempt_record_results(repo, results)
            break
        except Exception as e:
            print(f"Git error encountered: {e}")
            repo.git.reset("--hard")
    if k == max_attempts - 1:
        raise RuntimeError("Unable to update results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score submission.")
    parser.add_argument("commit_sha", help="SHA of commit that created the submission.")
    parser.add_argument("--submission_repo", default=".", help="Path to submissions repo.")
    parser.add_argument("--max_retries", type=int, default=10, help="Max attempts at pushing updated results.")
    parser.add_argument("--check_exist", action="store_true", default=False, help="Only check for existence of matching submission.")
    args = parser.parse_args()

    # Retrieve submission, if any
    try:
        submission = attempt_retrieve_submission(args.submission_repo, args.commit_sha)
    except ValueError as e:
        print(e)
        sys.exit(0)

    # If we're only checking existence, were done!
    if args.check_exist:
        sys.exit(0)

    # Grade submission and gather metadata
    result = grade_submission(submission)
    metadata = get_metadata(args.submission_repo, args.commit_sha)

    # Push report
    record_results(metadata + result, args.max_retries)

    # Update leaderboard
    subprocess.run(
        "jupyter nbconvert leaderboard.ipynb --to notebook --execute --output leaderboard.ipynb",
        cwd="site",
        shell=True
    )
    repo = git.Repo(search_parent_directories=True)
    repo.git.add("site")
    repo.index.commit(f"Updating leaderboard...")
    origin = repo.remotes.origin
    origin.push()
    print(f"Success! Recorded {result[0]} score for commit {args.commit_sha} from {metadata[0]}!")
