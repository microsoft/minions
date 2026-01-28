from datasets import load_dataset
import logging
from pathlib import Path
import sys
import subprocess
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from microbots import WritingBot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

"""
Difficulty Levels:
1-4 hours
15 min - 1 hour
<15 min fix
>4 hours
"""
# SWE_BENCH_SUITE = "SWE-bench/SWE-bench_Lite"
SWE_BENCH_SUITE = "SWE-bench/SWE-bench_Verified"
# TEST_DIR = Path(__file__).parent.resolve() / "test_dir"
TEST_DIR = Path("/tmp/swe_bench_test_dir")


def clone_repo_and_checkout(repo_url, commit_hash, dest_path):
    logger.info(f"Cloning repository {repo_url} into {dest_path}")
    subprocess.run(["git", "clone", repo_url, str(dest_path)], check=True)
    subprocess.run(["git", "checkout", commit_hash], cwd=str(dest_path), check=True)
    logger.info(f"Checked out to commit {commit_hash}")


def setup_test_directory(dataset):
    test_path = TEST_DIR / dataset['instance_id']
    # delete if already exist
    if test_path.exists():
        subprocess.run(["rm", "-rf", str(test_path)], check=True)
    test_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Test directory set up at: {test_path}")

    # clone repo dataset['repo'] and checkout to dataset['base_commit']
    clone_repo_and_checkout(f"https://github.com/{dataset['repo']}.git", dataset['base_commit'], test_path)


def verify_fix(dataset, test_path):
    logger.info(f"Verifying fix for dataset: {dataset['instance_id']}")

    for fail_to_pass_test in dataset["FAIL_TO_PASS"]:
        logger.info(f"Running test expected to fail then pass: {fail_to_pass_test}")
        try:
            subprocess.run(
                [sys.executable, str(test_path / fail_to_pass_test)],
                check=True
            )
            logger.error(f"Test {fail_to_pass_test} was expected to fail but passed.")
        except subprocess.CalledProcessError:
            logger.info(f"Test {fail_to_pass_test} failed as expected.")
        logger.info(f"Test {fail_to_pass_test} passed as expected.")

    for pass_to_pass_test in dataset["PASS_TO_PASS"]:
        try:
            subprocess.run(
                [sys.executable, str(test_path / pass_to_pass_test)],
                check=True
            )
            logger.error(f"Test {pass_to_pass_test} was expected to fail but passed.")
        except subprocess.CalledProcessError:
            logger.info(f"Test {pass_to_pass_test} failed as expected.")


def run_agent(dataset):
    myBot = WritingBot(
        model="anthropic/claude-opus-4-5",
        folder_to_mount=str(TEST_DIR / dataset['instance_id']),
    )
    myBot.run(
        task=dataset['problem_statement'] + "\n\nHint: " +dataset['hints_text'],
        max_iterations=50,
    )

    verify_fix(dataset, TEST_DIR / dataset['instance_id'])


def test_swe_bench():
    datasets = load_dataset(SWE_BENCH_SUITE, split="test")

    for dataset in datasets:
        logger.info(dataset)
        setup_test_directory(dataset)
        run_agent(dataset)
        return # For demo purposes, we run only one dataset
