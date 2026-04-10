import json
from pprint import pprint
from datasets import load_dataset
import logging
from pathlib import Path
import sys
import subprocess
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from microbots import AgentBoss, CopilotBot

LOG_DIR = Path(__file__).parent.resolve() / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console output (always active)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(LOG_FORMAT)
logger.addHandler(console_handler)

# Track per-instance file handlers so they can be swapped between test cases
_active_file_handlers: list[logging.Handler] = []


def setup_instance_logging(instance_id: str):
    """Create per-instance log directory and swap file handlers."""
    root = logging.getLogger()

    # Remove previous instance file handlers
    for h in _active_file_handlers:
        root.removeHandler(h)
        h.close()
    _active_file_handlers.clear()

    instance_log_dir = LOG_DIR / instance_id
    instance_log_dir.mkdir(parents=True, exist_ok=True)

    # Info log file
    info_handler = logging.FileHandler(instance_log_dir / "info.log")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(LOG_FORMAT)

    # Debug log file
    debug_handler = logging.FileHandler(instance_log_dir / "debug.log")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(LOG_FORMAT)

    root.setLevel(logging.DEBUG)
    root.addHandler(info_handler)
    root.addHandler(debug_handler)
    _active_file_handlers.extend([info_handler, debug_handler])

    logger.info("Logging for instance %s -> %s", instance_id, instance_log_dir)

# Verification method
# `pip install swebench`
# `python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --max_workers 2 --predictions_path predictions.jsonl --run_id minion`

"""
Difficulty Levels:
1-4 hours
15 min - 1 hour
<15 min fix
>4 hours
"""
DIFFICULTY_ENUM = {
    "EASY": "<15 min fix",
    "MEDIUM": "15 min - 1 hour",
    "HARD": "1-4 hours",
    "VERY_HARD": ">4 hours",
}

# SWE_BENCH_SUITE = "SWE-bench/SWE-bench_Lite"
SWE_BENCH_SUITE = "princeton-nlp/SWE-bench_Verified"
# TEST_DIR = Path(__file__).parent.resolve() / "test_dir"
TEST_DIR = Path("/tmp/swe_bench_test_dir")
PREDICTION_PATH = TEST_DIR / "predictions.jsonl"
REPORT_PATH = TEST_DIR / "report.jsonl"
TEST_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = {}


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


def verify_fix():
    logger.info(f"Verifying predictions from {PREDICTION_PATH} using SWE-bench evaluation harness.")
    result = None
    result = subprocess.run(
        [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", SWE_BENCH_SUITE,
            "--max_workers", "2",
            "--predictions_path", str(PREDICTION_PATH),
            "--run_id", "minion",
            "--report_dir", str(REPORT_PATH) # This option is not working
        ],
        capture_output=True,
        text=True
    )
    logger.info(f"Evaluation result: {result}")
    logger.info("Evaluation completed successfully.")


def run_agent(dataset):
    myBot = AgentBoss(
        model="anthropic/claude-opus-4-5",
        folder_to_mount=str(TEST_DIR / dataset['instance_id']),
    )
    myBot.run(
        task=dataset['problem_statement'] + "\n\nHint: " +dataset['hints_text'],
        max_iterations=100,
        timeout_in_seconds=3600*4, # 4 hours
    )


def run_copilot_agent(dataset):
    bot = CopilotBot(
        model="gpt-5.4",
        folder_to_mount=str(TEST_DIR / dataset['instance_id']),
        permission="READ_WRITE",
    )
    try:
        result = bot.run(
            task=dataset['problem_statement'] + "\n\nHint: " + dataset['hints_text'],
            timeout_in_seconds=3600 * 4,  # 4 hours
        )
        if not result.status:
            logger.error(f"CopilotBot failed on {dataset['instance_id']}: {result.error}")
    finally:
        bot.stop()


def generate_prediction(dataset):
    repo_path = TEST_DIR / dataset['instance_id']
    diff_output = subprocess.run(
        ["git", "diff"],
        cwd=str(repo_path),
        capture_output=True,
        text=True
    )
    # if prediction file exist, load json data
    if os.path.exists(PREDICTION_PATH):
        with open(PREDICTION_PATH, "r") as f:
            existing_predictions = [json.loads(line) for line in f.readlines()]
    else:
        existing_predictions = []

    for pred in existing_predictions:
        if pred["instance_id"] == dataset['instance_id']:
            logger.info(f"Prediction for {dataset['instance_id']} already exists. Updating.")
            pred["model_patch"] = diff_output.stdout
            break
    else:
        prediction = {
            "instance_id": dataset['instance_id'],
            "model_name_or_path": "microbots-opus-4-5",
            "model_patch": diff_output.stdout
        }
        existing_predictions.append(prediction)

    with open(PREDICTION_PATH, "w") as f:
        for pred in existing_predictions:
            f.write(json.dumps(pred) + "\n")


selected_dataset = [
    "astropy__astropy-12907", # Medium
    "astropy__astropy-13033", # Medium
    "astropy__astropy-13398", # Hard
    ]
def test_swe_bench():
    datasets = load_dataset(SWE_BENCH_SUITE, split="test")

    for instance in selected_dataset:
        setup_instance_logging(instance)
        dataset = datasets.filter(lambda x: x['instance_id'] == instance)[0]
        logger.info(f"DATASET: {pprint(dataset)}")
        setup_test_directory(dataset)
        run_agent(dataset)
        generate_prediction(dataset)
        # break # For testing purpose. Remove this to run all datasets.

    verify_fix()


def test_swe_bench_copilot():
    datasets = load_dataset(SWE_BENCH_SUITE, split="test")

    for instance in selected_dataset:
        setup_instance_logging(instance)
        dataset = datasets.filter(lambda x: x['instance_id'] == instance)[0]
        logger.info(f"DATASET: {pprint(dataset)}")
        setup_test_directory(dataset)
        run_copilot_agent(dataset)
        generate_prediction(dataset)

    verify_fix()


if __name__ == "__main__":
    test_swe_bench_copilot()