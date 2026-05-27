#!/usr/bin/env python3
"""
Script to prepare a task dataset from MLE-Bench competitions.

This script:
1. Runs mlebench prepare command on a competition
2. Copies public and private data to tasks/competition-id/data/
3. Renames description.md to task.md in data/public/
4. Generates similar tasks using Gemini API (optional)
5. Creates SAMPLE_TASK_DESCRIPTIONS.md in reference/
6. Copies reference_target_agent.py from _shared/ to reference/
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

_ = load_dotenv()


def run_mlebench_prepare(competition_id: str) -> bool:
    """Run mlebench prepare command for the given competition."""
    print(f"[1/6] Running mlebench prepare for competition: {competition_id}")
    try:
        result = subprocess.run(
            ["mlebench", "prepare", "-c", competition_id], capture_output=True, text=True, check=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running mlebench prepare: {e}", file=sys.stderr)
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        return False


def copy_dataset_files(competition_id: str, tasks_dir: Path) -> bool:
    """Copy public and private data from mlebench cache to tasks directory."""
    print(f"[2/6] Copying dataset files for {competition_id}")

    # Source directory (mlebench cache) - data is in prepared/ subdirectory
    cache_dir = Path.home() / ".cache" / "mle-bench" / "data" / competition_id
    prepared_dir = cache_dir / "prepared"

    # Destination directory
    task_dir = tasks_dir / competition_id
    data_dir = task_dir / "data"

    # Create destination directories
    data_dir.mkdir(parents=True, exist_ok=True)

    if not prepared_dir.exists():
        print(f"Error: Prepared directory not found: {prepared_dir}", file=sys.stderr)
        return False

    # Copy public directory if exists
    public_src = prepared_dir / "public"
    if public_src.exists() and any(public_src.iterdir()):
        public_dst = data_dir / "public"
        if public_dst.exists():
            shutil.rmtree(public_dst)
        shutil.copytree(public_src, public_dst)
        print(f"  ✓ Copied public data to {public_dst}")
    else:
        print(f"  ⚠ No public directory found or empty at {public_src}")

    # Copy private directory if exists
    private_src = prepared_dir / "private"
    if private_src.exists() and any(private_src.iterdir()):
        private_dst = data_dir / "private"
        if private_dst.exists():
            shutil.rmtree(private_dst)
        shutil.copytree(private_src, private_dst)
        print(f"  ✓ Copied private data to {private_dst}")
    else:
        print(f"  ⚠ No private directory found or empty at {private_src}")

    return True


def move_description_to_task(competition_id: str, tasks_dir: Path) -> bool:
    """Rename description.md to task.md in data/public."""
    print("[3/6] Renaming description.md to task.md")

    task_dir = tasks_dir / competition_id
    data_dir = task_dir / "data"
    public_dir = data_dir / "public"

    # Look for description.md in data/public (already copied from prepared/public)
    description_file = public_dir / "description.md"

    if description_file.exists():
        # Rename description.md to task.md
        task_md_public = public_dir / "task.md"
        description_file.rename(task_md_public)
        print(f"  ✓ Renamed description.md to {task_md_public}")

        return True
    else:
        print(f"  ⚠ No description.md found at {description_file}")
        return False


def get_similar_tasks_from_gemini(competition_id: str, task_description: str) -> str:
    """Use Gemini to generate similar task descriptions."""
    print("[4/6] Generating similar tasks using Gemini API")

    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  ⚠ No GEMINI_API_KEY environment variable set. Skipping similar tasks generation.")
        return ""

    genai.configure(api_key=api_key)

    try:
        # Use Gemini 2.0 Flash Thinking (closest to "Gemini 3 Pro Preview")
        model = genai.GenerativeModel("gemini-3-flash-preview")

        prompt = f"""Given the following Kaggle competition task:

Competition ID: {competition_id}

Task Description:
{task_description}

Generate 3-5 DIVERSE machine learning task descriptions that cover different problem types, domains, and data modalities.

IMPORTANT: Create diversity across:
- Problem types: Classification, regression, clustering, forecasting, generation, etc.
- Data types: Tabular, text, images, time-series, graphs, etc.
- Domains: Healthcare, finance, retail, transportation, social media, etc.

FORMATTING REQUIREMENTS:
1. Match the same level of detail, structure, and writing style as the original task description
2. Each task should be a complete, standalone problem statement
3. Include all relevant sections: overview, dataset description, evaluation metrics, submission format, etc.
4. Separate each task with exactly 5 dashes: -----

EXAMPLE FORMAT:

## Task 1: [Title]

[Complete problem description matching the style and detail level of the original task]

-----

## Task 2: [Title]

[Complete problem description matching the style and detail level of the original task]

-----

[Continue for remaining tasks]

Generate tasks that will help train a generalizable AI agent capable of handling diverse machine learning problems."""

        response = model.generate_content(prompt)
        print("  ✓ Generated similar tasks from Gemini")
        return response.text

    except Exception as e:
        print(f"  ⚠ Error calling Gemini API: {e}", file=sys.stderr)
        return ""


def create_sample_task_descriptions(competition_id: str, tasks_dir: Path, similar_tasks: str) -> bool:
    """Create SAMPLE_TASK_DESCRIPTIONS.md in reference directory."""
    print("[5/6] Creating SAMPLE_TASK_DESCRIPTIONS.md")

    task_dir = tasks_dir / competition_id
    reference_dir = task_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    sample_file = reference_dir / "SAMPLE_TASK_DESCRIPTIONS.md"

    content = f"""
{similar_tasks if similar_tasks else "No similar tasks generated."}
"""

    sample_file.write_text(content)
    print(f"  ✓ Created {sample_file}")
    return True


def copy_reference_agent(competition_id: str, tasks_dir: Path) -> bool:
    """Copy reference_target_agent.py from _shared to competition reference directory."""
    print("[6/6] Copying reference_target_agent.py")

    shared_dir = tasks_dir / "_shared"
    reference_file = shared_dir / "reference_target_agent.py"

    if not reference_file.exists():
        print(f"  ⚠ Reference agent not found at {reference_file}")
        return False

    task_dir = tasks_dir / competition_id
    reference_dir = task_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    dest_file = reference_dir / "reference_target_agent.py"

    shutil.copy2(reference_file, dest_file)
    print(f"  ✓ Copied reference agent to {dest_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare task dataset from MLE-Bench competition")
    parser.add_argument("-c", "--competition", required=True, help="Competition ID (e.g., 'spaceship-titanic')")
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path("./sia/tasks"),
        help="Base tasks directory (default: ./sia/tasks)",
    )
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Gemini API call for similar tasks")

    args = parser.parse_args()

    competition_id = args.competition
    tasks_dir = args.tasks_dir.resolve()

    print(f"\n{'=' * 60}")
    print(f"Preparing SIA Dataset for: {competition_id}")
    print(f"Tasks directory: {tasks_dir}")
    print(f"{'=' * 60}\n")

    # Step 1: Run mlebench prepare
    if not run_mlebench_prepare(competition_id):
        print("\n❌ Failed to prepare dataset with mlebench")
        return 1

    # Step 2: Copy dataset files
    if not copy_dataset_files(competition_id, tasks_dir):
        print("\n❌ Failed to copy dataset files")
        return 1

    # Step 3: Move and rename description
    move_description_to_task(competition_id, tasks_dir)

    # Step 4 & 5: Get similar tasks from Gemini and create sample descriptions
    similar_tasks = ""
    if not args.skip_gemini:
        task_md = tasks_dir / competition_id / "data" / "public" / "task.md"
        task_description = ""
        if task_md.exists():
            task_description = task_md.read_text()

        if task_description:
            similar_tasks = get_similar_tasks_from_gemini(competition_id, task_description)
        else:
            print("  ⚠ No task description found for Gemini API call")

    create_sample_task_descriptions(competition_id, tasks_dir, similar_tasks)

    # Step 6: Copy reference agent
    copy_reference_agent(competition_id, tasks_dir)

    print(f"\n{'=' * 60}")
    print("✅ Dataset preparation complete!")
    print(f"{'=' * 60}")
    print(f"\nTask directory: {tasks_dir / competition_id}")
    print("  - data/public/                      : Public dataset")
    print("      - task.md                       : Task description")
    print("      - *.csv                         : Data files")
    print("  - data/private/                     : Private dataset")
    print("  - reference/SAMPLE_TASK_DESCRIPTIONS.md  : Similar tasks from Gemini")
    print("  - reference/reference_target_agent.py    : Reference agent template")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
