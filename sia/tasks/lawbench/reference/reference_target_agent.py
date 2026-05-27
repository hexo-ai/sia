#!/usr/bin/env python3
"""
Reference Target Agent for LawBench Criminal Charge Prediction

Uses GPT-OSS-120B via Tinker API to predict criminal charges from Chinese legal cases.

Usage:
    python reference_target_agent.py --dataset_dir /path/to/dataset --working_dir /path/to/working
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Tinker configuration (hardcoded)
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
TINKER_MODEL = "openai/gpt-oss-120b"


def extract_charge(response_text: str, valid_classes: list) -> str:
    """
    Extract the charge prediction from LLM response.
    Tries to match against valid classes.
    """
    # Clean the response
    text = response_text.strip()

    # Try to find exact match in valid classes
    for valid_class in valid_classes:
        if valid_class in text:
            return valid_class

    # If no exact match, return the first few characters (likely the charge)
    # Remove common prefixes/suffixes
    text = text.replace('罪名：', '').replace('[罪名]', '').replace('罪', '').strip()

    # Try again with cleaned text
    for valid_class in valid_classes:
        if valid_class.replace('罪', '').strip() in text:
            return valid_class

    # Fallback: return first valid class (better than crashing)
    logger.warning(f"Could not extract valid charge from: {text[:100]}")
    return valid_classes[0] if valid_classes else "盗窃"


def main():
    parser = argparse.ArgumentParser(
        description='Reference agent for LawBench charge prediction using GPT-OSS-120B'
    )
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset directory (READ-ONLY)')
    parser.add_argument('--working_dir', required=True, help='Path to working directory (READ-WRITE)')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    # Get API key from environment
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        logger.error("❌ TINKER_API_KEY environment variable not set")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("LawBench Criminal Charge Prediction - GPT-OSS-120B Agent")
    logger.info("=" * 70)
    logger.info(f"Model: {TINKER_MODEL}")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Working directory: {working_dir}")

    # Initialize OpenAI client with Tinker
    client = OpenAI(
        api_key=api_key,
        base_url=TINKER_BASE_URL,
    )

    # Track execution
    execution_log = []
    start_time = datetime.now()

    try:
        # Step 1: Load test data
        logger.info("\n[Step 1] Loading test data...")
        test_path = dataset_dir / "test.csv"
        test_df = pd.read_csv(test_path)
        logger.info(f"  ✓ Loaded {len(test_df)} test samples")

        execution_log.append({
            "step": "load_test",
            "status": "success",
            "samples": len(test_df)
        })

        # Step 2: Load valid classes
        logger.info("\n[Step 2] Loading valid classes...")
        classes_path = dataset_dir / "classes.json"
        with open(classes_path, 'r', encoding='utf-8') as f:
            valid_classes = json.load(f)
        logger.info(f"  ✓ Loaded {len(valid_classes)} valid classes")

        execution_log.append({
            "step": "load_classes",
            "status": "success",
            "num_classes": len(valid_classes)
        })

        # Step 3: Load training data for few-shot examples (optional)
        logger.info("\n[Step 3] Loading training data for few-shot examples...")
        train_path = dataset_dir / "train.csv"
        train_df = pd.read_csv(train_path)
        sample_charges = train_df['label'].unique().tolist()[:10]
        logger.info(f"  ✓ Using {len(sample_charges)} sample charges for prompting")

        execution_log.append({
            "step": "load_train_samples",
            "status": "success",
            "sample_charges": len(sample_charges)
        })

        # Step 4: Generate predictions with multi-trajectory logging
        logger.info("\n[Step 4] Generating predictions using GPT-OSS-120B...")
        logger.info("  → Processing test cases...")

        # Create agent_execution directory for multi-trajectory format
        execution_dir = working_dir / "agent_execution"
        execution_dir.mkdir(exist_ok=True)

        predictions = []
        failed_count = 0
        trajectories = []

        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            test_id = row['id']
            case_text = row['text']

            # Build prompt for this case
            examples = ""
            if sample_charges:
                examples = "\n示例：\n" + "\n".join([
                    f"- {charge}" for charge in sample_charges[:10]
                ]) + "\n"

            prompt = f"""你是一位法律专家。根据以下案件事实，预测被告人被判定的罪名。

{examples}
案件事实：
{case_text}

请从以下191个罪名中选择一个最合适的罪名：
{", ".join(valid_classes[:20])}... (共191个罪名)

只回答罪名，不要其他解释。格式：罪名"""

            trajectory = []

            try:
                # Record the request
                trajectory.append({
                    "role": "user",
                    "content": prompt,
                    "timestamp": datetime.now().isoformat()
                })

                # Make API call
                response = client.chat.completions.create(
                    model=TINKER_MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50,
                    temperature=0.1,
                )

                prediction = response.choices[0].message.content.strip()
                charge = extract_charge(prediction, valid_classes)

                # Record the response
                trajectory.append({
                    "role": "assistant",
                    "content": prediction,
                    "extracted_charge": charge,
                    "timestamp": datetime.now().isoformat()
                })

                predictions.append(charge)

            except Exception as e:
                logger.warning(f"  ⚠ Failed on sample {idx}: {e}")
                predictions.append(valid_classes[0])  # Fallback
                failed_count += 1

                # Record the error
                trajectory.append({
                    "role": "error",
                    "error": str(e),
                    "fallback_charge": valid_classes[0],
                    "timestamp": datetime.now().isoformat()
                })

            # Save trajectory for this test case
            trajectory_file = execution_dir / f"execution_q{idx}.json"
            with open(trajectory_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory, f, ensure_ascii=False, indent=2)

            trajectories.append({
                "test_id": test_id,
                "trajectory_file": f"execution_q{idx}.json",
                "prediction": predictions[-1]
            })

        logger.info(f"  ✓ Generated {len(predictions)} predictions")
        logger.info(f"  ✓ Saved {len(trajectories)} trajectory files to {execution_dir}")
        if failed_count > 0:
            logger.warning(f"  ⚠ {failed_count} predictions failed (used fallback)")

        execution_log.append({
            "step": "predict",
            "status": "success",
            "num_predictions": len(predictions),
            "failed_count": failed_count,
            "trajectories_saved": len(trajectories)
        })

        # Step 5: Save submission
        logger.info("\n[Step 5] Saving submission file...")
        submission_df = pd.DataFrame({
            'id': test_df['id'].values,
            'label': predictions
        })

        submission_path = working_dir / "submission.csv"
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"  ✓ Submission saved to: {submission_path}")

        # Verify submission format
        assert len(submission_df) == len(test_df), "Submission must have all test IDs"
        assert list(submission_df.columns) == ['id', 'label'], "Submission must have columns: id, label"
        logger.info("  ✓ Submission format validated")

        execution_log.append({
            "step": "save_submission",
            "status": "success",
            "path": str(submission_path),
            "rows": len(submission_df)
        })

        # Step 6: Save summary execution log
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        summary_log = {
            "agent": "reference_target_agent_gpt_oss",
            "model": TINKER_MODEL,
            "task": "lawbench_charge_prediction",
            "execution_format": "multi-trajectory",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "status": "success",
            "total_trajectories": len(trajectories),
            "steps": execution_log,
            "trajectories_location": str(execution_dir)
        }

        summary_path = working_dir / "execution_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_log, f, ensure_ascii=False, indent=2)
        logger.info(f"\n  ✓ Execution summary saved to: {summary_path}")
        logger.info(f"  ✓ Individual trajectories saved to: {execution_dir}")

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("✅ Agent completed successfully!")
        logger.info("=" * 70)
        logger.info(f"Test samples: {len(test_df)}")
        logger.info(f"Predictions generated: {len(predictions)}")
        logger.info(f"Failed predictions: {failed_count}")
        logger.info(f"Execution time: {duration:.2f}s")
        logger.info(f"Submission: {submission_path}")
        logger.info("=" * 70)

        sys.exit(0)

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        # Save error log
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        error_log = {
            "agent": "reference_target_agent_gpt_oss",
            "model": TINKER_MODEL,
            "task": "lawbench_charge_prediction",
            "execution_format": "multi-trajectory",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "status": "failed",
            "error": str(e),
            "steps": execution_log
        }

        error_path = working_dir / "execution_summary.json"
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_log, f, ensure_ascii=False, indent=2)

        sys.exit(1)


if __name__ == "__main__":
    main()
