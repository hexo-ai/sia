#!/usr/bin/env python3
"""
Fetch paper data from the OpenReview API and build the paper-review benchmark datasets.

Creates:
  data/public/iclr2024.json    (200 papers: 100 accept + 100 reject)
  data/private/iclr2023.json   (200 papers: 100 accept + 100 reject)
  data/private/iclr2022.json   (200 papers: 100 accept + 100 reject)

Usage:
    python fetch_openreview.py           # run from tasks/paper-review/
    python fetch_openreview.py --task-dir /path/to/tasks/paper-review
"""

import argparse
import json
import random
import time
from pathlib import Path

import requests

PAGE_SIZE = 1000
REQUEST_DELAY = 0.3  # seconds between pages


def _get(api_base: str, path: str, params: dict) -> dict:
    url = f"{api_base}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_all_submissions(invitation: str, api_version: int = 2) -> list[dict]:
    """Fetch every submission note for a venue."""
    api_base = "https://api2.openreview.net" if api_version == 2 else "https://api.openreview.net"
    notes = []
    offset = 0
    while True:
        data = _get(api_base, "/notes", {"invitation": invitation, "limit": PAGE_SIZE, "offset": offset})
        batch = data.get("notes", [])
        if not batch:
            break
        notes.extend(batch)
        print(f"    fetched {len(notes)} submissions...", end="\r", flush=True)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY)
    print()
    return notes


def _field(content: dict, key: str) -> str:
    val = content.get(key, "")
    if isinstance(val, dict):
        return val.get("value", "") or ""
    return val or ""


def parse_note(note: dict, year: int) -> dict | None:
    content = note.get("content", {})
    title    = _field(content, "title").strip()
    abstract = _field(content, "abstract").strip()
    venue    = _field(content, "venue").lower()

    if not title or not abstract or len(abstract) < 100:
        return None

    # Accept: explicit venue label (poster / spotlight / oral / notable)
    if any(x in venue for x in ("poster", "spotlight", "oral", "notable", "accept")):
        label = "accept"
    # Reject: explicitly rejected or withdrawn
    elif any(x in venue for x in ("reject", "withdrawn", "withdraw")):
        label = "reject"
    # ICLR 2022/2023 pattern: "Submitted to ICLR {year}" or "ICLR {year} Submitted" = not accepted
    elif f"submitted to iclr {year}" in venue or f"iclr {year} submitted" in venue:
        label = "reject"
    else:
        return None

    return {"title": title, "abstract": abstract, "label": label}


def make_balanced_dataset(notes: list[dict], n_each: int = 100, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    accepts = [n for n in notes if n["label"] == "accept"]
    rejects = [n for n in notes if n["label"] == "reject"]
    available = min(len(accepts), len(rejects), n_each)
    if available < n_each:
        print(f"    Warning: only {available} per class available (wanted {n_each})")
    sampled = rng.sample(accepts, available) + rng.sample(rejects, available)
    rng.shuffle(sampled)
    return sampled


VENUES = [
    # (year, invitation, api_version, relative_output_path, description)
    (2024, "ICLR.cc/2024/Conference/-/Submission",       2, "data/public/iclr2024.json",  "ICLR 2024"),
    (2023, "ICLR.cc/2023/Conference/-/Blind_Submission", 1, "data/private/iclr2023.json", "ICLR 2023"),
    (2022, "ICLR.cc/2022/Conference/-/Blind_Submission", 1, "data/private/iclr2022.json", "ICLR 2022"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-dir",
        default=None,
        help="Path to tasks/paper-review/ (default: directory of this script)",
    )
    args = parser.parse_args()

    task_dir = Path(args.task_dir).resolve() if args.task_dir else Path(__file__).resolve().parent

    for year, invitation, api_v, rel_path, description in VENUES:
        out_path = task_dir / rel_path
        if out_path.exists():
            print(f"[{description}] Already exists at {out_path} — skipping.")
            continue

        print(f"\n[{description}] Fetching from OpenReview (API v{api_v})...")
        try:
            raw_notes = fetch_all_submissions(invitation, api_version=api_v)
        except requests.HTTPError as e:
            print(f"  HTTP error: {e} — skipping {description}")
            continue

        print(f"  Raw submissions: {len(raw_notes)}")

        parsed = [p for note in raw_notes if (p := parse_note(note, year)) is not None]
        n_acc = sum(1 for p in parsed if p["label"] == "accept")
        n_rej = sum(1 for p in parsed if p["label"] == "reject")
        print(f"  Parsed: {len(parsed)} ({n_acc} accept, {n_rej} reject)")

        if n_acc < 10 or n_rej < 10:
            print(f"  Skipping — too few labeled papers.")
            continue

        dataset = make_balanced_dataset(parsed, n_each=100)
        print(f"  Final dataset: {len(dataset)} papers (balanced)")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
