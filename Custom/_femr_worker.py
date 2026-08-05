"""
_femr_worker.py
---------------
Headless FEMR feature extraction worker. Spawned as a subprocess by
app_feature_extraction.py to sidestep the Streamlit + multiprocessing deadlock:

    multiprocessing.Pool  fork()s from the Streamlit process, which is
    multi-threaded.  Forking a multi-threaded process copies every thread's
    lock state; worker threads that were mid-lock in the parent never release
    and the child deadlocks silently.

Running extraction here (a clean, single-threaded process) lets FEMR's pool
fork safely with any num_threads value.

Usage (called by the app, not directly):
    python _femr_worker.py <spec_json_path> <output_pkl_path> [num_threads]
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: _femr_worker.py <spec_json> <output_pkl> [num_threads]",
              flush=True)
        sys.exit(1)

    spec_path = Path(sys.argv[1])
    out_path  = Path(sys.argv[2])
    num_threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    # Ensure the repo root is on the path (same logic as app_feature_extraction.py)
    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    spec = json.loads(spec_path.read_text())

    # GROUP_TABLES duplicated here to avoid importing Streamlit via the app module
    GROUP_TABLES = {
        "vitals_labs": ["measurement", "observation"],
        "diag_proc":   ["condition_occurrence", "procedure_occurrence", "device_exposure"],
        "drugs":       ["drug_exposure"],
        "visits":      ["visit_occurrence", "visit_detail"],
    }

    from Custom.temporal_features import FeatureGroup, TemporalFeatureExtractor

    groups = [
        FeatureGroup(
            name=n,
            tables=frozenset(GROUP_TABLES[n]),
            bins=g["bins"],
            catch_all=g["catch_all"],
        )
        for n, g in spec["groups"].items()
    ]

    pth = spec.get("paths", {})

    print(f"[worker] task={spec['task']}  anchor={spec['anchor']}", flush=True)
    print(f"[worker] cohort  : {pth.get('cohort')}", flush=True)
    print(f"[worker] database: {pth.get('database')}", flush=True)
    print(f"[worker] num_threads={num_threads}", flush=True)

    fm = TemporalFeatureExtractor(
        cohort=pth.get("cohort"),
        database=pth.get("database"),
        labels=pth.get("labels"),
        num_threads=num_threads,
        verbose=True,          # prints go to the log file the app tails
    ).build(
        task=spec["task"],
        groups=groups,
        anchor=None if spec["anchor"] == "auto" else spec["anchor"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(fm, f)

    size_mb = out_path.stat().st_size / 1e6
    print(f"[worker] saved → {out_path}  ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
