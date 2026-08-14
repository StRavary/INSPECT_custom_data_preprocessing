"""
_route_b_worker.py
------------------
Subprocess worker for Route B lab extraction. Spawned by the Streamlit app to
run the DuckDB extraction in a clean process (same pattern as _femr_worker.py).

Usage (called by the app, not directly):
    python _route_b_worker.py <spec_json_path> <output_pkl_path>
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: _route_b_worker.py <spec_json> <output_pkl>", flush=True)
        sys.exit(1)

    spec_path = Path(sys.argv[1])
    out_path  = Path(sys.argv[2])

    # Ensure the repo root is on the path
    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    spec = json.loads(spec_path.read_text())
    pth  = spec.get("paths", {})

    feature_types         = spec.get("feature_types", ["labs"])
    # count_window_days is a dict {feature_type: days} e.g. {"diagnoses": 365, "drugs": 60}
    count_window_days     = spec.get("count_window_days", {})
    use_concept_ancestor  = spec.get("use_concept_ancestor", False)
    ancestor_levels       = spec.get("ancestor_levels", None)
    min_studies_ancestor  = spec.get("min_studies_ancestor", 100)
    max_ancestor_features = spec.get("max_ancestor_features", 2000)
    one_study_per_patient = spec.get("one_study_per_patient") or None
    admission_anchored    = spec.get("admission_anchored", False)
    pre_admission_days    = spec.get("pre_admission_days") or None

    print(f"[route_b_worker] task={spec['task']}  anchor={spec.get('anchor', 'auto')}",
          flush=True)
    print(f"[route_b_worker] cohort              : {pth.get('cohort')}",          flush=True)
    print(f"[route_b_worker] measurement         : {pth.get('measurement')}",      flush=True)
    print(f"[route_b_worker] concept             : {pth.get('concept')}",          flush=True)
    print(f"[route_b_worker] feature_types       : {feature_types}",               flush=True)
    print(f"[route_b_worker] windows_days        : {spec.get('windows_days')}",    flush=True)
    print(f"[route_b_worker] count_window        : {count_window_days}",            flush=True)
    print(f"[route_b_worker] loinc_codes         : {spec.get('loinc_codes') or 'all LOINC'}",
          flush=True)
    print(f"[route_b_worker] min_studies         : {spec.get('min_studies_per_lab', 50)}",
          flush=True)
    print(f"[route_b_worker] use_concept_ancestor: {use_concept_ancestor}",        flush=True)
    print(f"[route_b_worker] one_study_per_patient: {one_study_per_patient or 'all'}", flush=True)
    print(f"[route_b_worker] admission_anchored  : {admission_anchored}", flush=True)
    if admission_anchored and pre_admission_days:
        print(f"[route_b_worker] pre_admission_days  : {pre_admission_days}", flush=True)
    if use_concept_ancestor:
        print(f"[route_b_worker] ancestor_levels     : {ancestor_levels}",          flush=True)
        print(f"[route_b_worker] min_studies_ancestor: {min_studies_ancestor}",     flush=True)
        print(f"[route_b_worker] max_ancestor_feats  : {max_ancestor_features}",    flush=True)

    from Custom.appd_route_b_labs import LabExtractor

    ex = LabExtractor(
        cohort=pth.get("cohort"),
        measurement_csv=pth.get("measurement"),
        concept_csv=pth.get("concept"),
        labels=pth.get("labels"),
        person=pth.get("person"),
        verbose=True,
    )

    fm = ex.build(
        task=spec["task"],
        windows_days=spec.get("windows_days", [2, 7, 30, 365]),
        loinc_codes=spec.get("loinc_codes") or None,
        anchor=None if spec.get("anchor") in ("auto", None) else spec["anchor"],
        min_studies_per_lab=spec.get("min_studies_per_lab", 50),
        feature_types=feature_types,
        count_window_days=count_window_days,
        use_concept_ancestor=use_concept_ancestor,
        ancestor_levels=ancestor_levels,
        min_studies_ancestor=min_studies_ancestor,
        max_ancestor_features=max_ancestor_features,
        one_study_per_patient=one_study_per_patient,
        admission_anchored=admission_anchored,
        pre_admission_days=pre_admission_days,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(fm, f)

    size_mb = out_path.stat().st_size / 1e6
    print(f"[route_b_worker] saved → {out_path}  ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
