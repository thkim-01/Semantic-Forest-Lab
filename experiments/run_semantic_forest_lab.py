# flake8: noqa

"""Run Semantic-Forest-lab with algorithm profiles (ID3/C4.5/CART).

Examples:
  python experiments/run_semantic_forest_lab.py --algorithm id3
  python experiments/run_semantic_forest_lab.py --algorithm c45 --datasets bbbp,clintox
  python experiments/run_semantic_forest_lab.py --algorithm cart --all-tasks
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # Insert at front so local `src` wins over any installed package named `src`.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms import load_algorithm_profile


def build_command(args, profile: dict):
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    key = profile["algorithm_key"]
    out_csv = out_base / f"semantic_forest_benchmark_{key}.csv"

    cmd = [
        sys.executable,
        "experiments/verify_semantic_forest_multi.py",
        "--split-criterion",
        str(profile["split_criterion"]),
        "--n-estimators",
        str(args.n_estimators if args.n_estimators is not None else profile.get("n_estimators", 5)),
        "--max-depth",
        str(args.max_depth if args.max_depth is not None else profile.get("max_depth", 10)),
        "--min-samples-split",
        str(args.min_samples_split if args.min_samples_split is not None else profile.get("min_samples_split", 20)),
        "--min-samples-leaf",
        str(args.min_samples_leaf if args.min_samples_leaf is not None else profile.get("min_samples_leaf", 5)),
        "--sample-size",
        str(args.sample_size),
        "--test-size",
        str(args.test_size),
        "--random-state",
        str(args.random_state),
        "--compute-backend",
        str(args.compute_backend),
        "--torch-device",
        str(args.torch_device),
        "--out",
        str(out_csv),
    ]

    if args.datasets:
        cmd.extend(["--datasets", args.datasets])
    if args.all_tasks:
        cmd.append("--all-tasks")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.feature_cache_dir is not None:
        cmd.extend(["--feature-cache-dir", args.feature_cache_dir])

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Semantic-Forest-lab algorithm runner"
    )
    parser.add_argument(
        "--algorithm",
        default="id3",
        choices=["id3", "c45", "cart"],
        help="Algorithm profile to run.",
    )
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--feature-cache-dir", default=str(Path("output") / "feature_cache"))
    parser.add_argument("--out-dir", default=str(Path("output") / "lab_runs"))
    parser.add_argument("--compute-backend", default="auto", choices=["auto", "numpy", "torch"])
    parser.add_argument("--torch-device", default="auto")

    # Optional overrides (profile defaults if omitted)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=None)

    args = parser.parse_args()

    profile = load_algorithm_profile(args.algorithm)

    print(
        f"[Lab] Algorithm={profile['name']} "
        f"split_criterion={profile['split_criterion']}"
    )
    cmd = build_command(args, profile)
    print(f"[Lab] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
