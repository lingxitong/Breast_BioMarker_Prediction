"""Re-run multi-center external evaluation using the five saved best weights."""

import argparse

from biomarkers.external import evaluate_external


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--external", action="append", required=True, metavar="CENTER=CSV")
    parser.add_argument("--group-col", default="patient_id")
    parser.add_argument("--feature-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--expected-folds", type=int, default=5,
                        help="Default 5; override only for an explicitly different K-fold experiment")
    args = parser.parse_args()
    evaluate_external(args.run_dir, args.external, args.group_col, args.feature_root,
                      args.device, args.batch_size, args.num_workers, args.expected_folds)


if __name__ == "__main__":
    main()
