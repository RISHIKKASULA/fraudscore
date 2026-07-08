"""fraudscore CLI: train, evaluate, serve, score-batch.

Data fetching is `python scripts/fetch_data.py` (Kaggle CLI + integrity checks);
the flow `fetch -> train -> evaluate` reproduces docs/eval-report.md bit-for-bit.
"""

from __future__ import annotations

import argparse

DEFAULT_DATA = "data/raw/creditcard.csv"
DEFAULT_ARTIFACT_DIR = "artifacts"
DEFAULT_COST_PARAMS = "cost_params.yaml"
DEFAULT_REPORT = "docs/eval-report.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraudscore", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="train, calibrate, freeze t*, save artifact")
    train.add_argument("--data", default=DEFAULT_DATA)
    train.add_argument("--cost-params", default=DEFAULT_COST_PARAMS)
    train.add_argument("--out", default=DEFAULT_ARTIFACT_DIR)

    evaluate = sub.add_parser("evaluate", help="write eval report from a trained artifact")
    evaluate.add_argument("--data", default=DEFAULT_DATA)
    evaluate.add_argument("--cost-params", default=DEFAULT_COST_PARAMS)
    evaluate.add_argument("--artifact", default=f"{DEFAULT_ARTIFACT_DIR}/model.joblib")
    evaluate.add_argument("--report", default=DEFAULT_REPORT)
    evaluate.add_argument("--bootstrap-b", type=int, default=None,
                          help="override bootstrap B (CI uses a reduced value)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        from fraudscore.train import run_training

        artifact = run_training(args.data, args.cost_params, args.out)
        print(f"artifact written: {artifact}")
        return 0

    if args.command == "evaluate":
        from fraudscore.evaluate import run_evaluation

        run_evaluation(args.artifact, args.data, args.cost_params, args.report,
                       bootstrap_b=args.bootstrap_b)
        print(f"report written: {args.report}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
