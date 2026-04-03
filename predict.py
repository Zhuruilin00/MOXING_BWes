from __future__ import annotations

import argparse

from src.star_predictor.pipeline import StarMarketDirectionPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成科创板未来 3-5 天方向预测")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = StarMarketDirectionPredictor()
    result = predictor.predict_latest(model_path=args.model_path, dataset_path=args.dataset_path, top_k=args.top_k)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
