from __future__ import annotations

import argparse
from pprint import pprint

from flexeval.run import run_evaluation
from flexeval.schemas.config import EngineConfig, RunConfig, SamplingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic FlexEval run entrypoint.")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", default="transformers")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--capture")
    parser.add_argument("--analyses")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = RunConfig(
        backend=args.backend,
        dataset=args.dataset,
        model=args.model,
        sampling=SamplingConfig(num_samples=args.num_samples),
        engine=EngineConfig(engine=args.engine, device=args.device),
        capture_targets=tuple(filter(None, (args.capture or "").split(","))),
        analysis_targets=tuple(filter(None, (args.analyses or "").split(","))),
    )
    outputs = run_evaluation(config)
    pprint(
        {
            "backend": config.backend,
            "engine": config.engine.engine,
            "dataset": config.dataset,
            "model": config.model,
            "predictions": len(outputs.predictions),
            "scores": len(outputs.scores),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
