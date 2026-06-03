"""
End-to-end pipeline for chess rating prediction.

Usage
-----
# Full run (all games in the PGN)
python -m app.run_pipeline

# Quick test with only 10 000 games
python -m app.run_pipeline --max-games 10000

# Skip steps that already produced output on disk
python -m app.run_pipeline --skip-parse --skip-graph

Flags
-----
--max-games INT   Cap the number of games parsed (default: no limit)
--skip-parse      Reuse data/interim/games.parquet if it exists
--skip-graph      Reuse data/interim/player_graph.pkl if it exists
--skip-features   Reuse data/processed/features.parquet if it exists
"""

from __future__ import annotations

import argparse
import sys

from app.config import INTERIM_DIR, PROCESSED_DIR
from app.utils import get_logger

logger = get_logger("pipeline")


def run(
    max_games: int | None = None,
    skip_parse: bool = False,
    skip_graph: bool = False,
    skip_features: bool = False,
) -> dict:
    # ------------------------------------------------------------------ #
    # Step 1 -- Parse PGN                                                   #
    # ------------------------------------------------------------------ #
    games_parquet = INTERIM_DIR / "games.parquet"
    if skip_parse and games_parquet.exists():
        logger.info(f"[Step 1] Skipping parse -- using {games_parquet}")
        import pandas as pd
        games_df = pd.read_parquet(games_parquet)
    else:
        logger.info("[Step 1] Parsing PGN ...")
        from app.parse_pgn import parse_pgn
        games_df = parse_pgn(max_games=max_games)

    # ------------------------------------------------------------------ #
    # Step 2 -- Build graph                                                 #
    # ------------------------------------------------------------------ #
    graph_pkl = INTERIM_DIR / "player_graph.pkl"
    if skip_graph and graph_pkl.exists():
        logger.info(f"[Step 2] Skipping graph build -- using {graph_pkl}")
        import pickle
        with open(graph_pkl, "rb") as fh:
            G = pickle.load(fh)
    else:
        logger.info("[Step 2] Building player graph ...")
        from app.build_graph import build_graph
        G = build_graph(games_df)

    # ------------------------------------------------------------------ #
    # Step 3 -- Feature extraction                                          #
    # ------------------------------------------------------------------ #
    features_parquet = PROCESSED_DIR / "features.parquet"
    if skip_features and features_parquet.exists():
        logger.info(f"[Step 3] Skipping feature extraction -- using {features_parquet}")
        import pandas as pd
        features_df = pd.read_parquet(features_parquet)
    else:
        logger.info("[Step 3] Extracting features ...")
        from app.features import extract_features
        features_df = extract_features(G)

    # ------------------------------------------------------------------ #
    # Step 4 -- Train model                                                 #
    # ------------------------------------------------------------------ #
    logger.info("[Step 4] Training XGBoost model ...")
    from app.train import train
    model, metrics = train(features_df)

    logger.info("=" * 50)
    logger.info("Pipeline complete.")
    logger.info(f"  MAE  : {metrics['mae']:.2f} Elo points")
    logger.info(f"  RMSE : {metrics['rmse']:.2f} Elo points")
    logger.info(f"  R²   : {metrics['r2']:.4f}")
    logger.info("=" * 50)

    return metrics


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Chess rating prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--max-games",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N valid games (omit for all games)",
    )
    p.add_argument(
        "--skip-parse",
        action="store_true",
        help="Reuse existing games.parquet if available",
    )
    p.add_argument(
        "--skip-graph",
        action="store_true",
        help="Reuse existing player_graph.pkl if available",
    )
    p.add_argument(
        "--skip-features",
        action="store_true",
        help="Reuse existing features.parquet if available",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(
        max_games=args.max_games,
        skip_parse=args.skip_parse,
        skip_graph=args.skip_graph,
        skip_features=args.skip_features,
    )
