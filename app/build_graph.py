"""
build_graph.py -- Build a weighted directed player-interaction graph.

Graph semantics
---------------
* Node  = chess player (username string)
* Edge  = winner_player  ->  loser_player
* weight (edge attribute) = number of times the source player beat the
  target player across all games in the dataset

Node attributes (computed from the filtered games DataFrame)
------------------------------------------------------------
total_games    total decisive games played (wins + losses)
total_wins     games won
total_losses   games lost
win_rate       total_wins / total_games
avg_elo        mean Elo across all games played as either colour
avg_opp_elo    mean Elo of opponents faced

Outputs
-------
* networkx.DiGraph          (returned + saved as data/interim/player_graph.pkl)
* edge list DataFrame       columns: source, target, weight
                            saved as data/processed/graph_edges.parquet

Scalability notes
-----------------
All aggregations use vectorised pandas groupby + value_counts -- no
Python-level row iteration.  nx.from_pandas_edgelist() constructs the
graph in a single pass over the edge list.  Memory usage is O(E) for
the edge table and O(V) for node attribute tables.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

from app.config import INTERIM_DIR, MIN_GAMES, PROCESSED_DIR
from app.utils import ensure_dirs, get_logger, timer

logger = get_logger(__name__)



def _build_edge_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate (winner_player, loser_player) pairs into a weighted edge list.

    Uses groupby + size() -- no Python loops.

    Returns
    -------
    DataFrame with columns: source, target, weight
      sorted by weight descending.
    """
    edges = (
        df.groupby(["winner_player", "loser_player"], sort=False)
        .size()
        .reset_index(name="weight")
        .rename(columns={"winner_player": "source", "loser_player": "target"})
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    return edges


def _build_node_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-player statistics entirely with vectorised pandas operations.

    Returns
    -------
    DataFrame indexed by player name with columns:
      total_wins, total_losses, total_games, win_rate,
      avg_elo, avg_opp_elo
    """
    # wins / losses counts
    wins   = df.groupby("winner_player").size().rename("total_wins")
    losses = df.groupby("loser_player").size().rename("total_losses")

    node_stats = (
        pd.concat([wins, losses], axis=1)
        .fillna(0)
        .astype(int)
    )
    node_stats.index.name = "player"
    node_stats["total_games"] = node_stats["total_wins"] + node_stats["total_losses"]
    node_stats["win_rate"] = node_stats["total_wins"] / node_stats["total_games"]

    # average Elo across all games (as winner + as loser)
    elo_as_winner = (
        df[["winner_player", "winner_elo"]]
        .rename(columns={"winner_player": "player", "winner_elo": "elo"})
    )
    elo_as_loser = (
        df[["loser_player", "loser_elo"]]
        .rename(columns={"loser_player": "player", "loser_elo": "elo"})
    )
    avg_elo = (
        pd.concat([elo_as_winner, elo_as_loser], ignore_index=True)
        .groupby("player")["elo"]
        .mean()
        .rename("avg_elo")
    )

    # average opponent Elo
    opp_as_winner = (
        df[["winner_player", "loser_elo"]]
        .rename(columns={"winner_player": "player", "loser_elo": "opp_elo"})
    )
    opp_as_loser = (
        df[["loser_player", "winner_elo"]]
        .rename(columns={"loser_player": "player", "winner_elo": "opp_elo"})
    )
    avg_opp_elo = (
        pd.concat([opp_as_winner, opp_as_loser], ignore_index=True)
        .groupby("player")["opp_elo"]
        .mean()
        .rename("avg_opp_elo")
    )

    node_stats = node_stats.join(avg_elo).join(avg_opp_elo)
    return node_stats


def _filter_by_min_games(
    edges: pd.DataFrame,
    node_attrs: pd.DataFrame,
    min_games: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove players (nodes) with fewer than *min_games* total games, and
    drop any edges that reference a removed node.

    Both operations are vectorised -- no Python loops.
    """
    if min_games <= 1:
        return edges, node_attrs

    qualified = node_attrs.index[node_attrs["total_games"] >= min_games]
    qualified_set = set(qualified)

    node_attrs = node_attrs.loc[qualified]
    mask = edges["source"].isin(qualified_set) & edges["target"].isin(qualified_set)
    edges = edges.loc[mask].reset_index(drop=True)

    return edges, node_attrs


# ── public API ────────────────────────────────────────────────────────────────


@timer
def build_graph(
    games_df: Optional[pd.DataFrame] = None,
    min_games: int = MIN_GAMES,
) -> tuple[nx.DiGraph, pd.DataFrame]:
 
    # ── load data ─────────────────────────────────────────────────────────
    if games_df is None:
        src = INTERIM_DIR / "games_filtered.parquet"
        logger.info(f"Loading games from {src}")
        games_df = pd.read_parquet(src)

    logger.info(f"Input: {len(games_df):,} games, "
                f"{games_df['winner_player'].nunique():,} unique winners, "
                f"{games_df['loser_player'].nunique():,} unique losers")

    # ── vectorised aggregations ───────────────────────────────────────────
    logger.info("Aggregating edge weights ...")
    edges = _build_edge_list(games_df)

    logger.info("Computing node attributes ...")
    node_attrs = _build_node_attributes(games_df)

    logger.info(
        f"Before filtering: {len(node_attrs):,} players, "
        f"{len(edges):,} directed edges"
    )

    # ── min-games filter ─────────────────────────────────────────────────
    edges, node_attrs = _filter_by_min_games(edges, node_attrs, min_games)
    logger.info(
        f"After  filtering (min_games={min_games}): "
        f"{len(node_attrs):,} players, {len(edges):,} directed edges"
    )

    # ── build DiGraph ─────────────────────────────────────────────────────
    logger.info("Constructing DiGraph ...")
    G: nx.DiGraph = nx.from_pandas_edgelist(
        edges,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )

    # Ensure every qualified player is a node even if all their edges
    # were pruned (isolated nodes kept for completeness)
    G.add_nodes_from(node_attrs.index.difference(G.nodes()))

    # Attach node attributes in one vectorised call per attribute
    for col in node_attrs.columns:
        nx.set_node_attributes(G, node_attrs[col].to_dict(), name=col)

    logger.info(
        f"DiGraph: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges  "
        f"(density={nx.density(G):.6f})"
    )

    # ── persist ───────────────────────────────────────────────────────────
    ensure_dirs(INTERIM_DIR, PROCESSED_DIR)

    graph_path = INTERIM_DIR / "player_graph.pkl"
    with open(graph_path, "wb") as fh:
        pickle.dump(G, fh)
    logger.info(f"Graph saved   -> {graph_path}")

    edges_path = PROCESSED_DIR / "graph_edges.parquet"
    edges.to_parquet(edges_path, index=False)
    logger.info(f"Edge list saved -> {edges_path}  shape={edges.shape}")

    return G, edges


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build directed player graph.")
    p.add_argument(
        "--min-games",
        type=int,
        default=MIN_GAMES,
        help=f"Minimum games per player (default: {MIN_GAMES})",
    )
    args = p.parse_args()

    G, edges = build_graph(min_games=args.min_games)

    print(f"\nGraph summary")
    print(f"  Nodes  : {G.number_of_nodes():,}")
    print(f"  Edges  : {G.number_of_edges():,}")
    print(f"  Density: {nx.density(G):.6f}")
    print(f"\nTop 10 edges by weight:")
    print(edges.head(10).to_string(index=False))
