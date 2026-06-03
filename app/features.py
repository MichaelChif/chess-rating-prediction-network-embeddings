"""
features.py -- Compute graph features and Node2Vec embeddings per player,
then merge with per-player Elo statistics to produce the final feature matrix.

Graph features (computed on the directed graph)
------------------------------------------------
pagerank        Weighted PageRank -- captures "who beats strong players"
betweenness     Approximate betweenness centrality (sampled, see BETWEENNESS_K)
k_core          k-core number on the undirected projection of the graph
in_degree_w     Weighted in-degree  (total losses inflicted on this player)
out_degree_w    Weighted out-degree (total wins by this player)
total_degree    Unweighted total degree (in + out edge count)

Node2Vec embeddings
-------------------
n2v_0 ... n2v_{N2V_DIMENSIONS-1}
  Trained on the DiGraph; node IDs are cast to str before fitting.

Player statistics (recomputed from games_filtered.parquet)
----------------------------------------------------------
games_played    Total decisive games (wins + losses)
target_elo      Mean Elo across all games played (the regression target)

Row selection
-------------
Only players that (a) appear as nodes in the graph AND (b) have at least
MIN_FEATURES_GAMES decisive games are kept.

Output
------
data/processed/player_features.parquet
  Columns: player | graph features | n2v embeddings | games_played | target_elo
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec

from app.config import (
    BETWEENNESS_K,
    INTERIM_DIR,
    MIN_FEATURES_GAMES,
    N2V_DIMENSIONS,
    N2V_NUM_WALKS,
    N2V_P,
    N2V_Q,
    N2V_WALK_LENGTH,
    N2V_WORKERS,
    PROCESSED_DIR,
)
from app.utils import ensure_dirs, get_logger, timer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Graph-metric helpers
# ---------------------------------------------------------------------------

def _pagerank(G: nx.DiGraph) -> dict[str, float]:
    """Weighted PageRank on the directed graph."""
    logger.info("  pagerank (weighted) ...")
    return nx.pagerank(G, weight="weight", max_iter=300, tol=1e-6)


def _betweenness(G: nx.DiGraph) -> dict[str, float]:
    """
    Approximate betweenness centrality using BETWEENNESS_K pivot samples.
    Falls back to exact if the graph is smaller than the sample budget.
    """
    n = G.number_of_nodes()
    k = min(BETWEENNESS_K, n) if BETWEENNESS_K else n
    exact = k == n
    logger.info(
        f"  betweenness centrality "
        f"({'exact' if exact else f'sampled k={k}'}) ..."
    )
    return nx.betweenness_centrality(G, k=k, weight="weight", seed=42)


def _k_core(G: nx.DiGraph) -> dict[str, int]:
    """
    k-core number on the *undirected* projection of the graph.

    k-core is defined only for undirected graphs; we project with
    reciprocal=False so any directed edge (u->v or v->u) becomes an
    undirected edge u-v.
    """
    logger.info("  k-core (undirected projection) ...")
    G_und = G.to_undirected(reciprocal=False)
    return nx.core_number(G_und)


def _degree_features(G: nx.DiGraph) -> tuple[
    dict[str, float],   # in_degree_w
    dict[str, float],   # out_degree_w
    dict[str, int],     # total_degree (unweighted)
]:
    """Weighted in/out degree and unweighted total degree."""
    logger.info("  degree features ...")
    in_w  = dict(G.in_degree(weight="weight"))
    out_w = dict(G.out_degree(weight="weight"))
    # Unweighted total = in-edges + out-edges
    total = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}
    return in_w, out_w, total


# ---------------------------------------------------------------------------
# Node2Vec helper
# ---------------------------------------------------------------------------

def _train_node2vec(G: nx.DiGraph) -> object:  # returns gensim KeyedVectors
    """
    Fit Node2Vec on *G* and return trained KeyedVectors.

    Node IDs are explicitly converted to strings before fitting so that
    gensim's vocabulary lookup works consistently regardless of how
    NetworkX stored the original IDs.
    """
    logger.info(
        f"  Node2Vec: dims={N2V_DIMENSIONS}, walks={N2V_NUM_WALKS}, "
        f"length={N2V_WALK_LENGTH}, p={N2V_P}, q={N2V_Q}, "
        f"workers={N2V_WORKERS} ..."
    )

    # Relabel all node IDs to plain strings -- required for gensim's vocab
    G_str = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})

    n2v = Node2Vec(
        G_str,
        dimensions=N2V_DIMENSIONS,
        walk_length=N2V_WALK_LENGTH,
        num_walks=N2V_NUM_WALKS,
        workers=N2V_WORKERS,
        p=N2V_P,
        q=N2V_Q,
        weight_key="weight",
        quiet=False,
    )
    model = n2v.fit(
        window=10,
        min_count=1,
        batch_words=4,
        epochs=5,
        seed=42,
    )
    return model.wv   # KeyedVectors; keys are str(node_id)


# ---------------------------------------------------------------------------
# Target / stats helper
# ---------------------------------------------------------------------------

def _player_stats(games_path: Path) -> pd.DataFrame:
    """
    Compute per-player mean Elo and games_played from *games_filtered.parquet*.

    Returns
    -------
    DataFrame indexed by player with columns: games_played, target_elo
    """
    logger.info(f"  Loading player stats from {games_path} ...")
    df = pd.read_parquet(games_path)

    # Stack winner and loser rows so every appearance counts equally
    winner_rows = df[["winner_player", "winner_elo"]].rename(
        columns={"winner_player": "player", "winner_elo": "elo"}
    )
    loser_rows = df[["loser_player", "loser_elo"]].rename(
        columns={"loser_player": "player", "loser_elo": "elo"}
    )
    all_rows = pd.concat([winner_rows, loser_rows], ignore_index=True)

    stats = all_rows.groupby("player").agg(
        games_played=("elo", "count"),
        target_elo=("elo", "mean"),
    )
    return stats   # indexed by player name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@timer
def extract_features(
    G: Optional[nx.DiGraph] = None,
    games_path: Optional[Path] = None,
    min_games: int = MIN_FEATURES_GAMES,
) -> pd.DataFrame:
    """
    Build the full feature matrix and save it to parquet.

    Parameters
    ----------
    G : nx.DiGraph, optional
        Directed player graph from ``build_graph``.
        Loaded from ``data/interim/player_graph.pkl`` when not provided.
    games_path : Path, optional
        Path to the filtered games parquet.
        Defaults to ``data/interim/games_filtered.parquet``.
    min_games : int
        Minimum decisive games for a player to be included.
        Defaults to ``config.MIN_FEATURES_GAMES`` (5).

    Returns
    -------
    pd.DataFrame
        One row per qualifying player; columns described in the module
        docstring; saved to ``data/processed/player_features.parquet``.
    """
    # ------------------------------------------------------------------ #
    # 1. Load graph                                                        #
    # ------------------------------------------------------------------ #
    if G is None:
        graph_path = INTERIM_DIR / "player_graph.pkl"
        logger.info(f"Loading graph from {graph_path}")
        with open(graph_path, "rb") as fh:
            G = pickle.load(fh)

    if not isinstance(G, nx.DiGraph):
        raise TypeError(
            f"Expected nx.DiGraph, got {type(G).__name__}. "
            "Re-run build_graph to produce a directed graph."
        )

    logger.info(
        f"Graph: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} directed edges"
    )

    # ------------------------------------------------------------------ #
    # 2. Compute graph features                                            #
    # ------------------------------------------------------------------ #
    logger.info("Computing graph features ...")

    pr       = _pagerank(G)
    btwn     = _betweenness(G)
    kcore    = _k_core(G)
    in_w, out_w, total_deg = _degree_features(G)

    # Assemble into a DataFrame indexed by player name
    graph_feat = pd.DataFrame(
        {
            "pagerank":     pd.Series(pr,      dtype="float32"),
            "betweenness":  pd.Series(btwn,    dtype="float32"),
            "k_core":       pd.Series(kcore,   dtype="int16"),
            "in_degree_w":  pd.Series(in_w,    dtype="float32"),
            "out_degree_w": pd.Series(out_w,   dtype="float32"),
            "total_degree": pd.Series(total_deg, dtype="int32"),
        }
    )
    graph_feat.index.name = "player"

    # ------------------------------------------------------------------ #
    # 3. Train Node2Vec embeddings                                         #
    # ------------------------------------------------------------------ #
    logger.info("Training Node2Vec embeddings ...")
    wv = _train_node2vec(G)

    zero_emb = np.zeros(N2V_DIMENSIONS, dtype="float32")
    emb_cols  = [f"n2v_{i}" for i in range(N2V_DIMENSIONS)]

    emb_records = {}
    for node in G.nodes():
        key = str(node)
        vec = wv[key] if key in wv else zero_emb
        emb_records[node] = vec

    emb_df = pd.DataFrame.from_dict(
        emb_records,
        orient="index",
        columns=emb_cols,
        dtype="float32",
    )
    emb_df.index.name = "player"

    # ------------------------------------------------------------------ #
    # 4. Player stats (avg Elo + games count) from parquet                 #
    # ------------------------------------------------------------------ #
    if games_path is None:
        games_path = INTERIM_DIR / "games_filtered.parquet"
    stats = _player_stats(games_path)

    # ------------------------------------------------------------------ #
    # 5. Filter: keep players with >= min_games games                      #
    # ------------------------------------------------------------------ #
    qualified = stats[stats["games_played"] >= min_games].index
    logger.info(
        f"Players with >= {min_games} games: {len(qualified):,} "
        f"(out of {len(stats):,} total)"
    )

    # Intersect with graph nodes (some players may have been pruned earlier)
    graph_nodes = pd.Index(G.nodes())
    keep = qualified.intersection(graph_nodes)
    logger.info(
        f"Players in both graph and stats >= {min_games} games: {len(keep):,}"
    )

    # ------------------------------------------------------------------ #
    # 6. Merge everything                                                  #
    # ------------------------------------------------------------------ #
    df = (
        graph_feat.loc[graph_feat.index.isin(keep)]
        .join(emb_df, how="left")
        .join(stats[["games_played", "target_elo"]], how="left")
        .reset_index()          # player becomes a regular column
    )

    # Enforce column order: player | graph features | embeddings | stats | target
    ordered_cols = (
        ["player"]
        + ["pagerank", "betweenness", "k_core",
           "in_degree_w", "out_degree_w", "total_degree"]
        + emb_cols
        + ["games_played", "target_elo"]
    )
    df = df[ordered_cols]

    df["games_played"] = df["games_played"].astype("int32")
    df["target_elo"]   = df["target_elo"].astype("float32")

    # ------------------------------------------------------------------ #
    # 7. Save                                                              #
    # ------------------------------------------------------------------ #
    ensure_dirs(PROCESSED_DIR)
    out = PROCESSED_DIR / "player_features.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Saved {len(df):,} rows -> {out}  shape={df.shape}")

    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Extract graph features and Node2Vec embeddings per player."
    )
    p.add_argument(
        "--min-games",
        type=int,
        default=MIN_FEATURES_GAMES,
        help=f"Minimum games per player (default: {MIN_FEATURES_GAMES})",
    )
    args = p.parse_args()

    df = extract_features(min_games=args.min_games)
    print(f"\nFeature matrix: {df.shape[0]:,} players x {df.shape[1]} columns")
    print(df.dtypes.to_string())
    print(f"\ntarget_elo stats:\n{df['target_elo'].describe().round(1).to_string()}")
