"""
visualize_interactive_network.py
---------------------------------
Build a polished interactive Plotly network visualization of the chess
competition graph, then save it as HTML (and optionally PNG).

Data sources
------------
data/interim/player_graph.pkl          NetworkX DiGraph
data/processed/player_features.parquet  per-player stats & embeddings

Node encoding
-------------
color  = target_elo   (Plasma colorscale with colorbar)
size   = total_degree (log-scaled to 6-28 px)

Tooltip fields
--------------
player | target_elo | games_played | pagerank | k_core | total_degree

Outputs
-------
outputs/interactive_network.html   (full interactive Plotly figure)
outputs/interactive_network.png    (1600x1000 static screenshot, requires kaleido)

Usage
-----
python -m app.visualize_interactive_network
python -m app.visualize_interactive_network --max-nodes 500
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.config import INTERIM_DIR, OUTPUTS_DIR, PROCESSED_DIR
from app.utils import ensure_dirs, get_logger, timer

logger = get_logger(__name__)

# ── tuneable constants ────────────────────────────────────────────────────────

MAX_NODES      = 300      # subsample threshold
SPRING_K       = 1.2      # node repulsion (higher = more spread)
SPRING_ITERS   = 80       # layout iterations
RANDOM_SEED    = 42
NODE_MIN_PX    = 6        # smallest node marker size (px)
NODE_MAX_PX    = 28       # largest  node marker size (px)
EDGE_COLOR     = "rgba(200, 210, 220, 0.10)"
BG_COLOR       = "#0d1117"
COLORSCALE     = "Plasma"

# ── I/O helpers ───────────────────────────────────────────────────────────────


def _load_graph(path: Path) -> nx.DiGraph:
    with open(path, "rb") as fh:
        G = pickle.load(fh)
    if not isinstance(G, nx.DiGraph):
        raise TypeError(f"Expected DiGraph in {path}, got {type(G).__name__}")
    return G


def _load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


# ── subgraph sampling ─────────────────────────────────────────────────────────


def _sample_subgraph(
    G: nx.DiGraph,
    features: pd.DataFrame,
    max_nodes: int,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """
    Return the subgraph induced by the top-*max_nodes* players ranked by
    total_degree.

    Priority: nodes present in *features* (they have richer data) are
    ranked by their ``total_degree`` column; remaining graph nodes are
    ranked by their raw in+out degree.
    """
    n = G.number_of_nodes()
    if n <= max_nodes:
        logger.info(f"Graph has {n:,} nodes -- no sampling needed.")
        return G, features

    logger.info(f"Sampling top {max_nodes} nodes (by total_degree) from {n:,} ...")

    feat_idx = features.set_index("player")

    def _degree(node: str) -> float:
        if node in feat_idx.index:
            return float(feat_idx.at[node, "total_degree"])
        return float(G.in_degree(node) + G.out_degree(node))

    ranked = sorted(G.nodes(), key=_degree, reverse=True)
    top    = set(ranked[:max_nodes])

    sub_G    = G.subgraph(top).copy()
    sub_feat = features[features["player"].isin(top)].copy()

    logger.info(
        f"Subgraph: {sub_G.number_of_nodes():,} nodes, "
        f"{sub_G.number_of_edges():,} directed edges"
    )
    return sub_G, sub_feat


# ── layout ────────────────────────────────────────────────────────────────────


def _compute_layout(G: nx.DiGraph) -> dict[str, tuple[float, float]]:
    n = G.number_of_nodes()
    logger.info(
        f"Computing spring layout for {n:,} nodes "
        f"(k={SPRING_K}, iters={SPRING_ITERS}) ..."
    )
    # spring_layout works on directed graphs; the undirected projection
    # produces a more symmetric, readable result for large networks.
    G_und = G.to_undirected()
    pos = nx.spring_layout(
        G_und,
        k=SPRING_K,
        iterations=SPRING_ITERS,
        seed=RANDOM_SEED,
    )
    return pos


# ── size scaling ──────────────────────────────────────────────────────────────


def _log_scale(values: np.ndarray) -> np.ndarray:
    """Log-scale *values* into [NODE_MIN_PX, NODE_MAX_PX]."""
    v = np.log1p(values.clip(min=0).astype(float))
    lo, hi = v.min(), v.max()
    if hi == lo:
        return np.full(len(v), (NODE_MIN_PX + NODE_MAX_PX) / 2.0)
    return NODE_MIN_PX + (NODE_MAX_PX - NODE_MIN_PX) * (v - lo) / (hi - lo)


# ── trace builders ────────────────────────────────────────────────────────────


def _edge_trace(G: nx.DiGraph, pos: dict) -> go.Scatter:
    """
    Single Scatter trace for all edges, using ``None`` separators between
    segments.  This is an order of magnitude faster than one trace per edge.
    """
    xs: list[Optional[float]] = []
    ys: list[Optional[float]] = []

    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]

    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(width=0.6, color=EDGE_COLOR),
        hoverinfo="none",
        showlegend=False,
        name="edges",
    )


def _node_trace(
    G: nx.DiGraph,
    pos: dict,
    features: pd.DataFrame,
) -> go.Scatter:
    """
    Scatter trace for all nodes.  Hover text is pre-built as HTML so
    Plotly renders it with ``hovertemplate='%{text}<extra></extra>'``.
    """
    feat_idx = features.set_index("player")

    xs, ys          = [], []
    elo_vals        = []
    degree_vals     = []
    hover_texts     = []

    for node in G.nodes():
        if node not in pos:
            continue

        x, y = pos[node]
        xs.append(x)
        ys.append(y)

        if node in feat_idx.index:
            row = feat_idx.loc[node]
            elo     = float(row.get("target_elo",   1200))
            games   = int(row.get("games_played",   0))
            pr      = float(row.get("pagerank",     0.0))
            kc      = int(row.get("k_core",         0))
            deg     = int(row.get("total_degree",   0))
        else:
            # Fallback: use graph node attributes set by build_graph.py
            attrs = G.nodes[node]
            elo   = float(attrs.get("avg_elo",     1200))
            games = int(attrs.get("total_games",   0))
            pr    = 0.0
            kc    = 0
            deg   = G.in_degree(node) + G.out_degree(node)

        elo_vals.append(elo)
        degree_vals.append(deg)

        hover_texts.append(
            f"<b style='font-size:13px'>{node}</b><br>"
            f"<span style='color:#aaa'>Target Elo</span>&nbsp;&nbsp;&nbsp;"
            f"<b>{elo:.0f}</b><br>"
            f"<span style='color:#aaa'>Games played</span>&nbsp;&nbsp;"
            f"<b>{games:,}</b><br>"
            f"<span style='color:#aaa'>PageRank</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            f"<b>{pr:.5f}</b><br>"
            f"<span style='color:#aaa'>k-core</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            f"<b>{kc}</b><br>"
            f"<span style='color:#aaa'>Total degree</span>&nbsp;&nbsp;"
            f"<b>{deg}</b>"
        )

    sizes = _log_scale(np.array(degree_vals, dtype=float))
    elo_arr = np.array(elo_vals, dtype=float)

    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(
            size=sizes.tolist(),
            color=elo_arr.tolist(),
            colorscale=COLORSCALE,
            cmin=float(elo_arr.min()),
            cmax=float(elo_arr.max()),
            colorbar=dict(
                title=dict(
                    text="Target Elo",
                    font=dict(size=12, color="#cccccc"),
                    side="right",
                ),
                tickfont=dict(color="#bbbbbb", size=10),
                outlinecolor="#444",
                outlinewidth=1,
                thickness=14,
                len=0.55,
                x=1.01,
            ),
            showscale=True,
            opacity=0.90,
            line=dict(width=0.8, color="rgba(255,255,255,0.18)"),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        hoverlabel=dict(
            bgcolor="#1a1f2e",
            bordercolor="#4a5568",
            font=dict(size=11, color="white"),
        ),
        showlegend=False,
        name="players",
    )


# ── figure assembly ───────────────────────────────────────────────────────────


def _assemble_figure(
    e_trace: go.Scatter,
    n_trace: go.Scatter,
    n_nodes: int,
    n_edges: int,
) -> go.Figure:
    axis_style = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=False,
        ticks="",
    )

    fig = go.Figure(
        data=[e_trace, n_trace],
        layout=go.Layout(
            title=dict(
                text=(
                    "Chess Competition Network"
                    f"<br><sup style='color:#8899aa'>"
                    f"{n_nodes:,} players &nbsp;|&nbsp; "
                    f"{n_edges:,} directed edges &nbsp;|&nbsp; "
                    f"color = Elo &nbsp;|&nbsp; size = degree"
                    f"</sup>"
                ),
                font=dict(size=20, color="#e8eaf0", family="Arial, sans-serif"),
                x=0.5,
                xanchor="center",
                y=0.98,
            ),
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            template="plotly_dark",
            showlegend=False,
            hovermode="closest",
            margin=dict(t=90, l=10, r=80, b=10),
            xaxis=dict(**axis_style),
            yaxis=dict(**axis_style),
            font=dict(family="Arial, sans-serif"),
            annotations=[
                dict(
                    text=(
                        "Hover over a node to see player details. "
                        "Scroll to zoom, drag to pan."
                    ),
                    xref="paper", yref="paper",
                    x=0.01, y=0.01,
                    showarrow=False,
                    font=dict(size=10, color="#556677"),
                    align="left",
                )
            ],
        ),
    )
    return fig


# ── public API ────────────────────────────────────────────────────────────────


@timer
def build_visualization(
    graph_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
    max_nodes: int = MAX_NODES,
) -> go.Figure:
    """
    Build and save the interactive network visualization.

    Parameters
    ----------
    graph_path : Path, optional
        Path to ``player_graph.pkl``.
    features_path : Path, optional
        Path to ``player_features.parquet``.
    max_nodes : int
        Maximum number of nodes to render (top by total_degree).

    Returns
    -------
    go.Figure
    """
    if graph_path   is None: graph_path   = INTERIM_DIR   / "player_graph.pkl"
    if features_path is None: features_path = PROCESSED_DIR / "player_features.parquet"

    # ── load ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading graph    from {graph_path}")
    G = _load_graph(graph_path)
    logger.info(
        f"Graph loaded: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} directed edges"
    )

    logger.info(f"Loading features from {features_path}")
    features = _load_features(features_path)
    logger.info(f"Features loaded: {len(features):,} rows")

    # ── sample ────────────────────────────────────────────────────────────────
    G_plot, feat_plot = _sample_subgraph(G, features, max_nodes)
    n_nodes = G_plot.number_of_nodes()
    n_edges = G_plot.number_of_edges()

    # ── layout ────────────────────────────────────────────────────────────────
    pos = _compute_layout(G_plot)

    # ── traces ────────────────────────────────────────────────────────────────
    logger.info("Building Plotly traces ...")
    e_trace = _edge_trace(G_plot, pos)
    n_trace = _node_trace(G_plot, pos, feat_plot)

    # ── figure ────────────────────────────────────────────────────────────────
    logger.info("Assembling figure ...")
    fig = _assemble_figure(e_trace, n_trace, n_nodes, n_edges)

    # ── save HTML ─────────────────────────────────────────────────────────────
    ensure_dirs(OUTPUTS_DIR)
    html_path = OUTPUTS_DIR / "interactive_network.html"
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",     # ~3 KB stub instead of 3 MB inline
        full_html=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "chess_network",
                "width": 1600,
                "height": 1000,
                "scale": 2,
            },
        },
    )
    logger.info(f"HTML saved -> {html_path}")

    # ── save PNG (kaleido) ────────────────────────────────────────────────────
    png_path = OUTPUTS_DIR / "interactive_network.png"
    try:
        fig.write_image(
            str(png_path),
            width=1600,
            height=1000,
            scale=2,
            engine="kaleido",
        )
        logger.info(f"PNG  saved -> {png_path}")
    except Exception as exc:
        warnings.warn(
            f"Static PNG export failed ({exc.__class__.__name__}: {exc}). "
            "The HTML file is unaffected. "
            "To enable PNG export: pip install kaleido "
            "then run: choreo_get_chrome  (downloads headless Chrome).",
            stacklevel=2,
        )
        png_path = None

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\nNodes plotted : {n_nodes:,}")
    print(f"Edges plotted : {n_edges:,}")
    print(f"HTML          : {html_path}")
    if png_path and png_path.exists():
        print(f"PNG           : {png_path}")
    else:
        print("PNG           : skipped (kaleido / Chrome not ready)")

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build interactive chess network visualization."
    )
    p.add_argument(
        "--max-nodes",
        type=int,
        default=MAX_NODES,
        help=f"Maximum nodes to render (default: {MAX_NODES})",
    )
    p.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Path to player_graph.pkl",
    )
    p.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Path to player_features.parquet",
    )
    args = p.parse_args()
    build_visualization(
        graph_path=args.graph,
        features_path=args.features,
        max_nodes=args.max_nodes,
    )
