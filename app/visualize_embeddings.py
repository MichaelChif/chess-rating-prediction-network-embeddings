"""
visualize_embeddings.py
-----------------------
Reduce the 32-dimensional Node2Vec embeddings stored in
player_features.parquet to 2D with PCA, then produce a polished
interactive Plotly scatter plot.

Encoding
--------
x / y   PCA dimensions 1 and 2 (axis labels include variance explained %)
color   target_elo        (Plasma colorscale + colorbar)
size    games_played      (log-scaled to 4-20 px)

Hover tooltip
-------------
player | target_elo | games_played | pagerank | k_core

Outputs
-------
outputs/node2vec_embedding_map.html    interactive Plotly figure
outputs/node2vec_embedding_map.png     1600x1000 static screenshot (kaleido)

Usage
-----
python -m app.visualize_embeddings
python -m app.visualize_embeddings --features path/to/player_features.parquet
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from app.config import N2V_DIMENSIONS, OUTPUTS_DIR, PROCESSED_DIR
from app.utils import ensure_dirs, get_logger, timer

logger = get_logger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

BG_COLOR   = "#0d1117"
COLORSCALE = "Plasma"
NODE_MIN_PX = 4
NODE_MAX_PX = 20
RANDOM_SEED = 42


# ── helpers ───────────────────────────────────────────────────────────────────


def _embedding_columns(df: pd.DataFrame) -> list[str]:
    """Return n2v_0 … n2v_{N2V_DIMENSIONS-1}, in order, asserting they exist."""
    cols = [f"n2v_{i}" for i in range(N2V_DIMENSIONS)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{len(missing)} embedding column(s) missing from parquet "
            f"(first missing: '{missing[0]}'). "
            f"Re-run features.py to regenerate player_features.parquet."
        )
    return cols


def _log_scale(
    values: np.ndarray,
    lo: float = NODE_MIN_PX,
    hi: float = NODE_MAX_PX,
) -> np.ndarray:
    """Map *values* through log1p then linearly into [lo, hi]."""
    v = np.log1p(values.clip(min=0).astype(float))
    v_lo, v_hi = v.min(), v.max()
    if v_hi == v_lo:
        return np.full(len(v), (lo + hi) / 2.0)
    return lo + (hi - lo) * (v - v_lo) / (v_hi - v_lo)


def _run_pca(
    emb: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Fit PCA(n_components=2) on *emb*.

    Returns
    -------
    coords       (N, 2) array of 2D coordinates
    var1, var2   explained variance ratio for PC1 and PC2 (as percentages)
    """
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(emb)
    var1, var2 = pca.explained_variance_ratio_ * 100
    logger.info(
        f"PCA: PC1 explains {var1:.1f}%, "
        f"PC2 explains {var2:.1f}%  "
        f"(total {var1+var2:.1f}%)"
    )
    return coords, var1, var2


def _build_hover(row: pd.Series) -> str:
    return (
        f"<b style='font-size:13px'>{row['player']}</b><br>"
        f"<span style='color:#aaa'>Target Elo</span>&nbsp;&nbsp;&nbsp;"
        f"<b>{row['target_elo']:.0f}</b><br>"
        f"<span style='color:#aaa'>Games played</span>&nbsp;&nbsp;"
        f"<b>{int(row['games_played']):,}</b><br>"
        f"<span style='color:#aaa'>PageRank</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        f"<b>{row['pagerank']:.5f}</b><br>"
        f"<span style='color:#aaa'>k-core</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        f"<b>{int(row['k_core'])}</b>"
    )


def _scatter_trace(
    df: pd.DataFrame,
    coords: np.ndarray,
) -> go.Scatter:
    elo_arr   = df["target_elo"].astype(float).values
    games_arr = df["games_played"].astype(float).values
    sizes     = _log_scale(games_arr)
    hovers    = [_build_hover(row) for _, row in df.iterrows()]

    return go.Scatter(
        x=coords[:, 0].tolist(),
        y=coords[:, 1].tolist(),
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
                len=0.60,
                x=1.01,
            ),
            showscale=True,
            opacity=0.82,
            line=dict(width=0.4, color="rgba(255,255,255,0.15)"),
        ),
        text=hovers,
        hovertemplate="%{text}<extra></extra>",
        hoverlabel=dict(
            bgcolor="#1a1f2e",
            bordercolor="#4a5568",
            font=dict(size=11, color="white"),
        ),
        showlegend=False,
        name="players",
    )


def _assemble_figure(
    trace: go.Scatter,
    var1: float,
    var2: float,
    n_players: int,
    n_dims: int,
) -> go.Figure:
    axis_style = dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.12)",
        zerolinewidth=1,
        showticklabels=True,
        tickfont=dict(size=9, color="#667788"),
        ticks="outside",
        tickcolor="#334455",
    )

    fig = go.Figure(
        data=[trace],
        layout=go.Layout(
            title=dict(
                text=(
                    "Node2Vec Embedding Map  (PCA projection)"
                    f"<br><sup style='color:#8899aa'>"
                    f"{n_players:,} players &nbsp;|&nbsp; "
                    f"{n_dims}D embeddings -> 2D &nbsp;|&nbsp; "
                    f"color = Elo &nbsp;|&nbsp; size = games played"
                    f"</sup>"
                ),
                font=dict(size=20, color="#e8eaf0", family="Arial, sans-serif"),
                x=0.5,
                xanchor="center",
                y=0.97,
            ),
            xaxis=dict(
                title=dict(
                    text=f"PC 1  ({var1:.1f}% variance explained)",
                    font=dict(size=11, color="#99aabb"),
                ),
                **axis_style,
            ),
            yaxis=dict(
                title=dict(
                    text=f"PC 2  ({var2:.1f}% variance explained)",
                    font=dict(size=11, color="#99aabb"),
                ),
                **axis_style,
            ),
            paper_bgcolor=BG_COLOR,
            plot_bgcolor="#111820",
            template="plotly_dark",
            showlegend=False,
            hovermode="closest",
            margin=dict(t=95, l=70, r=80, b=60),
            font=dict(family="Arial, sans-serif"),
            annotations=[
                dict(
                    text=(
                        f"Total variance explained: {var1+var2:.1f}%"
                    ),
                    xref="paper", yref="paper",
                    x=0.01, y=0.01,
                    showarrow=False,
                    font=dict(size=10, color="#556677"),
                    align="left",
                ),
                dict(
                    text="Hover to inspect players",
                    xref="paper", yref="paper",
                    x=0.99, y=0.01,
                    showarrow=False,
                    font=dict(size=10, color="#556677"),
                    align="right",
                ),
            ],
        ),
    )
    return fig


# ── public API ────────────────────────────────────────────────────────────────


@timer
def build_embedding_viz(
    features_path: Optional[Path] = None,
) -> go.Figure:
    """
    Build and save the Node2Vec embedding scatter plot.

    Parameters
    ----------
    features_path : Path, optional
        Path to ``player_features.parquet``.
        Defaults to ``data/processed/player_features.parquet``.

    Returns
    -------
    go.Figure
    """
    if features_path is None:
        features_path = PROCESSED_DIR / "player_features.parquet"

    # ── load ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded: {len(df):,} players, {len(df.columns)} columns")

    # ── validate required columns ─────────────────────────────────────────────
    required = {"player", "target_elo", "games_played", "pagerank", "k_core"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Required column(s) missing from parquet: {missing}. "
            "Re-run features.py."
        )

    emb_cols = _embedding_columns(df)
    logger.info(
        f"Embedding columns: {len(emb_cols)} "
        f"({emb_cols[0]} ... {emb_cols[-1]})"
    )

    # ── PCA ───────────────────────────────────────────────────────────────────
    emb = df[emb_cols].astype(np.float32).values
    coords, var1, var2 = _run_pca(emb)

    # ── build traces ──────────────────────────────────────────────────────────
    logger.info("Building scatter trace ...")
    trace = _scatter_trace(df, coords)

    # ── assemble figure ───────────────────────────────────────────────────────
    fig = _assemble_figure(
        trace,
        var1=var1,
        var2=var2,
        n_players=len(df),
        n_dims=len(emb_cols),
    )

    # ── persist HTML ──────────────────────────────────────────────────────────
    ensure_dirs(OUTPUTS_DIR)
    html_path = OUTPUTS_DIR / "node2vec_embedding_map.html"
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "node2vec_embedding_map",
                "width": 1600,
                "height": 1000,
                "scale": 2,
            },
        },
    )
    logger.info(f"HTML saved -> {html_path}")

    # ── persist PNG ───────────────────────────────────────────────────────────
    png_path = OUTPUTS_DIR / "node2vec_embedding_map.png"
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
            "To enable PNG: pip install kaleido, "
            "then run choreo_get_chrome to download headless Chrome.",
            stacklevel=2,
        )
        png_path = None

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\nPlayers   : {len(df):,}")
    print(f"PC1 var   : {var1:.2f}%")
    print(f"PC2 var   : {var2:.2f}%")
    print(f"Total var : {var1+var2:.2f}%")
    print(f"HTML      : {html_path}")
    if png_path and png_path.exists():
        print(f"PNG       : {png_path}")
    else:
        print("PNG       : skipped (kaleido / Chrome not ready)")

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Visualize Node2Vec embeddings with PCA."
    )
    p.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Path to player_features.parquet "
             "(default: data/processed/player_features.parquet)",
    )
    args = p.parse_args()
    build_embedding_viz(features_path=args.features)
