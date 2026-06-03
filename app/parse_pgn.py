"""
parse_pgn.py -- Stream-parse Lichess .pgn or .pgn.zst files into a
filtered DataFrame.

Design
------
* Headers are extracted with a hand-rolled regex line scanner.
  Moves are never parsed -- each movetext block is skipped with a single
  readline loop, making this safe for files of any size.
* .zst files are decompressed on-the-fly via a zstandard streaming
  reader; the raw bytes never fully materialise in memory.
* All filtering happens per-game before any object is appended to the
  result list, keeping peak memory proportional to the *kept* rows only.

Filters applied
---------------
* time_control  ∈ {"300+0", "600+0"}   (5-min and 10-min games)
* result        ∈ {"1-0", "0-1"}        (decisive; draws discarded)
* WhiteElo and BlackElo both present and > 0

Output columns
--------------
white_player, black_player,
winner_player, loser_player,
white_elo, black_elo, winner_elo, loser_elo,
result, time_control, timestamp

Saved to
--------
data/interim/games_filtered.parquet
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import IO, Iterator, Optional

import pandas as pd

from app.config import INTERIM_DIR, RAW_DIR
from app.utils import ensure_dirs, get_logger, timer

logger = get_logger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

ALLOWED_TIME_CONTROLS: frozenset[str] = frozenset({"300+0", "600+0"})
DECISIVE_RESULTS: frozenset[str] = frozenset({"1-0", "0-1"})

_TAG_RE = re.compile(r'^\[(\w+)\s+"([^"]*)"\s*\]')

# ── PGN streaming helpers ─────────────────────────────────────────────────────


def _open_text_stream(path: Path) -> IO[str]:
    """
    Return a UTF-8 text stream for *path*.
    Handles both plain .pgn files and Zstandard-compressed .pgn.zst files.
    The caller is responsible for closing the returned stream.
    """
    suffix = path.suffix.lower()
    if suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError(
                "Install 'zstandard' to read .zst files:  pip install zstandard"
            ) from exc
        raw_fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        # max_window_size keeps decompressor memory bounded
        raw_stream = dctx.stream_reader(raw_fh, closefd=True)
        return io.TextIOWrapper(raw_stream, encoding="utf-8", errors="ignore")

    return open(path, encoding="utf-8", errors="ignore")


def _iter_game_headers(text_stream: IO[str]) -> Iterator[dict[str, str]]:
    """
    Yield one ``{tag: value}`` dict per game in *text_stream*.

    Algorithm
    ---------
    PGN games are separated by an empty line between the header block and
    the movetext, followed by another empty line after the movetext.
    We scan line by line:

    * Lines that match the ``[Tag "Value"]`` pattern accumulate the current
      game's headers.
    * The first empty line after at least one header tag marks the end of
      the header block -> yield the headers dict, then enter "skip-moves" mode.
    * In skip-moves mode every line is discarded until the next tag line
      starts a new game.
    """
    headers: dict[str, str] = {}
    in_moves = False  # True after the header block's trailing blank line

    for raw_line in text_stream:
        line = raw_line.rstrip("\r\n")

        m = _TAG_RE.match(line)
        if m:
            # New tag -> always means we are (back) in a header block
            if in_moves:
                in_moves = False
            headers[m.group(1)] = m.group(2)

        elif not line:
            # Empty line
            if headers and not in_moves:
                # End of header block -- emit the game, then skip moves
                yield headers
                headers = {}
                in_moves = True
            # else: blank line inside movetext or between games -> ignore

        # Non-empty, non-tag line -> movetext; ignore

    # Edge-case: file doesn't end with a trailing blank line
    if headers:
        yield headers


def _find_pgn_files(raw_dir: Path) -> list[Path]:
    """Return all .pgn and .pgn.zst files found directly in *raw_dir*."""
    files: list[Path] = []
    for pattern in ("*.pgn", "*.pgn.zst"):
        files.extend(raw_dir.glob(pattern))
    return sorted(files)


# ── row builder / filter ─────────────────────────────────────────────────────


def _build_row(h: dict[str, str]) -> Optional[dict]:
    """
    Convert a raw header dict to a result row, applying all filters.
    Returns ``None`` when the game should be discarded.
    """
    # ── time-control filter ──────────────────────────────────────────
    tc = h.get("TimeControl", "")
    if tc not in ALLOWED_TIME_CONTROLS:
        return None

    # ── result filter (decisive only) ────────────────────────────────
    result = h.get("Result", "*")
    if result not in DECISIVE_RESULTS:
        return None

    # ── rating filter ────────────────────────────────────────────────
    try:
        white_elo = int(h["WhiteElo"])
        black_elo = int(h["BlackElo"])
    except (KeyError, ValueError, TypeError):
        return None

    if white_elo <= 0 or black_elo <= 0:
        return None

    # ── player names ────────────────────────────────────────────────
    white = h.get("White", "").strip()
    black = h.get("Black", "").strip()
    if not white or not black or white == "?" or black == "?":
        return None

    # ── winner / loser ───────────────────────────────────────────────
    if result == "1-0":
        winner, loser = white, black
        winner_elo, loser_elo = white_elo, black_elo
    else:  # "0-1"
        winner, loser = black, white
        winner_elo, loser_elo = black_elo, white_elo

    # ── timestamp ───────────────────────────────────────────────────
    date_str = h.get("UTCDate", "").replace(".", "-")  # "2013.02.14" -> "2013-02-14"
    time_str = h.get("UTCTime", "")
    timestamp: Optional[pd.Timestamp] = None
    if date_str and time_str and "?" not in date_str:
        try:
            timestamp = pd.Timestamp(f"{date_str} {time_str}", tz="UTC")
        except Exception:
            pass

    return {
        "white_player": white,
        "black_player": black,
        "winner_player": winner,
        "loser_player": loser,
        "white_elo": white_elo,
        "black_elo": black_elo,
        "winner_elo": winner_elo,
        "loser_elo": loser_elo,
        "result": result,
        "time_control": tc,
        "timestamp": timestamp,
    }


# ── public API ────────────────────────────────────────────────────────────────


@timer
def parse_pgn(
    raw_dir: Path = RAW_DIR,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Stream-parse every .pgn / .pgn.zst file in *raw_dir* and return a
    filtered, typed DataFrame.

    Parameters
    ----------
    raw_dir : Path
        Directory that contains the PGN source files.
    output_path : Path, optional
        Destination parquet path.  Defaults to
        ``data/interim/games_filtered.parquet``.

    Returns
    -------
    pd.DataFrame
        One row per kept game with the columns described in the module
        docstring.
    """
    if output_path is None:
        output_path = INTERIM_DIR / "games_filtered.parquet"

    pgn_files = _find_pgn_files(raw_dir)
    if not pgn_files:
        raise FileNotFoundError(
            f"No .pgn or .pgn.zst files found in: {raw_dir}"
        )

    logger.info(
        f"Found {len(pgn_files)} file(s): {[f.name for f in pgn_files]}"
    )

    rows: list[dict] = []
    total_seen = 0
    total_kept = 0

    for pgn_path in pgn_files:
        logger.info(f"Streaming {pgn_path.name} ...")
        file_seen = file_kept = 0

        stream = _open_text_stream(pgn_path)
        try:
            for raw_headers in _iter_game_headers(stream):
                total_seen += 1
                file_seen += 1

                row = _build_row(raw_headers)
                if row is not None:
                    rows.append(row)
                    total_kept += 1
                    file_kept += 1

                if file_seen % 50_000 == 0:
                    pct = 100.0 * file_kept / file_seen
                    logger.info(
                        f"  {file_seen:>8,} scanned | "
                        f"{file_kept:>7,} kept ({pct:.1f}%)"
                    )
        finally:
            stream.close()

        pct = 100.0 * file_kept / file_seen if file_seen else 0.0
        logger.info(
            f"  {pgn_path.name}: {file_seen:,} scanned, "
            f"{file_kept:,} kept ({pct:.1f}%)"
        )

    pct_total = 100.0 * total_kept / total_seen if total_seen else 0.0
    logger.info(
        f"Total: {total_seen:,} games scanned, "
        f"{total_kept:,} kept ({pct_total:.1f}%)"
    )

    if not rows:
        logger.warning(
            "No games passed the filters. "
            "Check that the PGN contains 300+0 or 600+0 time controls."
        )
        df = pd.DataFrame(
            columns=[
                "white_player", "black_player",
                "winner_player", "loser_player",
                "white_elo", "black_elo", "winner_elo", "loser_elo",
                "result", "time_control", "timestamp",
            ]
        )
    else:
        df = pd.DataFrame(rows)
        df["white_elo"] = df["white_elo"].astype("int16")
        df["black_elo"] = df["black_elo"].astype("int16")
        df["winner_elo"] = df["winner_elo"].astype("int16")
        df["loser_elo"] = df["loser_elo"].astype("int16")
        df["result"] = df["result"].astype("category")
        df["time_control"] = df["time_control"].astype("category")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    ensure_dirs(INTERIM_DIR)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df):,} rows -> {output_path}  shape={df.shape}")

    return df


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Parse PGN files into a filtered parquet.")
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing .pgn / .pgn.zst files",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .parquet path (default: data/interim/games_filtered.parquet)",
    )
    args = p.parse_args()
    parse_pgn(raw_dir=args.raw_dir, output_path=args.output)
