"""
saver.py

Responsibility: write outputs to the correct data folder.

Per guide 3.3: raw data is immutable and lives in data/raw/. Cleaned and
derived outputs go in data/processed/ so a raw observation can always be
compared against its processed counterpart.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_raw(df: pd.DataFrame, filename: str, raw_dir: Path) -> Path:
    """Save an untouched, as-downloaded DataFrame to data/raw/.

    Parameters
    ----------
    df:
        Raw DataFrame or Series to persist, exactly as downloaded.
    filename:
        Output file name, e.g. 'prices_adjusted_close.csv'.
    raw_dir:
        Destination directory (config.DATA_RAW_DIR).

    Returns
    -------
    Path
        Full path to the saved file.
    """
    path = Path(raw_dir) / filename
    df.to_csv(path)
    logger.info("Saved raw output: %s (%s)", path, getattr(df, "shape", len(df)))
    return path


def save_processed(df: pd.DataFrame, filename: str, processed_dir: Path) -> Path:
    """Save a cleaned/derived DataFrame to data/processed/.

    Parameters
    ----------
    df:
        Cleaned or derived DataFrame/Series to persist.
    filename:
        Output file name, e.g. 'returns_daily_log.csv'.
    processed_dir:
        Destination directory (config.DATA_PROCESSED_DIR).

    Returns
    -------
    Path
        Full path to the saved file.
    """
    path = Path(processed_dir) / filename
    df.to_csv(path)
    logger.info("Saved processed output: %s (%s)", path, getattr(df, "shape", len(df)))
    return path
