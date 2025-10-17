"""
Data splitting utilities for RiceHealthAI.

Provides a function to split a dataset of rice leaf images
into train, validation, and test sets while preserving label distribution.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from ricehealthai.infrastructure.logger import setup_logger


def split_riceleaf_dataset(
    df: pd.DataFrame,
    train_size: float | None = None,
    valid_size: float | None = None,
    test_size: float | None = None,
    random_state: int = 42,
    shuffle: bool = True,
    stratify_col: str | None = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a pandas DataFrame into train/validation/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (must contain at least the columns 'image_path' and 'label').
    train_size : float, optional
        Proportion for the training set. Defaults to 0.7 if not provided.
    valid_size : float, optional
        Proportion for the validation set. Defaults to 0.15 if not provided.
    test_size : float, optional
        Proportion for the test set. Defaults to 0.15 if not provided.
    random_state : int, default=42
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    stratify_col : str, optional
        Column name to use for stratification. Defaults to 'label'.

    Returns
    -------
    train_df, valid_df, test_df : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Three DataFrames preserving all original columns.
    """
    logger = setup_logger(__name__)

    # Vérifications préliminaires
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if stratify_col and stratify_col not in df.columns:
        raise ValueError(f"Column '{stratify_col}' not found in DataFrame.")

    # Proportions par défaut
    if train_size is None and valid_size is None and test_size is None:
        train_size, valid_size, test_size = 0.7, 0.15, 0.15

    # Normalisation si la somme ≠ 1
    total = sum(x for x in [train_size, valid_size, test_size] if x is not None)
    if abs(total - 1.0) > 1e-6:
        logger.warning(f"Proportions do not sum to 1 (total={total:.2f}). Normalizing...")
        train_size = (train_size or 0.7) / total
        valid_size = (valid_size or 0.15) / total
        test_size = (test_size or 0.15) / total

    stratify = df[stratify_col] if stratify_col else None

    # 1. Split principal : train vs temp
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    # 2. Split du reste : valid vs test
    relative_valid_size = valid_size / (valid_size + test_size)
    stratify_temp = temp_df[stratify_col] if stratify_col else None

    valid_df, test_df = train_test_split(
        temp_df,
        train_size=relative_valid_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_temp,
    )

    logger.info(
        f"Dataset split completed - "
        f"Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}"
    )

    return train_df, valid_df, test_df
