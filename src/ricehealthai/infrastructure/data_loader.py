"""
Data loading utilities for RiceHealthAI.

Provides functions to build a structured DataFrame
containing image file paths and labels, and to summarize
the dataset distribution across categories.
"""

import os
import pandas as pd
from ricehealthai.core.settings import load_config, find_project_root
from ricehealthai.infrastructure.logger import setup_logger


def build_image_dataframe(data_dir: str = None, show_summary: bool = False) -> pd.DataFrame:
    """
    Build a DataFrame with image file paths and their corresponding labels.

    Args:
        data_dir (str, optional): Relative or absolute path to the directory
            containing image category subfolders. If None, uses the path defined
            in `configs/data_config.yaml`.
        show_summary (bool, optional): If True, automatically display a
            summary of image counts per category after building the DataFrame.
            Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - 'image_path': full path to each image file
            - 'label': category name (folder name)
    """
    logger = setup_logger(__name__)
    data_cfg = load_config("configs/data_config.yaml")

    # Determine dataset directory
    data_dir = data_dir or data_cfg["data"]["raw_dir"]
    project_root = find_project_root()
    full_data_dir = os.path.join(project_root, data_dir)

    logger.info(f"Building image DataFrame from directory: {full_data_dir}")

    if not os.path.exists(full_data_dir):
        raise FileNotFoundError(f"Data directory not found: {full_data_dir}")

    data = []
    categories = [
        d for d in sorted(os.listdir(full_data_dir))
        if os.path.isdir(os.path.join(full_data_dir, d))
    ]

    logger.info(f"Detected categories: {categories}")

    for category in categories:
        category_path = os.path.join(full_data_dir, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            if os.path.isfile(file_path):
                data.append({
                    "image_path": file_path,
                    "label": category
                })

    df = pd.DataFrame(data)
    logger.info(f"DataFrame created with {df.shape[0]} rows and {df.shape[1]} columns.")
    logger.info("Image DataFrame building completed successfully.")

    if show_summary:
        summarize_image_counts(df)

    return df


def summarize_image_counts(df: pd.DataFrame) -> None:
    """
    Log detailed statistics about image distribution per category
    based on the DataFrame produced by `build_image_dataframe`.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels.
                           Must include columns 'image_path' and 'label'.
    """
    logger = setup_logger(__name__)

    if not {"image_path", "label"}.issubset(df.columns):
        raise ValueError("The DataFrame must contain 'image_path' and 'label' columns.")

    logger.info("Starting image distribution summary...")

    counts = df["label"].value_counts().sort_index()
    total_images = len(df)

    logger.info(f"Detected {len(counts)} categories with a total of {total_images} images.")

    for label, count in counts.items():
        percentage = (count / total_images) * 100
        logger.info(f"Category '{label}': {count} images ({percentage:.2f}% of total)")

    logger.info("Image count summary completed successfully.")
