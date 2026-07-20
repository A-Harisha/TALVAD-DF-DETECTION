from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitConfig:
    train: float
    val: float
    test: float


def load_metadata(metadata_file: str | Path, root_dir: str | Path, dataset_name: str) -> pd.DataFrame:
    """Load normalized metadata for LAV-DF, DFDC, and FF++ datasets."""
    metadata_file = Path(metadata_file)
    if not metadata_file.exists():
        raise FileNotFoundError(
            "Metadata CSV not found: "
            f"{metadata_file}. Generate one with python -m datasets.prepare_metadata --root <dataset_root> --output <metadata_csv> --dataset {dataset_name}"
        )

    df = pd.read_csv(metadata_file)

    if "video_path" not in df.columns:
        for alt_col in ["path", "filepath", "file", "video"]:
            if alt_col in df.columns:
                df = df.rename(columns={alt_col: "video_path"})
                break

    if "label" not in df.columns:
        for alt_col in ["target", "is_fake", "class"]:
            if alt_col in df.columns:
                df = df.rename(columns={alt_col: "label"})
                break

    df["video_path"] = df["video_path"].astype(str)
    df["video_path"] = df["video_path"].apply(lambda p: str(Path(root_dir) / p) if not Path(p).is_absolute() else p)

    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"real": 0, "fake": 1}).fillna(df["label"]).astype(int)

    if "dataset" not in df.columns:
        df["dataset"] = dataset_name

    exists_mask = df["video_path"].map(lambda p: Path(p).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count:
        warnings.warn(
            f"Dropped {missing_count} metadata rows with missing video files from {metadata_file}.",
            RuntimeWarning,
            stacklevel=2,
        )
    df = df.loc[exists_mask].reset_index(drop=True)

    return df


def _balance_binary(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    counts = df["label"].value_counts()
    if len(counts) != 2:
        return df

    min_count = counts.min()
    balanced_parts = []
    for _, class_df in df.groupby("label"):
        balanced_parts.append(class_df.sample(min_count, random_state=random_state))

    balanced = pd.concat(balanced_parts, ignore_index=True)
    return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_splits(
    df: pd.DataFrame,
    split_cfg: Dict[str, float],
    random_state: int = 42,
    balance_classes: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if balance_classes:
        df = _balance_binary(df, random_state=random_state)

    train_df, holdout_df = train_test_split(
        df,
        test_size=split_cfg["val"] + split_cfg["test"],
        random_state=random_state,
        stratify=df["label"],
    )

    rel_test = split_cfg["test"] / (split_cfg["val"] + split_cfg["test"])
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=rel_test,
        random_state=random_state,
        stratify=holdout_df["label"],
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
