from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .metadata import build_splits, load_metadata


def _limit_samples(df: pd.DataFrame, max_samples: int | None, random_state: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or max_samples >= len(df):
        return df.reset_index(drop=True)

    n_classes = int(df["label"].nunique())
    if n_classes == 0:
        return df.reset_index(drop=True)

    per_class = max(1, max_samples // n_classes)
    sampled_parts = []
    for _, class_df in df.groupby("label"):
        sampled_parts.append(class_df.sample(n=min(len(class_df), per_class), random_state=random_state))

    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else df.iloc[0:0].copy()

    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_dataset_splits(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds_cfg = config["dataset"]
    df = load_metadata(ds_cfg["metadata_file"], ds_cfg["root_dir"], ds_cfg["name"])
    df = _limit_samples(df, ds_cfg.get("max_samples"), config["seed"])

    if ds_cfg.get("use_official_split", False):
        # Use the split encoded in the directory structure (train / dev / test).
        # For LAV-DF this avoids source-video leakage: a source real clip and all
        # its fake siblings are guaranteed to stay in the same partition.
        split_map = ds_cfg.get("official_split_map", {"train": "train", "val": "dev", "test": "test"})
        df["_split"] = df["video_path"].map(lambda p: Path(p).parent.name)
        missing = df[~df["_split"].isin(split_map.values())]
        if len(missing):
            import warnings
            warnings.warn(
                f"use_official_split: {len(missing)} rows have unrecognised parent dirs "
                f"{missing['_split'].unique().tolist()} and will be dropped.",
                RuntimeWarning,
                stacklevel=2,
            )
        train_df = df[df["_split"] == split_map["train"]].drop(columns=["_split"]).reset_index(drop=True)
        val_df   = df[df["_split"] == split_map["val"]  ].drop(columns=["_split"]).reset_index(drop=True)
        test_df  = df[df["_split"] == split_map["test"] ].drop(columns=["_split"]).reset_index(drop=True)
    else:
        train_df, val_df, test_df = build_splits(
            df,
            ds_cfg["split"],
            random_state=config["seed"],
            balance_classes=ds_cfg.get("balance_classes", True),
        )

    if ds_cfg.get("train_on_all", False):
        train_df = df.sample(frac=1.0, random_state=config["seed"]).reset_index(drop=True)
    return train_df, val_df, test_df


def iter_kfold(train_df: pd.DataFrame, config: Dict) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    fold_cfg = config["dataset"]["k_fold"]
    if not fold_cfg.get("enabled", False):
        yield train_df, train_df
        return

    skf = StratifiedKFold(
        n_splits=fold_cfg["n_splits"],
        shuffle=True,
        random_state=fold_cfg.get("random_state", 42),
    )

    y = train_df["label"].values
    for train_idx, val_idx in skf.split(train_df, y):
        yield train_df.iloc[train_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)
