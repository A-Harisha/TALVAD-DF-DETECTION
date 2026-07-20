from multiprocessing import freeze_support
from pathlib import Path

import yaml

from datasets.builder import build_dataset_splits
from features.parallel_extract import parallel_extract_features


def main() -> None:
    cfg = yaml.safe_load(Path('configs/lavdf_full_train.yaml').read_text())
    train_df, val_df, _ = build_dataset_splits(cfg)
    print('Train rows:', len(train_df))
    sub = train_df.head(5)
    print('Starting extract...')
    res = parallel_extract_features(
        sub,
        cfg,
        work_dir=Path('2c335981f605/preprocessed'),
        cache_dir=Path('data/cache/2c335981f605_smoke'),
        sequence_length=cfg['training']['sequence_length'],
        n_workers=2,
        checkpoint_every=2,
        show_progress=True,
        log_dir=Path('logs/test_extract'),
    )
    print('Smoke result:', res)


if __name__ == '__main__':
    freeze_support()
    main()
