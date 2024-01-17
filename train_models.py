import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import yaml

from src.datasets.detection_dataset import DetectionDataset
from src.models import models
from src.trainer import GDTrainer
from src.commons import set_seed


def save_model(
    model: torch.nn.Module,
    model_dir: Union[Path, str],
    name: str,
) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{full_model_dir}/ckpt.pth")


def get_datasets(
    datasets_paths: List[Union[Path, str]],
    amount_to_use: Tuple[Optional[int], Optional[int]],
) -> Tuple[DetectionDataset, DetectionDataset]:
    data_train = DetectionDataset(
        asvspoof_path=datasets_paths[0],
        subset="train",
        reduced_number=amount_to_use[0],
        oversample=True,
    )
    data_test = DetectionDataset(
        asvspoof_path=datasets_paths[0],
        subset="test",
        reduced_number=amount_to_use[1],
        oversample=True,
    )

    return data_train, data_test


def train_nn(
    datasets_paths: List[Union[Path, str]],
    batch_size: int,
    epochs: int,
    device: str,
    config: Dict,
    model_dir: Optional[Path] = None,
    amount_to_use: Tuple[Optional[int], Optional[int]] = (None, None),
    config_save_path: str = "config",
) -> Tuple[str, str]:
    # ... (existing code)
    pass
    # Rest of the code


def main(args):
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    with open('E:\deepfake-whisper-features-main\config', "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    # fix all seeds
    set_seed(seed)

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_dir = Path(args.ckpt)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_nn(
        datasets_paths=[
            args.asv_path,
            args.wavefake_path,
            args.celeb_path,
            args.asv19_path,
        ],
        device=device,
        amount_to_use=(args.train_amount, args.test_amount),
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_dir=model_dir,
        config=config,
    )


def parse_args():
    # ... (existing code)


    if __name__ == "__main__":
        main(parse_args())
