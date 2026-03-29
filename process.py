import sys

from tqdm import tqdm

# sys.path.append('/home/furqon/TFDA')

import os
import pandas as pd

import collections
import argparse
import warnings
import sklearn.exceptions


# timematch
import numpy as np
import torch
from torch.utils import data
import random
from dataset import PixelSetData, create_evaluation_loaders
from timematch_utils import label_utils
from timematch_utils.train_utils import bool_flag
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    RandomTemporalShift
)
from torchvision import transforms

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


def shape_adjust(batch_dict):
    pixels = batch_dict['pixels']  # [B, T, C, N]
    pixel_labels = batch_dict['label']  # [B, N]
    doy = batch_dict['positions']

    B, T, C, N = pixels.shape

    doy = doy.repeat_interleave(N, dim=0)  # (B*N, T)

    # --- Step 1: 展平所有样本 ---
    # [B, T, C, N] -> [S, C, T] where S = B * N
    x_flat = pixels.permute(0, 3, 2, 1).reshape(-1, C, T)
    y_flat = pixel_labels.reshape(-1)  # (S,)
    x = x_flat.clone()

    x_np = x.cpu().numpy() if torch.is_tensor(x) else x
    y_np = y_flat.cpu().numpy() if torch.is_tensor(y_flat) else y_flat

    return x_np, y_np, doy

def get_data_loaders(splits, config):
    def create_data_loader(dataset):
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

    train_transform = transforms.Compose(
        [
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            RandomTemporalShift(max_shift=config.max_shift_aug, p=config.shift_aug_p) ,
            Normalize(),
            ToTensor(),
        ]
    )

    source_dataset = PixelSetData(
        config.data_root,
        config.source,
        config.classes,
        train_transform,
        indices=splits[config.source]["train"],
    )
    source_loader = create_data_loader(source_dataset)
    target_dataset = PixelSetData(
        config.data_root,
        config.target,
        config.classes,
        train_transform,
        indices=splits[config.target]["train"],
    )
    target_loader = create_data_loader(target_dataset)

    print(
        f"size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)"
    )
    print(
        f"size of target dataset: {len(target_dataset)} ({len(target_loader)} batches)"
    )

    return source_loader, target_loader



def create_train_val_test_folds(datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    """Creates train/val/test splits."""
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset in datasets:
            if isinstance(num_indices, dict):
                indices = list(range(num_indices[dataset]))
            else:
                indices = list(range(num_indices))
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val

            random.shuffle(indices)

            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train:n_train + n_val])
            test_indices = set(indices[-n_test:])
            assert set.intersection(train_indices, val_indices, test_indices) == set()
            assert len(train_indices) + len(val_indices) + len(test_indices) == n

            splits[dataset] = {'train': train_indices, 'val': val_indices, 'test': test_indices}
        folds.append(splits)
    return folds

def match(domain):
    match domain:
        case 'france/30TXT/2017':
            return 'FR1'
        case 'france/31TCJ/2017':
            return 'FR2'
        case 'denmark/32VNH/2017':
            return 'DK1'
    return 'AT1'


def data_collect(data_loader):
    x_parts, y_parts, doy_parts = [], [], []

    for batch in tqdm(data_loader, desc="Processing batches"):
        x, y, doy = shape_adjust(batch)
        x_parts.append(x)
        y_parts.append(y)
        doy_parts.append(doy)

    # 一次性拼接（比循环 vstack 内存效率高得多）
    x_res = np.concatenate(x_parts, axis=0)
    y_res = np.concatenate(y_parts, axis=0)
    doy_res = np.concatenate(doy_parts, axis=0)

    assert x_res is not None, "dataloader is empty"
    return x_res, y_res, doy_res


def save(folder, name, x, y, mode):
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray),print("save function only accept numpy arrays")
    os.makedirs(folder, exist_ok=True)
    x = torch.from_numpy(x.astype(np.float32))
    y = torch.from_numpy(y.astype(np.int64))
    torch.save({
        "samples": x,
        "labels": y
    }, os.path.join(folder, f"{mode}_{name}.pt"))


def train(config):

    source_name = match(args.source)
    print('source name:', source_name)
    target_name = match(args.target)
    print('target name:', target_name)

    # folder = os.path.join("processed_data", source_name)
    folder = os.path.join("processed_data", source_name)

    source_classes = label_utils.get_classes(
        config.source.split('/')[0],
        combine_spring_and_winter=config.combine_spring_and_winter
    )
    source_data = PixelSetData(config.data_root, config.source, source_classes)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    print('Using classes:', source_classes)
    config.classes = source_classes
    config.num_classes = len(source_classes)

    # indices = {config.source: len(source_data),
    #            config.target: len(PixelSetData(config.data_root, config.target, source_classes))}
    indices = {config.source: int(len(source_data)*0.1),
               config.target: int(len(PixelSetData(config.data_root, config.target, source_classes))*0.1)}

    folds = create_train_val_test_folds([config.source, config.target], config.num_folds, indices, config.val_ratio,
                                        config.test_ratio)
    splits = folds[0]

    src_train_loader, trg_train_loader = get_data_loaders(splits, config)

    _, source_test_loader = create_evaluation_loaders(config.target, splits, config, config.sample_pixels_val)
    src_x_train, src_y_train, doy_train = data_collect(src_train_loader)
    src_x_test, src_y_test, doy_test = data_collect(source_test_loader)
    print('src_x_train shape:', src_x_train.shape)
    print('src_y_train shape:', src_y_train.shape)
    save(folder, source_name, src_x_train, src_y_train, "train")
    save(folder, source_name, src_x_test, src_y_test, "test")
    del src_x_train, src_y_train ,doy_train
    del src_x_test, src_y_test, doy_test

    trg_val_loader, trg_test_loader = create_evaluation_loaders(config.target, splits, config, config.sample_pixels_val)

    trg_x_test, trg_y_test, _ = data_collect(trg_test_loader)
    trg_x_train, trg_y_train, _ = data_collect(trg_train_loader)
    print('trg_x_train shape:', trg_x_train.shape)
    print('x_test shape:', trg_x_test.shape)
    print('y_test shape:', trg_y_test.shape)
    save(folder, target_name, trg_x_train, trg_y_train, "train")
    save(folder, target_name, trg_x_test, trg_y_test, "test")









if __name__ == "__main__":
    # timematch
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Device to use (e.g., cuda:0, cpu). Auto-detected if not specified.')
    parser.add_argument('--per', default=1, type=float,
                        help='Percentage of labeled samples to use for training/validation.')
    parser.add_argument('--seed', default=111, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for data loading.')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training.')
    parser.add_argument('--balance_source', type=bool_flag, default=True, help='Use class balanced batches for source.')
    parser.add_argument('--num_pixels', default=1, type=int, help='Number of pixels to sample from the input sample.')
    parser.add_argument('--seq_length', default=30, type=int,
                        help='Number of time steps to sample from the input sample.')
    parser.add_argument('--data_root', default='/data/user/DBL/timematch_data', type=str,
                        help='Path to datasets root directory.')

    parser.add_argument('--source', default='france/31TCJ/2017', type=str, help='Source domain.')
    parser.add_argument('--target', default='france/31TCJ/2017', type=str)

    parser.add_argument('--combine_spring_and_winter', action='store_true', help='Combine spring and winter classes.')
    parser.add_argument('--num_folds', default=1, type=int, help='Number of cross-validation folds.')
    parser.add_argument("--val_ratio", default=0.1, type=float, help='Validation ratio.')
    parser.add_argument("--test_ratio", default=0.2, type=float, help='Test ratio.')
    parser.add_argument('--sample_pixels_val', action='store_true', help='Sample pixels during validation.')

    parser.add_argument('--with_shift_aug', default=True, action='store_true',
                        help='whether to apply random temporal shift augmentation')
    parser.add_argument('--shift_aug_p', default=1.0, type=float,
                        help='probability to apply temporal shift augmentation')
    parser.add_argument('--max_shift_aug', default=60, type=int,
                        help='highest shift to apply for temporal shift augmentation')
    args = parser.parse_args()
    # args.source = args.target
    args.sample_pixels_val = True
    train(args)
