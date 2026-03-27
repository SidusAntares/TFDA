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
    AddPixelLabels
)
from torchvision import transforms

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


def shape_adjust(batch_dict):
    pixels = batch_dict['pixels']  # [B, T, C, N]
    valid_pixels = batch_dict['valid_pixels']  # [B, T, N] - 注意这个维度！
    pixel_labels = batch_dict['pixel_labels']  # [B, N]
    doy = batch_dict['positions']

    B, T, C, N = pixels.shape

    doy = doy.repeat_interleave(N, dim=0)  # (B*N, T)

    # --- Step 1: 展平所有样本 ---
    # [B, T, C, N] -> [S, C, T] where S = B * N
    x_flat = pixels.permute(0, 3, 2, 1).reshape(-1, C, T)
    y_flat = pixel_labels.reshape(-1)  # (S,)
    valid_flat = valid_pixels.permute(0, 2, 1).reshape(-1, T)  # (S, T)

    # --- Step 2: 【关键修复】处理全无效时间步样本 ---
    has_valid_time = valid_flat.any(dim=1)  # (S,)
    all_invalid_mask = ~has_valid_time  # (S,)

    if all_invalid_mask.any():
        count = all_invalid_mask.sum().item()
        print(f"⚠️ Fixing {count} all-invalid samples (forcing t=0 valid).")
        # 强制将第一个时间步标记为有效
        valid_flat[all_invalid_mask, 0] = 1.0
    x = x_flat.clone()

    x_np = x.cpu().numpy() if torch.is_tensor(x) else x
    if np.isnan(x_np).any() or np.isinf(x_np).any():
        print(f"DEBUG: Shape Adjust - Found NaN/Inf! Replacing with 0.")
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)

    y_np = y_flat.cpu().numpy() if torch.is_tensor(y_flat) else y_flat

    return x_np, y_np, doy


def get_data_loaders(splits, config, balance_source=True):
    """Creates and returns the training DataLoader."""
    strong_aug = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        RandomSampleTimeSteps(config.seq_length),
        Normalize(),
        ToTensor(),
        AddPixelLabels()
    ])

    source_dataset = PixelSetData(
        config.data_root, config.source, config.classes, strong_aug,
        indices=splits[config.source]['train'],
    )

    if balance_source:
        source_labels = source_dataset.get_labels()
        from collections import Counter
        freq = Counter(source_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_labels]
        sampler = torch.utils.data.WeightedRandomSampler(source_weights, len(source_labels))
        print("Using balanced loader for source")
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    else:
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
    print(f'Size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)')
    return source_loader


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
    x_temp, y_temp, doy_temp = [], [], []
    first_batch = True
    for batch in tqdm(data_loader, desc="Processing batches"):
        x, y, doy = shape_adjust(batch)

        x_temp.append(x)
        y_temp.append(y)
        doy_temp.append(doy)

        # 控制列表长度，避免内存溢出
        if len(x_temp) > 10:  # 每100个batch合并一次
            x_temp = np.vstack(x_temp)
            doy_temp = np.vstack(doy_temp)
            y_temp = np.hstack(y_temp)
            if first_batch:
                x_res, y_res, doy_res = x_temp, y_temp, doy_temp
                first_batch = False
            else:
                x_res = np.vstack([x_res, x_temp])
                y_res = np.hstack([y_res, y_temp])
                doy_res = np.vstack([doy_res, doy_temp])

            x_temp, y_temp, doy_temp = [], [], []  # 清空临时列表

    # 处理剩余的数据
    if x_temp:
        x_temp = np.vstack(x_temp)
        y_temp = np.hstack(y_temp)
        doy_temp = np.vstack(doy_temp)
        if first_batch:
            x_res, y_res, doy_res = x_temp, y_temp, doy_temp
            first_batch = False
        else:
            x_res = np.vstack([x_res, x_temp])
            y_res = np.hstack([y_res, y_temp])
            doy_res = np.vstack([doy_res, doy_temp])
    assert x_res is not None, "dataloader is empty"
    return x_res, y_res, doy_res


def save(folder, name, x, y, mode):
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray),print("save function only accept numpy arrays")
    folder = os.path.join(folder, name)
    os.makedirs(folder, exist_ok=True)
    if not os.path.exists(os.path.join(folder,f"{mode}_{name}.pt")):
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.int64))
        torch.save({
            "samples": x,
            "labels": y
        }, os.path.join(folder, f"{mode}_{name}.pt"))


def train(config):
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

    indices = {config.source: len(source_data),
               config.target: len(PixelSetData(config.data_root, config.target, source_classes))}
    folds = create_train_val_test_folds([config.source, config.target], config.num_folds, indices, config.val_ratio,
                                        config.test_ratio)
    splits = folds[0]
    source_loader = get_data_loaders(splits, config, config.balance_source)
    val_loader, test_loader = create_evaluation_loaders(config.target, splits, config, config.sample_pixels_val)
    x_train, y_train, _ = data_collect(source_loader)
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    x_test, y_test, _ = data_collect(test_loader)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    source_name = match(args.source)
    target_name = match(args.target)
    folder = "processed_data"
    save(folder, source_name, x_train, y_train, "train")
    save(folder, target_name, x_test, y_test, "test")








if __name__ == "__main__":
    # timematch
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Device to use (e.g., cuda:0, cpu). Auto-detected if not specified.')
    parser.add_argument('--per', default=1, type=float,
                        help='Percentage of labeled samples to use for training/validation.')
    parser.add_argument('--seed', default=111, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for data loading.')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training.')
    parser.add_argument('--balance_source', type=bool_flag, default=True, help='Use class balanced batches for source.')
    parser.add_argument('--num_pixels', default=1, type=int, help='Number of pixels to sample from the input sample.')
    parser.add_argument('--seq_length', default=30, type=int,
                        help='Number of time steps to sample from the input sample.')
    parser.add_argument('--data_root', default='/mnt/d/All_Documents/documents/ViT/dataset/timematch', type=str,
                        help='Path to datasets root directory.')

    parser.add_argument('--source', default='france/31TCJ/2017', type=str, help='Source domain.')
    parser.add_argument('--target', default='france/31TCJ/2017', type=str)

    parser.add_argument('--combine_spring_and_winter', action='store_true', help='Combine spring and winter classes.')
    parser.add_argument('--num_folds', default=1, type=int, help='Number of cross-validation folds.')
    parser.add_argument("--val_ratio", default=0.1, type=float, help='Validation ratio.')
    parser.add_argument("--test_ratio", default=0.2, type=float, help='Test ratio.')
    parser.add_argument('--sample_pixels_val', action='store_true', help='Sample pixels during validation.')
    args = parser.parse_args()
    # args.source = args.target

    train(args)
