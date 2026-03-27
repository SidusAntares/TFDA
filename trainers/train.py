import sys

sys.path.append('/home/furqon/TFDA')

import os
import pandas as pd

import collections
import argparse
import warnings
import sklearn.exceptions

from utils import fix_randomness, starting_logs, AverageMeter # plot_tsne
from abstract_trainer import AbstractTrainer

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

class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # Logging
        self.args = args
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def train(self):

        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)
        
        config = self.args
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
        
        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                # Average meters
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)

                # Train model
                src_FE, src_Classifier, tgt_last_adapted_FE, tgt_last_adapted_Classifier, tgt_best_adapted_FE, tgt_best_adapted_Classifier, non_adapted_model, last_adapted_model, best_adapted_model = self.train_model()
                
                # Save checkpoint
                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model, last_adapted_model,
                                     best_adapted_model)

                # Calculate risks and metrics
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')


if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs13', type=str,
                        help='Directory containing all experiments')
    parser.add_argument('-run_description', default=None, type=str, help='Description of run, if none, DA method name will be used')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='TFDA', type=str, help='SHOT, AaD, NRC, MAPU,')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=3, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--num_neighbors', default=10, type=int)
    parser.add_argument('--temporal_length', default=5, type=int)
    parser.add_argument('--plot_tsne', default=True, type=bool, help='Plot t-sne for training and testing or not?')

    # timematch
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use (e.g., cuda:0, cpu). Auto-detected if not specified.')
    parser.add_argument('--per', default= 1, type=float, help='Percentage of labeled samples to use for training/validation.')
    parser.add_argument('--seed', default=111, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for data loading.')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training.')
    parser.add_argument('--balance_source', type=bool_flag, default=True, help='Use class balanced batches for source.')
    parser.add_argument('--num_pixels', default=2, type=int, help='Number of pixels to sample from the input sample.')
    parser.add_argument('--seq_length', default=30, type=int, help='Number of time steps to sample from the input sample.')
    parser.add_argument('--data_root', default='/mnt/d/All_Documents/documents/ViT/dataset/timematch', type=str, help='Path to datasets root directory.')

    parser.add_argument('--source', default='france/31TCJ/2017', type=str, help='Source domain.')
    parser.add_argument('--target', default='france/31TCJ/2017', type=str)

    parser.add_argument('--combine_spring_and_winter', action='store_true', help='Combine spring and winter classes.')
    parser.add_argument('--num_folds', default=1, type=int, help='Number of cross-validation folds.')
    parser.add_argument("--val_ratio", default=0.1, type=float, help='Validation ratio.')
    parser.add_argument("--test_ratio", default=0.2, type=float, help='Test ratio.')
    parser.add_argument('--sample_pixels_val', action='store_true', help='Sample pixels during validation.')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
   