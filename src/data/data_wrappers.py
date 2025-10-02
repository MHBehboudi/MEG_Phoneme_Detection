# src/data/data_wrappers.py

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Tuple, Optional
from pnpl.datasets import LibriBrainPhoneme, LibriBrainCompetitionHoldout

# Import constants
from config.default_config import (
    NUM_PHONEME_CLASSES, TARGET_EPOCH_SAMPLES, NUM_WORKERS,
    DROP_REMAINING, WRAPPER_STANDARDIZE, BASE_PATH
)

# ----------------------------
# Small utils
# ----------------------------
def per_sample_z(x: torch.Tensor, eps=1e-5):
    # x: (B,C,T) -> z-score per (B,C) along T
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True)
    return (x - mu) / (sd + eps)

def _standardize_single(x: torch.Tensor, eps=1e-5) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True)
    return (x - mu) / (sd + eps)


# ----------------------------
# DATA WRAPPERS
# ----------------------------

class LabelAveragingDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, group_size: int, drop_remaining: bool, standardize: bool):
        super().__init__()
        self.base = base_ds
        self.K = group_size
        self.drop_remaining = drop_remaining
        self.standardize = standardize
        self.label_to_idxs = [[] for _ in range(NUM_PHONEME_CLASSES)]
        print("[Index] Grouping by label for label-averaging...")
        for i in range(len(self.base)):
            _, y = self.base[i]
            self.label_to_idxs[int(y)].append(i)
        self._build_groups()
    def _build_groups(self):
        self.groups = []
        for y, idxs in enumerate(self.label_to_idxs):
            if len(idxs) == 0: continue
            for s in range(0, len(idxs), self.K):
                chunk = idxs[s:s+self.K]
                if len(chunk) < self.K and self.drop_remaining: continue
                self.groups.append((y, chunk))
        self.groups.sort(key=lambda t: t[0])
    def __len__(self): return len(self.groups)
    def __getitem__(self, gi: int) -> Tuple[torch.Tensor, int]:
        y, idxs = self.groups[gi]
        x_sum = None; count = 0
        for idx in idxs:
            x, y2 = self.base[idx]; assert int(y2)==int(y)
            x = x.float().cpu()
            if self.standardize: x = _standardize_single(x)
            x_sum = x if x_sum is None else x_sum + x
            count += 1
        x_avg = x_sum / max(count, 1)
        return x_avg, int(y)

class MultiKLabelAveragingDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, k_min: int, k_max: int, k_step: int, drop_remaining: bool, standardize: bool):
        super().__init__()
        self.base = base_ds
        self.k_list = [k for k in range(k_min, k_max+1, k_step)]
        self.drop_remaining = drop_remaining
        self.standardize = standardize
        self.label_to_idxs = [[] for _ in range(NUM_PHONEME_CLASSES)]
        print("[Index] Grouping by label for label-averaging...")
        for i in range(len(self.base)):
            _, y = self.base[i]
            self.label_to_idxs[int(y)].append(i)
        self._build_groups()
    def _build_groups(self):
        self.groups = []
        for K in self.k_list:
            for y, idxs in enumerate(self.label_to_idxs):
                if len(idxs)==0: continue
                for s in range(0, len(idxs), K):
                    chunk = idxs[s:s+K]
                    if len(chunk)<K and self.drop_remaining: continue
                    self.groups.append((y, chunk))
        self.groups.sort(key=lambda t: (t[0], len(t[1])))
    def __len__(self): return len(self.groups)
    def __getitem__(self, gi: int) -> Tuple[torch.Tensor, int]:
        y, idxs = self.groups[gi]
        x_sum = None; count = 0
        for idx in idxs:
            x, y2 = self.base[idx]; assert int(y2)==int(y)
            x = x.float().cpu()
            if self.standardize: x = _standardize_single(x)
            x_sum = x if x_sum is None else x_sum + x
            count += 1
        x_avg = x_sum / max(count, 1)
        return x_avg, int(y)

class PassDataset(torch.utils.data.Dataset):
    """For classifier: produces (x, empty, y) to match Lightning step signature."""
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, torch.empty(0), y

class ValSingleDataset(torch.utils.data.Dataset):
    """Validation K=1 passthrough (x,y) -> we will wrap to (x, empty, y)."""
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, torch.empty(0), y

def _balanced_sampler_for_groups(groups_labels: np.ndarray, class_counts: np.ndarray, target_samples: Optional[int]):
    class_counts_safe = np.clip(class_counts, 1, None)
    sample_weights = np.array([1.0 / class_counts_safe[y] for y in groups_labels], dtype=np.float64)
    num_draws = len(sample_weights) if target_samples is None else target_samples
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=num_draws,
        replacement=True
    )

def get_labels_and_counts(dataset: torch.utils.data.Dataset) -> np.ndarray:
    counts = np.zeros(NUM_PHONEME_CLASSES, dtype=np.int64)
    for i in range(len(dataset)):
        # Assuming dataset[i] returns (data, label)
        _, y = dataset[i]
        counts[int(y)] += 1
    return counts

def build_loaders(base_path, batch_size, train_mode="fixed", k_value=32,
                  multik_min=60, multik_max=100, multik_step=10,
                  val_k=100, drop_remaining=DROP_REMAINING, standardize_wrapper=WRAPPER_STANDARDIZE,
                  per_sample_zscore_val_avg=False, per_sample_zscore_val_single=False):
    # Raw datasets
    train_raw = LibriBrainPhoneme(data_path=f"{base_path}/data/", partition="train", tmin=0.0, tmax=0.5, standardize=True)
    val_raw   = LibriBrainPhoneme(data_path=f"{base_path}/data/", partition="validation", tmin=0.0, tmax=0.5, standardize=True)

    # Averaged train
    if train_mode == "fixed":
        train_avg = LabelAveragingDataset(train_raw, group_size=k_value, drop_remaining=drop_remaining, standardize=standardize_wrapper)
    elif train_mode == "multik":
        train_avg = MultiKLabelAveragingDataset(train_raw, k_min=multik_min, k_max=multik_max, k_step=multik_step,
                                                drop_remaining=drop_remaining, standardize=standardize_wrapper)
    else:
        raise ValueError(f"Unknown train_mode={train_mode}")
        
    # Averaged val (K=100)
    val_avg = LabelAveragingDataset(val_raw, group_size=val_k, drop_remaining=drop_remaining, standardize=standardize_wrapper)
    # Single val (K=1)
    val_single = ValSingleDataset(val_raw)

    # Class counts & sampler
    class_counts = get_labels_and_counts(train_avg)
    y_all = np.array([train_avg.groups[i][0] for i in range(len(train_avg))], dtype=np.int64)
    sampler = _balanced_sampler_for_groups(y_all, class_counts, TARGET_EPOCH_SAMPLES)

    train_ds = PassDataset(train_avg)
    val_avg_ds    = PassDataset(val_avg)
    val_single_ds = val_single

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_avg_loader = DataLoader(val_avg_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_single_loader = DataLoader(val_single_ds, batch_size=batch_size, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    return {
        "train": train_loader,
        "val_avg": val_avg_loader,
        "val_single": val_single_loader,
        "class_counts": class_counts,
        "val_flags": {
            "z_avg": per_sample_zscore_val_avg,
            "z_single": per_sample_zscore_val_single
        }
    }
