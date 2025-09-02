"""
Episodic dataset and dataloaders for In-Context Learning (ICL).

Groups epochs by night (metadata['file']) and creates episodes with K support 
and M query examples for prototype-based learning.
"""

import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EpisodicDataset(Dataset):
    """
    Dataset that creates episodes from a base dataset grouped by nights.
    
    Each episode contains:
    - support_x: [K, ch, 30, 200] support examples
    - support_y: [K] support labels  
    - query_x: [M, ch, 30, 200] query examples
    - query_y: [M] query labels
    """
    
    def __init__(self, base_ds, indices: List[int], k: int, m: int, balance_support: bool = False):
        """
        Args:
            base_ds: Base dataset with .metadata attribute
            indices: List of indices to use from base_ds
            k: Number of support examples per episode
            m: Number of query examples per episode  
            balance_support: Whether to balance support set by class
        """
        self.base_ds = base_ds
        self.k = k
        self.m = m
        self.balance_support = balance_support
        
        # Group indices by night (metadata['file'])
        self.nights_data = defaultdict(list)
        for idx in indices:
            night_id = base_ds.metadata[idx]['file']
            self.nights_data[night_id].append(idx)
        
        # Filter nights that have at least k+m examples
        self.valid_nights = []
        for night_id, night_indices in self.nights_data.items():
            if len(night_indices) >= k + m:
                self.valid_nights.append(night_id)
        
        print(f"📊 EpisodicDataset: {len(self.valid_nights)} valid nights out of {len(self.nights_data)} total")
        print(f"   K={k}, M={m}, balance_support={balance_support}")
        
        if len(self.valid_nights) == 0:
            raise ValueError(f"No nights have enough data (need {k+m}, max available: {max([len(v) for v in self.nights_data.values()]) if self.nights_data else 0})")
    
    def __len__(self):
        return len(self.valid_nights)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Create an episode for the given night.
        
        Returns:
            Dictionary with keys: support_x, support_y, query_x, query_y
        """
        night_id = self.valid_nights[idx]
        night_indices = self.nights_data[night_id].copy()
        random.shuffle(night_indices)
        
        # Get data and labels for this night
        night_data = []
        night_labels = []
        for idx in night_indices:
            sample = self.base_ds[idx]
            # Handle different return formats (x,y) or (x,y,sid)
            if len(sample) == 2:
                x, y = sample
            elif len(sample) == 3:
                x, y, _ = sample  # Ignore subject ID
            else:
                raise ValueError(f"Unexpected sample format: {len(sample)} values")
            night_data.append(x)
            night_labels.append(y)
        
        night_data = torch.stack(night_data)  # [N, ch, 30, 200]
        night_labels = torch.tensor(night_labels)  # [N]
        
        # Create support set
        if self.balance_support and len(night_indices) >= self.k:
            support_indices = self._balanced_sample(night_labels, self.k)
        else:
            support_indices = list(range(min(self.k, len(night_indices))))
        
        # Create query set from remaining indices
        remaining_indices = [i for i in range(len(night_indices)) if i not in support_indices]
        
        # If we don't have enough remaining for M queries, recycle some data
        if len(remaining_indices) < self.m:
            # Repeat indices to fill M queries
            query_indices = (remaining_indices * ((self.m // len(remaining_indices)) + 1))[:self.m]
        else:
            query_indices = remaining_indices[:self.m]
        
        support_x = night_data[support_indices]  # [K, ch, 30, 200]
        support_y = night_labels[support_indices]  # [K]
        query_x = night_data[query_indices]  # [M, ch, 30, 200]
        query_y = night_labels[query_indices]  # [M]
        
        return {
            'support_x': support_x,
            'support_y': support_y, 
            'query_x': query_x,
            'query_y': query_y,
            'night_id': night_id
        }
    
    def _balanced_sample(self, labels: torch.Tensor, k: int) -> List[int]:
        """
        Sample K indices with balanced class distribution using round-robin.
        
        Args:
            labels: Label tensor for the night
            k: Number of samples to select
            
        Returns:
            List of selected indices
        """
        # Group indices by class
        class_indices = defaultdict(list)
        for i, label in enumerate(labels):
            class_indices[int(label)].append(i)
        
        # Remove empty classes
        class_indices = {c: indices for c, indices in class_indices.items() if indices}
        
        if not class_indices:
            return list(range(min(k, len(labels))))
        
        selected = []
        classes = list(class_indices.keys())
        
        # Round-robin sampling across classes
        while len(selected) < k and any(class_indices.values()):
            for class_id in classes:
                if len(selected) >= k:
                    break
                if class_indices[class_id]:
                    idx = class_indices[class_id].pop(0)
                    selected.append(idx)
        
        # If still need more samples, randomly fill from all available
        if len(selected) < k:
            all_available = [i for indices in class_indices.values() for i in indices]
            remaining_needed = k - len(selected)
            if all_available:
                additional = random.sample(all_available, min(remaining_needed, len(all_available)))
                selected.extend(additional)
        
        return selected


def episodic_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for episodic batches.
    
    Args:
        batch: List of episode dictionaries
        
    Returns:
        Batched episode dictionary
    """
    if len(batch) == 1:
        return batch[0]
    
    # Stack tensors across episodes
    collated = {}
    for key in ['support_x', 'support_y', 'query_x', 'query_y']:
        collated[key] = torch.stack([episode[key] for episode in batch])
    
    # Keep night_ids as list
    collated['night_id'] = [episode['night_id'] for episode in batch]
    
    return collated


def make_episodic_loaders(
    base_ds,
    train_idx: List[int],
    val_idx: List[int], 
    test_idx: List[int],
    k: int,
    m: int,
    balance_support: bool,
    batch_episodes: int = 4,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create episodic dataloaders for train/val/test splits.
    
    Args:
        base_ds: Base dataset  
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        k: Support examples per episode
        m: Query examples per episode
        balance_support: Balance support by class
        batch_episodes: Episodes per batch (train/val), test uses batch_size=1
        num_workers: Number of dataloader workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create episodic datasets
    train_episodes = EpisodicDataset(base_ds, train_idx, k, m, balance_support)
    val_episodes = EpisodicDataset(base_ds, val_idx, k, m, balance_support) 
    test_episodes = EpisodicDataset(base_ds, test_idx, k, m, balance_support)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_episodes,
        batch_size=batch_episodes,
        shuffle=True,
        collate_fn=episodic_collate,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_episodes,
        batch_size=batch_episodes,
        shuffle=False, 
        collate_fn=episodic_collate,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test loader uses batch_size=1 for per-night evaluation
    test_loader = DataLoader(
        test_episodes,
        batch_size=1,
        shuffle=False,
        collate_fn=episodic_collate, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader