"""
Utility Functions - Essential helper functions only
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


def count_parameters(model: nn.Module) -> int:
    """Count total model parameters"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> dict:
    """Get model summary information"""
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    memory_mb = (total_params * 4) / (1024 ** 2)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'memory_mb': memory_mb
    }


def print_model_summary(model: nn.Module):
    """Print model summary"""
    summary = get_model_summary(model)
    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"Total Parameters:        {summary['total_parameters']:,}")
    print(f"Trainable Parameters:    {summary['trainable_parameters']:,}")
    print(f"Non-trainable Parameters: {summary['non_trainable_parameters']:,}")
    print(f"Model Size (float32):    {summary['memory_mb']:.2f} MB")
    print("=" * 50)


class EMA:
    """Exponential Moving Average (for stable training)"""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """Apply shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def set_seed(seed: int):
    """Set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get latest checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if not checkpoints:
        return None
    
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def get_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get best checkpoint"""
    checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
    if checkpoint_path.exists():
        return str(checkpoint_path)
    return None


def get_data_loaders(
    dataset_name: str = 'cifar10',
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Get data loaders
    
    Args:
        dataset_name: Dataset name ('cifar10' or 'mnist')
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, channels, img_size
    """
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        channels, img_size = 3, 32
    elif dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        channels, img_size = 3, 28
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, channels, img_size
