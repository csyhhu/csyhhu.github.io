"""
DDPM Main Program Entry Point
Supports training, inference, and sampling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from model import DDPM, UNet, DDPMSchedule
from utils import print_model_summary, set_seed, get_best_checkpoint, get_data_loaders


def train_epoch(ddpm: DDPM, train_loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch"""
    ddpm.model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for images, _ in progress_bar:
        images = images.to(device)
        loss = ddpm.train_step(images, optimizer)
        total_loss += loss
        progress_bar.set_postfix({'loss': f'{loss:.4f}'})
    
    return total_loss / len(train_loader)


def validate(ddpm: DDPM, val_loader: DataLoader, device: torch.device) -> float:
    """Validate model"""
    ddpm.model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            batch_size = images.size(0)
            
            t = torch.randint(0, ddpm.schedule.num_timesteps, (batch_size,), device=device)
            xt, noise = ddpm.schedule.add_noise(images, t)
            noise_pred = ddpm.model(xt, t)
            
            loss = nn.functional.mse_loss(noise_pred, noise)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_mode(args):
    """Training mode"""
    print(f"Using device: {args.device}")
    device = torch.device(args.device)
    set_seed(args.seed)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, val_loader, channels, img_size = get_data_loaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"✓ Dataset loaded (Training: {len(train_loader) * args.batch_size}, Validation: {len(val_loader) * args.batch_size})")
    
    # Create model
    print("\nCreating DDPM model...")
    model = UNet(
        in_channels=channels,
        out_channels=channels,
        channels=(64, 128, 256),
        num_res_blocks=2,
        time_dim=256
    )
    schedule = DDPMSchedule(num_timesteps=args.num_timesteps)
    ddpm = DDPM(model, schedule, device)
    print_model_summary(model)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': [],
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\nStarting training ({args.epochs} epochs)...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(ddpm, train_loader, optimizer, device)
        val_loss = validate(ddpm, val_loader, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Best model saved")
        
        if (epoch + 1) % args.sample_every == 0:
            print(f"Sampling images (using DDPM)...")
            samples, _ = ddpm.sample_with_progress(
                batch_size=4,
                img_size=img_size,
                channels=channels,
                method='ddpm',
                num_steps=1000,
                eta=0.0,
                progress_interval=200
            )
            sample_path = checkpoint_dir / f'samples_epoch_{epoch + 1}.pth'
            torch.save(samples, sample_path)
            print(f"✓ Sampling complete")
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    
    history_path = checkpoint_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {history_path}")
    
    print("\nGenerating final samples (using DDPM)...")
    samples, trajectory = ddpm.sample_with_progress(
        batch_size=8,
        img_size=img_size,
        channels=channels,
        method='ddpm',
        num_steps=1000,
        eta=0.0,
        progress_interval=100
    )
    
    final_samples_path = checkpoint_dir / 'final_samples.pth'
    torch.save(samples, final_samples_path)
    print(f"✓ Final samples saved to {final_samples_path}")


def infer_mode(args):
    """Inference mode"""
    print(f"Using device: {args.device}")
    device = torch.device(args.device)
    
    # Get checkpoint
    checkpoint_path = get_best_checkpoint(args.checkpoint_dir)
    if checkpoint_path is None:
        print(f"✗ No model checkpoint found in {args.checkpoint_dir}")
        return
    
    print(f"Loading model: {checkpoint_path}")
    
    # Create model
    model = UNet(
        in_channels=3,
        out_channels=3,
        channels=(64, 128, 256),
        num_res_blocks=2,
        time_dim=256
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    schedule = DDPMSchedule(num_timesteps=1000)
    ddpm = DDPM(model, schedule, device)
    
    # Sampling method info
    print(f"\n{'='*60}")
    print(f"Sampling Configuration")
    print(f"{'='*60}")
    print(f"Method: {args.sample_method.upper()}")
    
    if args.sample_method.lower() == 'ddpm':
        print(f"Steps: 1000 (complete)")
        num_steps_info = 1000
    else:  # ddim
        print(f"Steps: {args.ddim_steps}")
        print(f"Eta (stochasticity): {args.eta}")
        num_steps_info = args.ddim_steps
    
    print(f"Number of samples: {args.num_samples}")
    print(f"{'='*60}\n")
    
    print(f"Generating {args.num_samples} samples using {args.sample_method.upper()}...")
    samples = ddpm.sample(
        batch_size=args.num_samples,
        img_size=32,
        channels=3,
        method=args.sample_method,
        num_steps=args.ddim_steps if args.sample_method.lower() == 'ddim' else 1000,
        eta=args.eta
    )
    
    save_path = Path(args.checkpoint_dir) / f'inference_samples_{args.sample_method}.pth'
    torch.save(samples, save_path)
    print(f"✓ Samples saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='DDPM - Denoising Diffusion Probabilistic Models')
    parser.add_argument('mode', type=str, choices=['train', 'infer'], help='Run mode: train or infer')
    
    # Common arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computing device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Training arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset: cifar10 or mnist')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--sample_every', type=int, default=5, help='Sample every N epochs')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Diffusion timesteps')
    
    # Inference arguments
    parser.add_argument('--num_samples', type=int, default=8, help='Number of samples to generate')
    parser.add_argument('--sample_method', type=str, default='ddpm', choices=['ddpm', 'ddim'],
                        help='Sampling method: ddpm or ddim')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of steps for DDIM sampling')
    parser.add_argument('--eta', type=float, default=0.0, help='Stochasticity coefficient for DDIM (0=deterministic)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'infer':
        infer_mode(args)


if __name__ == "__main__":
    main()
