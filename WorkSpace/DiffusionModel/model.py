"""
DDPM Model Architecture
Contains U-Net, temporal encoding, diffusion schedule and other core modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timesteps"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class ResidualBlock(nn.Module):
    """Residual block with temporal embedding"""
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_channels),
            nn.GELU()
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm1 = nn.GroupNorm(32, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        time_proj = self.time_mlp(time_emb)[:, :, None, None]
        
        h = F.gelu(self.group_norm1(self.conv1(x)))
        h = h + time_proj
        
        h = F.gelu(self.group_norm2(self.conv2(h)))
        
        return h + self.skip_connection(x)


class UNet(nn.Module):
    """U-Net architecture for DDPM noise prediction"""
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: Tuple[int, ...] = (64, 128, 256),
        num_res_blocks: int = 2,
        time_dim: int = 256
    ):
        super().__init__()
        
        self.channels = list(channels)
        self.num_res_blocks = num_res_blocks
        
        self.time_encoding = PositionalEncoding(time_dim)
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for ch in channels[1:]:
            down_block = nn.ModuleList()
            for _ in range(num_res_blocks):
                down_block.append(ResidualBlock(in_ch, ch, time_dim))
                in_ch = ch
            down_block.append(nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1))
            self.down_blocks.append(down_block)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.bottleneck.append(ResidualBlock(channels[-1], channels[-1], time_dim))
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        channels_for_up = list(reversed(channels))
        for i, ch in enumerate(channels_for_up[1:]):
            up_block = nn.ModuleList()
            up_block.append(nn.ConvTranspose2d(channels_for_up[i], ch, kernel_size=4, stride=2, padding=1))
            
            for _ in range(num_res_blocks):
                up_block.append(ResidualBlock(ch * 2, ch, time_dim))
            
            self.up_blocks.append(up_block)
        
        self.group_norm_out = nn.GroupNorm(32, channels[0])
        self.conv_out = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_encoding(t.float())
        
        h = self.conv_in(x)
        skips = [h]
        
        for down_block in self.down_blocks:
            for layer in down_block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
            skips.append(h)
        
        for layer in self.bottleneck:
            h = layer(h, time_emb)
        
        for i, up_block in enumerate(self.up_blocks):
            for j, layer in enumerate(up_block):
                if j == 0:
                    h = layer(h)
                elif isinstance(layer, ResidualBlock):
                    h = torch.cat([h, skips[-(i+2)]], dim=1)
                    h = layer(h, time_emb)
        
        h = F.gelu(self.group_norm_out(h))
        h = self.conv_out(h)
        
        return h


class DDPMSchedule:
    """DDPM diffusion process schedule"""
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear"
    ):
        self.num_timesteps = num_timesteps
        
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            s = 0.008
            steps = torch.arange(num_timesteps + 1)
            alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward process: add noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        while len(sqrt_alphas_cumprod_t.shape) < len(x0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return xt, noise
    
    def denoise_step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        xt: torch.Tensor
    ) -> torch.Tensor:
        """Reverse process: single denoising step"""
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        var = self.posterior_variance[t]
        
        while len(coef1.shape) < len(xt.shape):
            coef1 = coef1.unsqueeze(-1)
            coef2 = coef2.unsqueeze(-1)
            var = var.unsqueeze(-1)
        
        mean = coef1 * xt + coef2 * model_output
        
        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float()
        
        while len(nonzero_mask.shape) < len(noise.shape):
            nonzero_mask = nonzero_mask.unsqueeze(-1)
        
        xt_minus_1 = mean + nonzero_mask * torch.sqrt(var) * noise
        
        return xt_minus_1


class DDPM:
    """DDPM model training and inference interface"""
    def __init__(
        self,
        model: nn.Module,
        schedule: DDPMSchedule,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model.to(device)
        self.schedule = schedule
        self.device = device
    
    def train_step(
        self,
        x0: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Single training step"""
        batch_size = x0.size(0)
        x0 = x0.to(self.device)
        
        t = torch.randint(0, self.schedule.num_timesteps, (batch_size,), device=self.device)
        xt, noise = self.schedule.add_noise(x0, t)
        
        noise_pred = self.model(xt, t)
        loss = F.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _get_timesteps(self, method: str = 'ddpm', num_steps: int = 50) -> torch.Tensor:
        """Get timesteps based on sampling method"""
        if method.lower() == 'ddpm':
            # Use all timesteps in reverse order
            return torch.arange(self.schedule.num_timesteps - 1, -1, -1)
        elif method.lower() == 'ddim':
            # Use sparse timesteps uniformly sampled
            step_ratio = self.schedule.num_timesteps / num_steps
            timesteps = torch.round(torch.arange(0, self.schedule.num_timesteps, step_ratio)).long()
            return torch.flip(timesteps, dims=[0])
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        img_size: int = 32,
        channels: int = 3,
        method: str = 'ddpm',
        num_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Generate images using DDPM or DDIM
        
        Args:
            batch_size: Batch size
            img_size: Image size
            channels: Number of channels
            method: Sampling method ('ddpm' or 'ddim')
            num_steps: Number of steps for DDIM (ignored for DDPM)
            eta: Stochasticity coefficient for DDIM (0=deterministic, 1=DDPM-like)
        
        Returns:
            Generated images
        """
        self.model.eval()
        
        xt = torch.randn(batch_size, channels, img_size, img_size, device=self.device)
        timesteps = self._get_timesteps(method, num_steps)
        timesteps = timesteps.to(self.device)
        
        if method.lower() == 'ddpm':
            return self._sample_ddpm(xt, timesteps)
        elif method.lower() == 'ddim':
            return self._sample_ddim(xt, timesteps, eta)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    @torch.no_grad()
    def _sample_ddpm(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """DDPM sampling: iterate through all timesteps"""
        for t in timesteps:
            t_tensor = torch.full((xt.size(0),), t.item(), dtype=torch.long, device=self.device)
            noise_pred = self.model(xt, t_tensor)
            xt = self.schedule.denoise_step(noise_pred, t_tensor, xt)
        
        return xt
    
    @torch.no_grad()
    def _sample_ddim(self, xt: torch.Tensor, timesteps: torch.Tensor, eta: float) -> torch.Tensor:
        """DDIM sampling: jump across sparse timesteps"""
        from tqdm import tqdm
        
        for i in tqdm(range(len(timesteps) - 1), desc="DDIM sampling"):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            
            t_tensor = torch.full((xt.size(0),), t_curr.item(), dtype=torch.long, device=self.device)
            
            # Predict noise
            noise_pred = self.model(xt, t_tensor)
            
            # Get alpha coefficients
            alpha_curr = self.schedule.alphas_cumprod[t_curr]
            alpha_next = self.schedule.alphas_cumprod[t_next]
            
            # Estimate x_0
            sqrt_alphas_cumprod_t = self.schedule.sqrt_alphas_cumprod[t_curr]
            sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t_curr]
            
            # Reshape for broadcasting
            while len(sqrt_alphas_cumprod_t.shape) < len(xt.shape):
                sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
                sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
                alpha_curr = alpha_curr.unsqueeze(-1)
                alpha_next = alpha_next.unsqueeze(-1)
            
            x_0_pred = (xt - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
            
            # Calculate direction (variance)
            c1 = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_curr) * (1 - alpha_curr / alpha_next))
            c2 = torch.sqrt(1 - alpha_next - c1 ** 2)
            
            # Next step
            xt = torch.sqrt(alpha_next) * x_0_pred + c2 * noise_pred
            
            if eta > 0:
                xt = xt + c1 * torch.randn_like(xt)
        
        return xt
    
    @torch.no_grad()
    def sample_with_progress(
        self,
        batch_size: int = 1,
        img_size: int = 32,
        channels: int = 3,
        method: str = 'ddpm',
        num_steps: int = 50,
        eta: float = 0.0,
        progress_interval: int = 100
    ) -> Tuple[torch.Tensor, list]:
        """
        Sampling with progress tracking
        
        Args:
            batch_size: Batch size
            img_size: Image size
            channels: Number of channels
            method: Sampling method ('ddpm' or 'ddim')
            num_steps: Number of steps for DDIM
            eta: Stochasticity coefficient for DDIM
            progress_interval: Interval for saving trajectory
        
        Returns:
            Tuple of (generated images, trajectory)
        """
        self.model.eval()
        
        xt = torch.randn(batch_size, channels, img_size, img_size, device=self.device)
        timesteps = self._get_timesteps(method, num_steps)
        timesteps = timesteps.to(self.device)
        trajectory = [xt.cpu().clone()]
        
        if method.lower() == 'ddpm':
            for t in timesteps:
                t_tensor = torch.full((xt.size(0),), t.item(), dtype=torch.long, device=self.device)
                noise_pred = self.model(xt, t_tensor)
                xt = self.schedule.denoise_step(noise_pred, t_tensor, xt)
                
                if t.item() % progress_interval == 0:
                    trajectory.append(xt.cpu().clone())
        
        elif method.lower() == 'ddim':
            from tqdm import tqdm
            for i in tqdm(range(len(timesteps) - 1), desc="DDIM sampling with progress"):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                
                t_tensor = torch.full((xt.size(0),), t_curr.item(), dtype=torch.long, device=self.device)
                noise_pred = self.model(xt, t_tensor)
                
                alpha_curr = self.schedule.alphas_cumprod[t_curr]
                alpha_next = self.schedule.alphas_cumprod[t_next]
                
                sqrt_alphas_cumprod_t = self.schedule.sqrt_alphas_cumprod[t_curr]
                sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t_curr]
                
                while len(sqrt_alphas_cumprod_t.shape) < len(xt.shape):
                    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
                    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
                    alpha_curr = alpha_curr.unsqueeze(-1)
                    alpha_next = alpha_next.unsqueeze(-1)
                
                x_0_pred = (xt - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
                
                c1 = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_curr) * (1 - alpha_curr / alpha_next))
                c2 = torch.sqrt(1 - alpha_next - c1 ** 2)
                
                xt = torch.sqrt(alpha_next) * x_0_pred + c2 * noise_pred
                
                if eta > 0:
                    xt = xt + c1 * torch.randn_like(xt)
                
                trajectory.append(xt.cpu().clone())
        
        return xt, trajectory
