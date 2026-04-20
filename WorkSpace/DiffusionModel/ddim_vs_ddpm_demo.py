"""
DDIM vs DDPM 对比演示
展示两种采样方法的区别和性能差异
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm


class SimpleUNet(nn.Module):
    """
    简单的 U-Net 模型，用于预测噪声
    实际应用中应使用更复杂的架构
    """
    def __init__(self, in_channels=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 编码器
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        
        # 解码器
        self.dec1 = nn.Conv2d(128 + time_dim, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, in_channels, 3, padding=1)
        
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x, t):
        # 时间编码
        t_emb = self.time_mlp(t.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        
        # 编码
        e1 = self.act(self.enc1(x))
        e2 = self.act(self.enc2(self.pool(e1)))
        
        # 解码（注入时间信息）
        d1 = self.upsample(e2)
        d1 = torch.cat([d1, t_emb.expand_as(d1)], dim=1)
        d1 = self.act(self.dec1(d1))
        d2 = self.dec2(d1)
        
        return d2


class DiffusionSchedule:
    """
    扩散过程的时间表定义
    """
    def __init__(self, num_timesteps=1000, schedule_type='linear'):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        # 根据调度类型生成 beta 值
        if schedule_type == 'linear':
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif schedule_type == 'cosine':
            s = 0.008
            steps = torch.arange(num_timesteps + 1)
            alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        
        # 计算重要的系数
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 用于反向过程
        self.register_buffer(
            'sqrt_alphas_cumprod',
            torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod',
            torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_recip_alphas_cumprod',
            torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_recipm1_alphas_cumprod',
            torch.sqrt(1.0 / alphas_cumprod - 1)
        )
    
    def register_buffer(self, name, tensor):
        """为了兼容，手动管理张量"""
        setattr(self, name, tensor)


class DDPMSampler:
    """
    DDPM 采样器：标准的逐步采样方法
    需要完整的 T 步（通常是 1000 步）
    """
    def __init__(self, schedule: DiffusionSchedule, device='cpu'):
        self.schedule = schedule
        self.device = device
    
    def sample(self, model, batch_size=1, img_size=32, num_channels=3):
        """
        DDPM 采样：从纯噪声逐步生成图像
        
        Args:
            model: 噪声预测模型
            batch_size: 批次大小
            img_size: 图像尺寸
            num_channels: 通道数
        
        Returns:
            samples: 生成的图像
            trajectory: 采样轨迹（用于可视化）
        """
        model.eval()
        device = self.device
        
        # 初始化为纯噪声
        x = torch.randn(batch_size, num_channels, img_size, img_size, device=device)
        trajectory = [x.cpu().detach().clone()]
        
        # 反向过程：从 T 到 1
        for t in tqdm(range(self.schedule.num_timesteps - 1, 0, -1), desc="DDPM采样"):
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                # 预测噪声
                noise_pred = model(x, t_tensor / self.schedule.num_timesteps)
                
                # 提取所需的系数
                sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t]
                sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t]
                alpha = self.schedule.alphas[t]
                
                # 计算均值
                mean = (x - (1 - alpha) / sqrt_one_minus_alpha * noise_pred) / torch.sqrt(alpha)
                
                # 添加噪声（除了最后一步）
                if t > 1:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(self.schedule.betas[t])
                    x = mean + sigma * noise
                else:
                    x = mean
            
            # 保存采样轨迹（每 100 步保存一次）
            if t % 100 == 0:
                trajectory.append(x.cpu().detach().clone())
        
        return x, trajectory


class DDIMSampler:
    """
    DDIM 采样器：加速采样方法
    可以使用更少的步数（如 50、100 步）
    """
    def __init__(self, schedule: DiffusionSchedule, device='cpu'):
        self.schedule = schedule
        self.device = device
    
    def get_timesteps(self, num_steps: int, timesteps: int = None):
        """
        生成加速采样的时间步序列
        
        Args:
            num_steps: 采样步数
            timesteps: 总时间步数
        
        Returns:
            timesteps: 选定的时间步
        """
        if timesteps is None:
            timesteps = self.schedule.num_timesteps
        
        # 均匀采样：从总步数中均匀选择 num_steps 个
        step_ratio = timesteps / num_steps
        steps = np.round(np.arange(0, timesteps, step_ratio)).astype(np.int64)
        return steps
    
    def sample(self, model, batch_size=1, img_size=32, num_channels=3, num_steps=50, eta=0.0):
        """
        DDIM 采样：跳跃式采样方法
        
        Args:
            model: 噪声预测模型
            batch_size: 批次大小
            img_size: 图像尺寸
            num_channels: 通道数
            num_steps: 采样步数（关键：可以远小于总步数）
            eta: 随机性系数（0=完全确定性，1=回归DDPM）
        
        Returns:
            samples: 生成的图像
            trajectory: 采样轨迹
        """
        model.eval()
        device = self.device
        
        # 初始化为纯噪声
        x = torch.randn(batch_size, num_channels, img_size, img_size, device=device)
        trajectory = [x.cpu().detach().clone()]
        
        # 获取加速的时间步
        timesteps = self.get_timesteps(num_steps)
        timesteps = torch.from_numpy(timesteps).long().to(device)
        
        # 反向过程：跳跃式采样
        for i in tqdm(range(len(timesteps) - 1, 0, -1), desc=f"DDIM采样 ({num_steps}步)"):
            t_curr = timesteps[i]
            t_next = timesteps[i - 1]
            
            t_tensor = torch.tensor([float(t_curr)], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                # 预测噪声
                noise_pred = model(x, t_tensor / self.schedule.num_timesteps)
                
                # 提取系数
                alpha_curr = self.schedule.alphas_cumprod[t_curr]
                alpha_next = self.schedule.alphas_cumprod[t_next]
                
                # 计算 x_0 的估计
                x_0_est = (x - torch.sqrt(1 - alpha_curr) * noise_pred) / torch.sqrt(alpha_curr)
                
                # 计算方向（指向噪声）
                c1 = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_curr) * (1 - alpha_curr / alpha_next))
                c2 = torch.sqrt(1 - alpha_next - c1 ** 2)
                
                # 执行 DDIM 更新
                x = torch.sqrt(alpha_next) * x_0_est + c2 * noise_pred
                
                # 添加随机噪声
                if eta > 0:
                    x = x + c1 * torch.randn_like(x)
            
            trajectory.append(x.cpu().detach().clone())
        
        return x, trajectory


class DiffusionDemo:
    """
    演示类：对比 DDIM 和 DDPM
    """
    def __init__(self, device='cpu'):
        self.device = device
        
        # 创建调度
        self.schedule = DiffusionSchedule(num_timesteps=1000, schedule_type='linear')
        
        # 创建模型
        self.model = SimpleUNet(in_channels=3, time_dim=256).to(device)
        
        # 创建采样器
        self.ddpm_sampler = DDPMSampler(self.schedule, device)
        self.ddim_sampler = DDIMSampler(self.schedule, device)
    
    def compare_samplers(self, batch_size=1, img_size=32):
        """
        对比 DDPM 和 DDIM 的采样过程
        """
        print("=" * 60)
        print("DDIM vs DDPM 采样对比演示")
        print("=" * 60)
        
        # DDPM 采样（完整 1000 步）
        print("\n【DDPM 采样】")
        print(f"采样步数: 1000")
        print("采样方法: 完整马尔可夫链，逐步去噪")
        ddpm_samples, ddpm_traj = self.ddpm_sampler.sample(
            self.model, 
            batch_size=batch_size, 
            img_size=img_size
        )
        print(f"生成图像形状: {ddpm_samples.shape}")
        print(f"采样轨迹长度: {len(ddpm_traj)}")
        
        # DDIM 采样（50 步）
        print("\n【DDIM 采样 - 50 步】")
        print(f"采样步数: 50")
        print("采样方法: 跳跃式采样，非马尔可夫过程")
        print("采样速度: 约 20 倍快于 DDPM")
        ddim_samples_50, ddim_traj_50 = self.ddim_sampler.sample(
            self.model,
            batch_size=batch_size,
            img_size=img_size,
            num_steps=50,
            eta=0.0  # 完全确定性采样
        )
        print(f"生成图像形状: {ddim_samples_50.shape}")
        print(f"采样轨迹长度: {len(ddim_traj_50)}")
        
        # DDIM 采样（10 步）
        print("\n【DDIM 采样 - 10 步】")
        print(f"采样步数: 10")
        print("采样方法: 极致加速，跳跃更大")
        print("采样速度: 约 100 倍快于 DDPM")
        ddim_samples_10, ddim_traj_10 = self.ddim_sampler.sample(
            self.model,
            batch_size=batch_size,
            img_size=img_size,
            num_steps=10,
            eta=0.0
        )
        print(f"生成图像形状: {ddim_samples_10.shape}")
        print(f"采样轨迹长度: {len(ddim_traj_10)}")
        
        return {
            'ddpm': {'samples': ddpm_samples, 'trajectory': ddpm_traj},
            'ddim_50': {'samples': ddim_samples_50, 'trajectory': ddim_traj_50},
            'ddim_10': {'samples': ddim_samples_10, 'trajectory': ddim_traj_10},
        }
    
    def analyze_differences(self):
        """
        分析 DDIM 和 DDPM 的关键区别
        """
        print("\n" + "=" * 60)
        print("DDIM vs DDPM 关键区别分析")
        print("=" * 60)
        
        differences = {
            "特性": {
                "DDPM": "标准扩散模型",
                "DDIM": "改进的扩散模型"
            },
            "采样方式": {
                "DDPM": "逐步采样，必须遵循顺序",
                "DDIM": "跳跃式采样，可跳过中间步骤"
            },
            "采样步数": {
                "DDPM": "通常 1000 步",
                "DDIM": "50-100 步（甚至 10 步）"
            },
            "数学基础": {
                "DDPM": "马尔可夫链（每步依赖前一步）",
                "DDIM": "非马尔可夫过程（允许任意跳跃）"
            },
            "推理速度": {
                "DDPM": "基准速度",
                "DDIM": "10-100 倍加速"
            },
            "生成质量": {
                "DDPM": "最优质量",
                "DDIM": "略有下降（50 步时几乎无损）"
            },
            "确定性": {
                "DDPM": "本质随机（总有随机噪声）",
                "DDIM": "可调节（η=0 时完全确定性）"
            },
            "重新训练": {
                "DDPM": "需要完整训练",
                "DDIM": "无需重新训练，即插即用"
            },
            "时间步依赖": {
                "DDPM": "必须依赖 α_t 和 β_t",
                "DDIM": "只依赖 ᾱ_t（累积方差）"
            },
            "应用场景": {
                "DDPM": "理论研究、质量优先",
                "DDIM": "实际应用、速度优先"
            }
        }
        
        for key, values in differences.items():
            print(f"\n【{key}】")
            for method, desc in values.items():
                print(f"  {method:8s}: {desc}")
        
        print("\n" + "=" * 60)
        print("性能指标对比")
        print("=" * 60)
        
        metrics = {
            "方法": ["DDPM (1000步)", "DDIM (100步)", "DDIM (50步)", "DDIM (10步)"],
            "相对推理时间": ["100%", "~10%", "~5%", "~1%"],
            "预期质量": ["100%", "~98%", "~95%", "~70%"],
            "确定性": ["否", "可选（η）", "可选（η）", "可选（η）"],
            "实际应用": ["研究", "高质量生成", "平衡方案", "极速预览"]
        }
        
        # 打印表格
        headers = ["方法", "相对推理时间", "预期质量", "确定性", "实际应用"]
        print(f"\n{headers[0]:20s} {headers[1]:15s} {headers[2]:15s} {headers[3]:10s} {headers[4]:15s}")
        print("-" * 75)
        for i, method in enumerate(metrics["方法"]):
            print(f"{method:20s} {metrics['相对推理时间'][i]:15s} {metrics['预期质量'][i]:15s} {metrics['确定性'][i]:10s} {metrics['实际应用'][i]:15s}")


def main():
    """
    主函数：运行演示
    """
    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")
    
    # 创建演示
    demo = DiffusionDemo(device=device)
    
    # 对比采样
    results = demo.compare_samplers(batch_size=1, img_size=32)
    
    # 分析区别
    demo.analyze_differences()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n【关键要点总结】")
    print("1. DDPM 使用完整的 1000 步推理过程")
    print("2. DDIM 通过重参数化允许跳跃式采样")
    print("3. DDIM 50 步的质量接近 DDPM 1000 步")
    print("4. DDIM 10 步可以实现极速预览（但质量下降）")
    print("5. DDIM 无需重新训练，可直接应用到已训练的 DDPM 模型")
    print("6. 参数 η 控制随机性：η=0（确定性）→ η=1（回归DDPM）")
    print("=" * 60)


if __name__ == "__main__":
    main()
