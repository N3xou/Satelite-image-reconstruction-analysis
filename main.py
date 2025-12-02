"""
Satellite Image Cloud Removal - ML/DL Comparison Project
========================================================
Compares multiple approaches for reconstructing cloud-covered satellite imagery
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import rasterio (optional, needed only for real Sentinel-2 data)
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Note: rasterio not installed. Install for real Sentinel-2 data: pip install rasterio")

# ==================== 1. DATA ACQUISITION & PREPARATION ====================

class SatelliteDatasetPreparer:
    """Handles dataset acquisition and cloud simulation"""

    def __init__(self, data_dir='./satellite_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def download_sentinel2_sample(self):
        """
        Download sample Sentinel-2 data
        For real implementation, use: sentinelsat, Google Earth Engine, or AWS S3
        """
        print("Note: For real data, integrate with:")
        print("- Sentinelsat API for Sentinel-2")
        print("- Google Earth Engine Python API")
        print("- AWS S3 for Landsat/Sentinel data")
        print("\nCreating synthetic dataset for demonstration...")

    def create_synthetic_dataset(self, n_samples=100, img_size=(256, 256), n_bands=4):
        """Create synthetic multispectral images with simulated clouds"""
        clean_dir = self.data_dir / 'clean'
        cloudy_dir = self.data_dir / 'cloudy'
        clean_dir.mkdir(exist_ok=True)
        cloudy_dir.mkdir(exist_ok=True)

        for i in range(n_samples):
            # Generate synthetic multispectral image (R, G, B, NIR)
            clean_img = self._generate_landscape(img_size, n_bands)

            # Add clouds
            cloudy_img = self._add_clouds(clean_img)

            # Save
            np.save(clean_dir / f'img_{i:04d}.npy', clean_img)
            np.save(cloudy_dir / f'img_{i:04d}.npy', cloudy_img)

        print(f"Created {n_samples} synthetic image pairs")
        return clean_dir, cloudy_dir

    def _generate_landscape(self, size, n_bands):
        """Generate realistic-looking landscape with multiple spectral bands"""
        h, w = size
        img = np.zeros((n_bands, h, w))

        for band in range(n_bands):
            # Create terrain with fractal noise
            x = np.linspace(0, 4, w)
            y = np.linspace(0, 4, h)
            X, Y = np.meshgrid(x, y)

            terrain = (np.sin(X * 2) + np.cos(Y * 2) +
                      0.5 * np.sin(X * 4 + Y * 4) +
                      0.25 * np.random.randn(h, w))

            # Different spectral responses per band
            img[band] = (terrain + 3) / 6 * (0.8 + band * 0.1)

        return np.clip(img, 0, 1).astype(np.float32)

    def _add_clouds(self, img, cloud_density=0.3):
        """Add realistic cloud cover to image"""
        n_bands, h, w = img.shape
        cloudy = img.copy()

        # Generate cloud mask
        cloud_mask = np.random.rand(h, w) < cloud_density

        # Smooth cloud edges
        from scipy.ndimage import gaussian_filter
        cloud_mask = gaussian_filter(cloud_mask.astype(float), sigma=10) > 0.3

        # Apply clouds (bright in all bands)
        for band in range(n_bands):
            cloud_values = np.random.uniform(0.7, 1.0, (h, w))
            cloudy[band] = np.where(cloud_mask, cloud_values, cloudy[band])

        return cloudy.astype(np.float32)


class SatelliteDataset(Dataset):
    """PyTorch Dataset for satellite image pairs"""

    def __init__(self, clean_dir, cloudy_dir, transform=None, preload=True, cache_size=50):
        self.clean_dir = Path(clean_dir)
        self.cloudy_dir = Path(cloudy_dir)
        self.files = sorted(list(self.clean_dir.glob('*.npy')))
        self.transform = transform
        self.preload = preload
        self.cache = {}
        self.cache_size = cache_size

        # Preload all data to RAM if dataset is small (CPU bottleneck fix #1)
        if preload and len(self.files) <= cache_size:
            print(f"Preloading {len(self.files)} images to RAM...")
            for idx in range(len(self.files)):
                self.cache[idx] = self._load_from_disk(idx)
            print("Preloading complete!")

    def _load_from_disk(self, idx):
        """Load image pair from disk"""
        clean_path = self.files[idx]
        cloudy_path = self.cloudy_dir / clean_path.name

        clean = torch.from_numpy(np.load(clean_path))
        cloudy = torch.from_numpy(np.load(cloudy_path))

        if self.transform:
            clean = self.transform(clean)
            cloudy = self.transform(cloudy)

        return cloudy, clean

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Use cached data if available (CPU bottleneck fix #2)
        if idx in self.cache:
            return self.cache[idx]

        # Load from disk
        data = self._load_from_disk(idx)

        # Cache if within cache size
        if len(self.cache) < self.cache_size:
            self.cache[idx] = data

        return data


# ==================== 2. DATA EXPLORATION & INSIGHTS ====================

class DataExplorer:
    """Generate insights about the satellite dataset"""

    def __init__(self, dataset):
        self.dataset = dataset

    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)

        # Basic statistics
        print(f"\nDataset size: {len(self.dataset)} image pairs")

        # Sample images
        sample_cloudy, sample_clean = self.dataset[0]
        print(f"Image shape: {sample_clean.shape}")
        print(f"Number of spectral bands: {sample_clean.shape[0]}")
        print(f"Spatial resolution: {sample_clean.shape[1]}x{sample_clean.shape[2]}")

        # Statistical analysis
        self._compute_statistics()
        self._plot_spectral_profiles()
        self._visualize_samples()

    def _compute_statistics(self):
        """Compute dataset statistics"""
        all_clean = []
        all_cloudy = []

        for i in range(min(50, len(self.dataset))):
            cloudy, clean = self.dataset[i]
            all_clean.append(clean.numpy())
            all_cloudy.append(cloudy.numpy())

        all_clean = np.array(all_clean)
        all_cloudy = np.array(all_cloudy)

        print("\n--- Statistical Summary ---")
        print(f"Clean images - Mean: {all_clean.mean():.4f}, Std: {all_clean.std():.4f}")
        print(f"Cloudy images - Mean: {all_cloudy.mean():.4f}, Std: {all_cloudy.std():.4f}")

        # Cloud coverage estimation
        cloud_coverage = (all_cloudy > 0.7).mean(axis=(1,2,3)) * 100
        print(f"Average cloud coverage: {cloud_coverage.mean():.2f}%")

    def _plot_spectral_profiles(self):
        """Plot spectral profiles across bands"""
        cloudy, clean = self.dataset[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for band in range(clean.shape[0]):
            axes[0].plot(clean[band, 128, :].numpy(), label=f'Band {band+1}')
            axes[1].plot(cloudy[band, 128, :].numpy(), label=f'Band {band+1}')

        axes[0].set_title('Clean Image - Spectral Profile')
        axes[1].set_title('Cloudy Image - Spectral Profile')
        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        plt.savefig('spectral_profiles.png')
        plt.close()

    def _visualize_samples(self, n_samples=3):
        """Visualize sample image pairs"""
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))

        for i in range(n_samples):
            cloudy, clean = self.dataset[i]

            # Use first 3 bands as RGB
            cloudy_rgb = cloudy[:3].permute(1, 2, 0).numpy()
            clean_rgb = clean[:3].permute(1, 2, 0).numpy()
            diff = np.abs(cloudy_rgb - clean_rgb)

            axes[i, 0].imshow(cloudy_rgb)
            axes[i, 0].set_title('Cloudy Input')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(clean_rgb)
            axes[i, 1].set_title('Clean Target')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(diff)
            axes[i, 2].set_title('Difference')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig('sample_pairs.png')
        plt.close()
        print("\nVisualizations saved: spectral_profiles.png, sample_pairs.png")


# ==================== 3. MODEL IMPLEMENTATIONS ====================

# Model 1: Deep CNN (U-Net architecture)
class UNet(nn.Module):
    """U-Net architecture for image-to-image translation"""

    def __init__(self, in_channels=4, out_channels=4):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, 1)

        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))


# Model 2: Simple CNN
class SimpleCNN(nn.Module):
    """Simple CNN baseline model"""

    def __init__(self, in_channels=4, out_channels=4):
        super(SimpleCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Model 3: GAN (Generator + Discriminator)
class Generator(nn.Module):
    """GAN Generator - ResNet-based architecture"""

    def __init__(self, in_channels=4, out_channels=4):
        super(Generator, self).__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.down1 = self._downsample_block(64, 128)
        self.down2 = self._downsample_block(128, 256)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[self._residual_block(256) for _ in range(6)]
        )

        # Upsampling
        self.up1 = self._upsample_block(256, 128)
        self.up2 = self._upsample_block(128, 64)

        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Sigmoid()
        )

    def _downsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def _upsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def _residual_block(self, channels):
        return ResidualBlock(channels)

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.output(x)


class ResidualBlock(nn.Module):
    """Residual block for Generator"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Discriminator(nn.Module):
    """GAN Discriminator (PatchGAN)"""

    def __init__(self, in_channels=8):  # Input + output concatenated
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


# Model 4: Time Series LSTM (using temporal context)
class LSTMCloudRemover(nn.Module):
    """LSTM-based model for temporal cloud removal"""

    def __init__(self, in_channels=4, hidden_size=128):
        super(LSTMCloudRemover, self).__init__()

        self.hidden_size = hidden_size
        self.in_channels = in_channels

        # Spatial feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_size, 3, padding=1),
            nn.ReLU()
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, channels, H, W)
        batch_size, seq_len, c, h, w = x.shape

        # Extract features for each timestep
        features = []
        for t in range(seq_len):
            feat = self.feature_extractor(x[:, t])
            features.append(feat.mean(dim=[2, 3]))  # Spatial pooling

        features = torch.stack(features, dim=1)  # (batch, seq, hidden)

        # LSTM processing
        lstm_out, _ = self.lstm(features)

        # Use last timestep output
        last_feat = lstm_out[:, -1]  # (batch, hidden)

        # Reshape and decode
        last_feat = last_feat.unsqueeze(-1).unsqueeze(-1)
        last_feat = last_feat.expand(-1, -1, h, w)

        output = self.decoder(last_feat)
        return output


# Model 5: Diffusion Model
class DiffusionModel(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) for cloud removal
    Iteratively removes noise/clouds through multiple denoising steps
    """

    def __init__(self, in_channels=4, model_channels=64, num_res_blocks=2):
        super(DiffusionModel, self).__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])

        # Downsampling path
        ch = model_channels
        for level in range(3):
            for _ in range(num_res_blocks):
                self.input_blocks.append(
                    ResBlock(ch, time_embed_dim, ch)
                )
            if level < 2:
                self.input_blocks.append(
                    nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)
                )
                ch *= 2

        # Middle
        self.middle_block = nn.Sequential(
            ResBlock(ch, time_embed_dim, ch),
            ResBlock(ch, time_embed_dim, ch),
        )

        # Upsampling path
        self.output_blocks = nn.ModuleList([])
        for level in range(3):
            for _ in range(num_res_blocks):
                self.output_blocks.append(
                    ResBlock(ch, time_embed_dim, ch)
                )
            if level < 2:
                self.output_blocks.append(
                    nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1)
                )
                ch //= 2

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        """
        Args:
            x: Noisy input [batch, channels, H, W]
            timesteps: Current timestep [batch]
        """
        # Time embedding
        t_emb = self._get_timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        # Process
        h = x
        hs = []
        for module in self.input_blocks:
            if isinstance(module, ResBlock):
                h = module(h, emb)
            else:
                h = module(h)
            hs.append(h)

        h = self.middle_block[0](h, emb)
        h = self.middle_block[1](h, emb)

        for module in self.output_blocks:
            if isinstance(module, ResBlock):
                h = module(h, emb)
            else:
                h = module(h)

        return self.out(h)

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """Sinusoidal timestep embeddings"""
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time embedding for Diffusion Model"""

    def __init__(self, channels, time_embed_dim, out_channels):
        super(ResBlock, self).__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        if channels != out_channels:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)[:, :, None, None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class DiffusionTrainer:
    """Trainer for Diffusion Model"""

    def __init__(self, model, device, num_timesteps=1000):
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps

        # Define noise schedule (linear)
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to clean image"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]) ** 0.5

        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def train_step(self, x_clean, optimizer):
        """Single training step"""
        batch_size = x_clean.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

        # Sample noise
        noise = torch.randn_like(x_clean)

        # Add noise to clean images
        x_noisy = self.q_sample(x_clean, t, noise)

        # Predict noise
        predicted_noise = self.model(x_noisy, t)

        # Loss: difference between predicted and actual noise
        loss = nn.functional.mse_loss(predicted_noise, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def p_sample(self, x, t):
        """Single denoising step"""
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]) ** 0.5
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]
        sqrt_recip_alphas_t = (1.0 / self.alphas[t]) ** 0.5
        sqrt_recip_alphas_t = sqrt_recip_alphas_t[:, None, None, None]

        # Predict noise
        predicted_noise = self.model(x, t)

        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + (betas_t ** 0.5) * noise

    @torch.no_grad()
    def denoise(self, x_noisy, num_inference_steps=50):
        """
        Full denoising process (cloud removal)

        Args:
            x_noisy: Cloudy input image
            num_inference_steps: Number of denoising steps (fewer = faster)
        """
        self.model.eval()

        # Start from noisy input
        x = x_noisy

        # Reverse process
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long
        ).to(self.device)

        for t in timesteps:
            t_batch = t.repeat(x.shape[0])
            x = self.p_sample(x, t_batch)

        return torch.clamp(x, 0, 1)


# ==================== 4. TRAINING WITH K-FOLD ====================

class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=3, min_delta=0.0, verbose=True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """
        Returns:
            True if should stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Validation loss improved: {self.best_loss:.6f} → {val_loss:.6f}')
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

        return self.early_stop

    def load_best_model(self, model):
        """Load the best model weights"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class ModelTrainer:
    """Handles training with K-Fold cross-validation"""

    def __init__(self, model_type, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_type = model_type
        self.device = device
        self.models = []
        self.histories = []
        self.training_time = 0  # Track total training time

        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

            # Enable optimizations (CPU bottleneck fix #3)
            torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
            torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul
            torch.backends.cudnn.allow_tf32 = True

    def train_kfold(self, dataset, k=5, epochs=20, batch_size=8, lr=0.001,
                    patience=7, use_amp=False, num_workers=4, persistent_workers=True,
                    prefetch_factor=3):
        """
        Train model with K-fold cross-validation

        DIFFERENCE: train_kfold() splits data into K folds and trains K separate models.
                   Each fold uses different train/val split for robust evaluation.
                   _train_fold() is the internal method that trains a single model.

        Args:
            dataset: PyTorch Dataset
            k: Number of folds (K models will be trained)
            epochs: Training epochs per fold
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
            use_amp: Use Automatic Mixed Precision
            num_workers: Data loading workers (CPU bottleneck fix #4)
            persistent_workers: Keep workers alive between epochs (CPU bottleneck fix #5)
            prefetch_factor: Batches to prefetch per worker (CPU bottleneck fix #6)
        """
        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Training {self.model_type} with {k}-Fold CV")
        print(f"{'='*60}")

        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        indices = list(range(len(dataset)))

        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            print(f"\nFold {fold + 1}/{k}")
            print("-" * 40)

            # Create data loaders with optimizations (CPU bottleneck fix #7)
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                drop_last=True  # Avoid small last batch
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                num_workers=num_workers // 2,  # Fewer workers for validation
                pin_memory=True,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=prefetch_factor if num_workers > 0 else None
            )

            # Initialize model
            if self.model_type == 'UNet':
                model = UNet().to(self.device)
            elif self.model_type == 'SimpleCNN':
                model = SimpleCNN().to(self.device)
            elif self.model_type == 'GAN':
                model = self._train_gan_fold(train_loader, val_loader, epochs, lr,
                                            patience, use_amp)
                self.models.append(model)
                continue
            elif self.model_type == 'LSTM':
                model = LSTMCloudRemover().to(self.device)
            elif self.model_type == 'Diffusion':
                model = self._train_diffusion_fold(train_loader, val_loader, epochs,
                                                   lr, patience, use_amp)
                self.models.append(model)
                continue
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Train single fold
            history = self._train_fold(model, train_loader, val_loader, epochs, lr,
                                      patience, use_amp)

            self.models.append(model)
            self.histories.append(history)

        # Record total training time
        self.training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Total training time: {self.training_time/60:.2f} minutes")
        print(f"Average time per fold: {self.training_time/k/60:.2f} minutes")
        print(f"{'='*60}")

        return self.models, self.histories

    def _train_fold(self, model, train_loader, val_loader, epochs, lr,
                   patience, use_amp):
        """
        Train single fold with early stopping

        DIFFERENCE: This is an internal method called by train_kfold().
                   It trains ONE model on ONE specific train/val split.
                   train_kfold() calls this K times with different splits.
        """
        import time
        fold_start = time.time()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        # Mixed precision training (saves memory for GTX 1050)
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None

        history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}

        for epoch in range(epochs):
            epoch_start = time.time()

            # Training
            model.train()
            train_loss = 0
            for cloudy, clean in train_loader:
                cloudy, clean = cloudy.to(self.device, non_blocking=True), clean.to(self.device, non_blocking=True)

                # Handle LSTM input
                if self.model_type == 'LSTM':
                    cloudy = cloudy.unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)  # CPU bottleneck fix #8

                # Mixed precision forward pass
                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(cloudy)
                        loss = criterion(output, clean)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(cloudy)
                    loss = criterion(output, clean)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for cloudy, clean in val_loader:
                    cloudy, clean = cloudy.to(self.device, non_blocking=True), clean.to(self.device, non_blocking=True)

                    if self.model_type == 'LSTM':
                        cloudy = cloudy.unsqueeze(1)

                    if use_amp and self.device == 'cuda':
                        with torch.cuda.amp.autocast():
                            output = model(cloudy)
                            loss = criterion(output, clean)
                    else:
                        output = model(cloudy)
                        loss = criterion(output, clean)

                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            epoch_time = time.time() - epoch_start

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['epoch_times'].append(epoch_time)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s")

            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        early_stopping.load_best_model(model)
        fold_time = time.time() - fold_start
        print(f"Fold training time: {fold_time/60:.2f} minutes")
        print(f"Loaded best model with val_loss: {early_stopping.best_loss:.6f}")

        return history

    def _train_gan_fold(self, train_loader, val_loader, epochs, lr, patience, use_amp):
        """Train GAN for single fold with early stopping"""
        generator = Generator().to(self.device)
        discriminator = Discriminator().to(self.device)

        criterion_gan = nn.BCEWithLogitsLoss()
        criterion_l1 = nn.L1Loss()

        optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        early_stopping = EarlyStopping(patience=patience, verbose=True)
        scaler_g = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None
        scaler_d = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None

        for epoch in range(epochs):
            generator.train()
            discriminator.train()

            for cloudy, clean in train_loader:
                cloudy, clean = cloudy.to(self.device), clean.to(self.device)
                batch_size = cloudy.size(0)

                # Labels for discriminator
                real_label = torch.ones(batch_size, 1, 30, 30).to(self.device)
                fake_label = torch.zeros(batch_size, 1, 30, 30).to(self.device)

                # Train Discriminator
                optimizer_d.zero_grad()

                if use_amp and scaler_d is not None:
                    with torch.cuda.amp.autocast():
                        fake_clean = generator(cloudy)
                        real_loss = criterion_gan(discriminator(cloudy, clean), real_label)
                        fake_loss = criterion_gan(discriminator(cloudy, fake_clean.detach()), fake_label)
                        d_loss = (real_loss + fake_loss) / 2
                    scaler_d.scale(d_loss).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    fake_clean = generator(cloudy)
                    real_loss = criterion_gan(discriminator(cloudy, clean), real_label)
                    fake_loss = criterion_gan(discriminator(cloudy, fake_clean.detach()), fake_label)
                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()

                if use_amp and scaler_g is not None:
                    with torch.cuda.amp.autocast():
                        fake_clean = generator(cloudy)
                        gan_loss = criterion_gan(discriminator(cloudy, fake_clean), real_label)
                        l1_loss = criterion_l1(fake_clean, clean)
                        g_loss = gan_loss + 100 * l1_loss
                    scaler_g.scale(g_loss).backward()
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    fake_clean = generator(cloudy)
                    gan_loss = criterion_gan(discriminator(cloudy, fake_clean), real_label)
                    l1_loss = criterion_l1(fake_clean, clean)
                    g_loss = gan_loss + 100 * l1_loss
                    g_loss.backward()
                    optimizer_g.step()

            # Validation
            generator.eval()
            val_loss = 0
            with torch.no_grad():
                for cloudy, clean in val_loader:
                    cloudy, clean = cloudy.to(self.device), clean.to(self.device)
                    fake_clean = generator(cloudy)
                    val_loss += criterion_l1(fake_clean, clean).item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if early_stopping(val_loss, generator):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        early_stopping.load_best_model(generator)
        return generator

    def _train_diffusion_fold(self, train_loader, val_loader, epochs, lr, patience, use_amp):
        """Train Diffusion model for single fold"""
        model = DiffusionModel().to(self.device)
        diffusion_trainer = DiffusionTrainer(model, self.device, num_timesteps=1000)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for cloudy, clean in train_loader:
                clean = clean.to(self.device)

                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = diffusion_trainer.train_step(clean, optimizer)
                else:
                    loss = diffusion_trainer.train_step(clean, optimizer)

                train_loss += loss

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for cloudy, clean in val_loader:
                    cloudy, clean = cloudy.to(self.device), clean.to(self.device)
                    # Denoise cloudy image
                    denoised = diffusion_trainer.denoise(cloudy, num_inference_steps=20)
                    val_loss += nn.functional.mse_loss(denoised, clean).item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if early_stopping(val_loss, model):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        early_stopping.load_best_model(model)
        return model


# ==================== 5. EVALUATION & COMPARISON ====================

class ModelEvaluator:
    """Evaluate and compare model performance"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        self.timing_results = {}  # Store timing information

    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate single model"""
        import time

        model.eval()

        all_preds = []
        all_targets = []
        inference_times = []

        with torch.no_grad():
            for cloudy, clean in test_loader:
                cloudy = cloudy.to(self.device, non_blocking=True)

                if 'LSTM' in model_name:
                    cloudy = cloudy.unsqueeze(1)

                # Measure inference time
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()

                output = model(cloudy)

                if self.device == 'cuda':
                    torch.cuda.synchronize()
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                all_preds.append(output.cpu().numpy())
                all_targets.append(clean.numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Calculate metrics
        mse = mean_squared_error(targets.flatten(), preds.flatten())
        mae = mean_absolute_error(targets.flatten(), preds.flatten())
        rmse = np.sqrt(mse)

        # PSNR
        psnr_scores = []
        for i in range(len(preds)):
            psnr = self._calculate_psnr(targets[i], preds[i])
            psnr_scores.append(psnr)

        psnr = np.mean(psnr_scores)

        # SSIM (simplified version)
        ssim = self._calculate_ssim(targets, preds)

        # Timing statistics
        avg_inference_time = np.mean(inference_times)
        total_inference_time = np.sum(inference_times)
        images_per_second = len(preds) / total_inference_time

        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'PSNR': psnr,
            'SSIM': ssim,
            'Avg_Inference_Time': avg_inference_time,
            'Images_Per_Second': images_per_second
        }

        self.results[model_name] = metrics

        print(f"\n{model_name} Results:")
        print("-" * 40)
        for metric, value in metrics.items():
            if 'Time' in metric or 'Second' in metric:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value:.6f}")

        return metrics

    def _calculate_psnr(self, target, pred, max_val=1.0):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((target - pred) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))

    def _calculate_ssim(self, targets, preds):
        """Simplified SSIM calculation"""
        # Simplified version - full SSIM requires more complex windowing
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = np.mean(targets)
        mu_y = np.mean(preds)
        sigma_x = np.std(targets)
        sigma_y = np.std(preds)
        sigma_xy = np.mean((targets - mu_x) * (preds - mu_y))

        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))

        return ssim

    def compare_models(self, training_times=None):
        """Generate comparison visualizations including training time"""
        if not self.results:
            print("No results to compare")
            return

        df = pd.DataFrame(self.results).T

        # Add training times if provided
        if training_times:
            df['Training_Time_Minutes'] = [training_times.get(model, 0) / 60
                                           for model in df.index]

        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(df.to_string())

        # Plot comparison with training time
        n_metrics = len(df.columns)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        metrics = list(df.columns)
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']

        for i, metric in enumerate(metrics):
            if i < len(axes):
                df[metric].plot(kind='bar', ax=axes[i], color=colors[:len(df)])
                axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create training time vs quality plot
        if training_times:
            fig, ax = plt.subplots(figsize=(10, 6))

            train_times = [training_times.get(model, 0) / 60 for model in df.index]
            psnr_values = df['PSNR'].values

            ax.scatter(train_times, psnr_values, s=200, alpha=0.6, c=range(len(df)), cmap='viridis')

            for i, model in enumerate(df.index):
                ax.annotate(model, (train_times[i], psnr_values[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold')

            ax.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
            ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
            ax.set_title('Training Time vs Quality', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('time_vs_quality.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("\nTime vs Quality plot saved to: time_vs_quality.png")

        print("\nComparison saved to: model_comparison.png")

        # Best model analysis
        best_psnr_model = df['PSNR'].idxmax()
        best_speed_model = df['Images_Per_Second'].idxmax()

        print(f"\n{'='*60}")
        print("BEST MODEL ANALYSIS")
        print(f"{'='*60}")
        print(f"Best Quality (PSNR): {best_psnr_model} ({df.loc[best_psnr_model, 'PSNR']:.2f} dB)")
        print(f"Fastest Inference: {best_speed_model} ({df.loc[best_speed_model, 'Images_Per_Second']:.2f} img/s)")

        if training_times:
            fastest_training = df['Training_Time_Minutes'].idxmin()
            print(f"Fastest Training: {fastest_training} ({df.loc[fastest_training, 'Training_Time_Minutes']:.2f} min)")

        print(f"{'='*60}")

        return df


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution pipeline"""

    print("="*60)
    print("SATELLITE CLOUD REMOVAL - ML/DL COMPARISON PROJECT")
    print("="*60)

    # 1. Data Preparation
    print("\n### STEP 1: Data Preparation ###")
    preparer = SatelliteDatasetPreparer()
    clean_dir, cloudy_dir = preparer.create_synthetic_dataset(n_samples=100)

    dataset = SatelliteDataset(clean_dir, cloudy_dir)

    # 2. Data Exploration
    print("\n### STEP 2: Data Exploration ###")
    explorer = DataExplorer(dataset)
    explorer.analyze_dataset()

    # 3. Split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 4. Train Models with K-Fold
    print("\n### STEP 3: Model Training with K-Fold CV ###")

    #model_types = ['SimpleCNN', 'UNet', 'GAN', 'LSTM', 'Diffusion']
    model_types = ['GAN']
    trained_models = {}
    training_times = {}  # Track training time for each model

    for model_type in model_types:
        trainer = ModelTrainer(model_type)
        models, histories = trainer.train_kfold(
            train_dataset,
            k=3,  # 3-fold for faster execution
            epochs=30,
            batch_size=4,  # Optimized for GTX 1050
            lr=0.001,
            patience=3,  # Early stopping patience
            use_amp=True,  # Mixed precision for memory efficiency
            num_workers=6,  # CPU bottleneck fix: More workers
            persistent_workers=True,  # CPU bottleneck fix: Keep workers alive
            prefetch_factor=4  # CPU bottleneck fix: Prefetch more batches
        )
        trained_models[model_type] = models[0]  # Use first fold model
        training_times[model_type] = trainer.training_time  # Store training time

    # 5. Evaluate and Compare
    print("\n### STEP 4: Model Evaluation and Comparison ###")

    evaluator = ModelEvaluator()

    for model_name, model in trained_models.items():
        evaluator.evaluate_model(model, test_loader, model_name)

    # Generate comparison with training times
    comparison_df = evaluator.compare_models(training_times=training_times)

    # 6. Visualize Predictions
    print("\n### STEP 5: Visualizing Predictions ###")
    visualize_predictions(trained_models, test_dataset)

    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("- spectral_profiles.png")
    print("- sample_pairs.png")
    print("- model_comparison.png")
    print("- predictions_comparison.png")


def visualize_predictions(models, test_dataset, n_samples=3):
    """Visualize predictions from all models"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fig, axes = plt.subplots(n_samples, len(models) + 2, figsize=(20, 4*n_samples))

    for i in range(n_samples):
        cloudy, clean = test_dataset[i]
        cloudy_input = cloudy.unsqueeze(0).to(device)

        # Display cloudy input
        cloudy_rgb = cloudy[:3].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(np.clip(cloudy_rgb, 0, 1))
        axes[i, 0].set_title('Cloudy Input')
        axes[i, 0].axis('off')

        # Display clean target
        clean_rgb = clean[:3].permute(1, 2, 0).numpy()
        axes[i, 1].imshow(np.clip(clean_rgb, 0, 1))
        axes[i, 1].set_title('Clean Target')
        axes[i, 1].axis('off')

        # Display predictions from each model
        for j, (model_name, model) in enumerate(models.items()):
            model.eval()
            with torch.no_grad():
                if 'LSTM' in model_name:
                    cloudy_input_lstm = cloudy_input.unsqueeze(1)
                    pred = model(cloudy_input_lstm)
                else:
                    pred = model(cloudy_input)

                pred_rgb = pred[0, :3].cpu().permute(1, 2, 0).numpy()
                axes[i, j + 2].imshow(np.clip(pred_rgb, 0, 1))
                axes[i, j + 2].set_title(f'{model_name} Output')
                axes[i, j + 2].axis('off')

    plt.tight_layout()
    plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Predictions visualization saved to: predictions_comparison.png")


# ==================== ADDITIONAL UTILITIES ====================

class ExperimentLogger:
    """Log experiment results for reproducibility"""

    def __init__(self, log_file='experiment_log.txt'):
        self.log_file = log_file

    def log_experiment(self, config, results):
        """Log experiment configuration and results"""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"Experiment Date: {pd.Timestamp.now()}\n")
            f.write("\nConfiguration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nResults:\n")
            for model, metrics in results.items():
                f.write(f"\n{model}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.6f}\n")

        print(f"Experiment logged to: {self.log_file}")


def save_models(models, save_dir='./saved_models'):
    """Save trained models"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for model_name, model in models.items():
        path = save_dir / f'{model_name}.pth'
        torch.save(model.state_dict(), path)
        print(f"Saved {model_name} to {path}")


def load_real_sentinel2_data(data_path):
    """
    Load real Sentinel-2 data using rasterio

    Example usage with real data:
    ```python
    import rasterio

    with rasterio.open('sentinel2_image.tif') as src:
        bands = src.read()  # Read all bands
        metadata = src.meta
    ```
    """
    print("For loading real Sentinel-2 data:")
    print("1. Install: pip install sentinelsat rasterio")
    print("2. Download data from Copernicus Open Access Hub")
    print("3. Use rasterio to read multispectral bands")
    print("4. Preprocess: normalize, align, create cloud masks")


# ==================== ADVANCED FEATURES ====================

class CloudDetector:
    """Detect and mask clouds in images"""

    @staticmethod
    def detect_clouds(image, threshold=0.7):
        """Simple cloud detection based on brightness"""
        # Average across spectral bands
        brightness = image.mean(axis=0)
        cloud_mask = brightness > threshold
        return cloud_mask

    @staticmethod
    def create_cloud_free_composite(image_stack):
        """Create cloud-free composite from time series"""
        # Use median of non-cloudy pixels across time
        masks = [CloudDetector.detect_clouds(img) for img in image_stack]

        composite = np.zeros_like(image_stack[0])
        for band in range(image_stack[0].shape[0]):
            band_stack = np.array([img[band] for img in image_stack])
            # Mask cloudy pixels
            for i, mask in enumerate(masks):
                band_stack[i][mask] = np.nan
            # Take median
            composite[band] = np.nanmedian(band_stack, axis=0)

        return composite


class TemporalDataGenerator:
    """Generate temporal sequences for LSTM training"""

    def __init__(self, base_dataset, sequence_length=3):
        self.base_dataset = base_dataset
        self.sequence_length = sequence_length

    def create_sequences(self):
        """Create temporal sequences from dataset"""
        sequences = []

        for i in range(len(self.base_dataset) - self.sequence_length):
            seq_cloudy = []
            for j in range(self.sequence_length):
                cloudy, _ = self.base_dataset[i + j]
                seq_cloudy.append(cloudy)

            # Target is the clean version of last image
            _, clean = self.base_dataset[i + self.sequence_length - 1]

            sequences.append((torch.stack(seq_cloudy), clean))

        return sequences


# ==================== USAGE EXAMPLES ====================

def quick_start_example():
    """Quick start example with minimal dataset"""
    print("\n### QUICK START EXAMPLE ###\n")

    # Create small dataset
    preparer = SatelliteDatasetPreparer('./quick_test')
    clean_dir, cloudy_dir = preparer.create_synthetic_dataset(n_samples=20)

    # Load dataset
    dataset = SatelliteDataset(clean_dir, cloudy_dir)

    # Train single model
    trainer = ModelTrainer('SimpleCNN')
    models, _ = trainer.train_kfold(dataset, k=2, epochs=5, batch_size=4)

    # Quick evaluation
    test_loader = DataLoader(dataset, batch_size=4)
    evaluator = ModelEvaluator()
    evaluator.evaluate_model(models[0], test_loader, 'SimpleCNN')

    print("\nQuick start completed!")


def advanced_example():
    """Advanced example with custom configurations"""
    print("\n### ADVANCED EXAMPLE ###\n")

    # Custom configuration
    config = {
        'n_samples': 200,
        'img_size': (512, 512),
        'n_bands': 13,  # Full Sentinel-2 bands
        'k_folds': 3,
        'epochs': 30,
        'batch_size': 16,
        'learning_rate': 0.0001
    }

    print("Advanced configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nTo run advanced example:")
    print("1. Increase dataset size and image resolution")
    print("2. Use all 13 Sentinel-2 bands")
    print("3. Train longer with more folds")
    print("4. Implement data augmentation")
    print("5. Use learning rate scheduling")
    print("6. Add early stopping")


if __name__ == '__main__':
    # Run main pipeline
    main()

    # Uncomment for quick start or advanced examples
    #quick_start_example()
    # advanced_example()

    """
    Validation methods . pixel similiarity error, brightness,contrast,
    
    autotune vs no autotune
    cpu/ram bottleneck

    2. Implementing Sentinel-2 data:
        - 
    
    3. Model improvements:
       - Add attention mechanisms
       - Use perceptual loss functions
       - Add temporal consistency constraints
    
    4. Evaluation enhancements:
       - Calculate per-band metrics
       - Add cloud coverage stratified analysis
       - Implement visual quality metrics (FID, LPIPS)
       - Compare with traditional methods (linear interpolation)

    """