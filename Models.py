# ==================== 3. MODEL IMPLEMENTATIONS ====================
import torch.nn as nn
import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor


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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
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
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),          nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),         nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),          nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Model 3: GAN (Generator + Discriminator)
class Generator(nn.Module):
    """GAN Generator - ResNet-based architecture"""

    def __init__(self, in_channels=4, out_channels=4):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down1 = self._downsample_block(64, 128)
        self.down2 = self._downsample_block(128, 256)
        self.res_blocks = nn.Sequential(*[self._residual_block(256) for _ in range(6)])
        self.up1 = self._upsample_block(256, 128)
        self.up2 = self._upsample_block(128, 64)
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Sigmoid()
        )

    def _downsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def _upsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_c), nn.ReLU(inplace=True)
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
    """Residual block for GAN Generator"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Discriminator(nn.Module):
    """GAN Discriminator (PatchGAN)"""

    def __init__(self, in_channels=8):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),   nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),           nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),          nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, padding=1),                    nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


# ==================== MODEL 4: DSen2-CR (replaces LSTM) ====================

class DSen2CRBlock(nn.Module):
    """
    Single residual block used inside DSen2-CR.

    Mirrors the ResnetBlock from dsen2cr_pytorch_model.py:
        CONV → ReLU → CONV
    scaled by res_scale before the identity add.
    """

    def __init__(self, feature_size: int = 256, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv_block = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.conv_block(x)


class DSen2CR(nn.Module):
    """
    PyTorch port of the DSen2-CR deep residual cloud removal network.

    Reference
    ---------
    Meraner et al. (2020) – Cloud removal in Sentinel-2 imagery using a deep
    residual neural network and SAR-optical data fusion.
    ISPRS Journal of Photogrammetry and Remote Sensing, 166, 333-346.
    https://doi.org/10.1016/j.isprsjprs.2020.05.013

    Original PyTorch code: Code/dsen2cr_pytorch_model.py (ResnetStackedArchitecture)

    Architecture
    ------------
    Input : cat(S2_cloudy[n_s2], S1[2])   shape [B, n_s2+2, H, W]
    ┌─ head conv + ReLU → F channels
    ├─ B residual blocks (each: Conv-ReLU-Conv, scaled by res_scale)
    └─ tail conv → n_s2 channels
    Output = head_input[:, :n_s2] + tail(x)   ← long skip connection

    Differences from the original Keras model
    ------------------------------------------
    * Kaiming-uniform weight initialisation (same as pytorch version).
    * No gradient clipping built-in; handled by the trainer.
    * n_s2 is configurable (default 13 for all S2 bands; 4 for RGB+NIR).
    * num_layers (B) and feature_size (F) match paper defaults: 16 / 256.

    Input channel convention (must match data_loader + training_functions)
    -----------------------------------------------------------------------
    The pipeline concatenates [S1, S2_cloudy] → [S1[2], S2_cloudy[n_s2]].
    DSen2-CR originally uses [S2_cloudy, S1], so we reorder inside forward()
    to keep the rest of the codebase unchanged.
    """

    def __init__(
        self,
        in_channels: int = 15,   # S1(2) + S2(13) by default; set to 2+n_s2
        out_channels: int = 13,  # equals n_s2
        num_layers: int = 16,    # B in the paper
        feature_size: int = 256, # F in the paper
        res_scale: float = 0.1,
    ):
        super().__init__()

        self.n_s2 = out_channels  # store for long-skip indexing
        self.n_s1 = in_channels - out_channels  # = 2

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Body: B residual blocks
        self.body = nn.Sequential(
            *[DSen2CRBlock(feature_size, res_scale) for _ in range(num_layers)]
        )

        # Tail: map back to n_s2 bands
        self.tail = nn.Conv2d(feature_size, out_channels, kernel_size=3, padding=1, bias=True)

        # Weight initialisation (He uniform, same as original)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, n_s1+n_s2, H, W]
            Channels: [S1(2 bands), S2_cloudy(n_s2 bands)]
            (matches the cat([s1, s2_cloudy], dim=1) convention used everywhere
            else in this codebase)

        Returns
        -------
        Tensor [B, n_s2, H, W]  –  reconstructed cloud-free S2
        """
        # Reorder to [S2_cloudy, S1] as the original model expects
        s1      = x[:, :self.n_s1]           # [B, 2, H, W]
        s2_in   = x[:, self.n_s1:]           # [B, n_s2, H, W]
        x_cat   = torch.cat([s2_in, s1], dim=1)   # [B, n_s2+2, H, W]

        features = self.head(x_cat)
        features = self.body(features)
        residual = self.tail(features)

        # Long skip: add cloudy S2 optical input (same as original model)
        return s2_in + residual


# ==================== MODEL 5: Diffusion Model ====================

class DiffusionModel(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) for cloud removal.
    Dynamic in_channels: works with 4, 13, or 15 bands.
    """

    def __init__(self, in_channels=4, out_channels=4, model_channels=64, num_res_blocks=2):
        super(DiffusionModel, self).__init__()

        self.in_channels    = in_channels
        self.model_channels = model_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList([nn.Conv2d(in_channels, model_channels, 3, padding=1)])

        ch = model_channels
        for level in range(3):
            for _ in range(num_res_blocks):
                self.input_blocks.append(ResBlock(ch, time_embed_dim, ch))
            if level < 2:
                self.input_blocks.append(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1))
                ch *= 2

        self.middle_block = nn.Sequential(
            ResBlock(ch, time_embed_dim, ch),
            ResBlock(ch, time_embed_dim, ch),
        )

        self.output_blocks = nn.ModuleList([])
        for level in range(3):
            for _ in range(num_res_blocks):
                self.output_blocks.append(ResBlock(ch, time_embed_dim, ch))
            if level < 2:
                self.output_blocks.append(nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1))
                ch //= 2

        self.out = nn.Sequential(
            nn.GroupNorm(32, model_channels), nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        t_emb = self._get_timestep_embedding(timesteps, self.model_channels)
        emb   = self.time_embed(t_emb)
        h, hs = x, []
        for module in self.input_blocks:
            h = module(h, emb) if isinstance(module, ResBlock) else module(h)
            hs.append(h)
        h = self.middle_block[0](h, emb)
        h = self.middle_block[1](h, emb)
        for module in self.output_blocks:
            h = module(h, emb) if isinstance(module, ResBlock) else module(h)
        return self.out(h)

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time embedding for Diffusion Model"""

    def __init__(self, channels, time_embed_dim, out_channels):
        super(ResBlock, self).__init__()
        self.in_layers  = nn.Sequential(
            nn.GroupNorm(32, channels), nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(time_embed_dim, out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip_connection = (
            nn.Conv2d(channels, out_channels, 1)
            if channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        h = h + self.emb_layers(emb)[:, :, None, None]
        h = self.out_layers(h)
        return self.skip_connection(x) + h


# ==================== Random Forest ====================

class RandomForestCloudRemover:
    """Random Forest model for pixel-level cloud removal."""

    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1
        )
        self.is_fitted = False

    def _prepare_features(self, s1, s2_cloudy, cloud_mask):
        if isinstance(s1, torch.Tensor):        s1         = s1.cpu().numpy()
        if isinstance(s2_cloudy, torch.Tensor): s2_cloudy  = s2_cloudy.cpu().numpy()
        if isinstance(cloud_mask, torch.Tensor):cloud_mask = cloud_mask.cpu().numpy()

        if len(s1.shape) == 4:
            b, c1, h, w = s1.shape
            c2 = s2_cloudy.shape[1]; cm = cloud_mask.shape[1]
            s1_flat   = s1.transpose(0,2,3,1).reshape(-1, c1)
            s2_flat   = s2_cloudy.transpose(0,2,3,1).reshape(-1, c2)
            mask_flat = cloud_mask.transpose(0,2,3,1).reshape(-1, cm)
        else:
            c1, h, w = s1.shape; b = 1
            c2 = s2_cloudy.shape[0]; cm = cloud_mask.shape[0]
            s1_flat   = s1.transpose(1,2,0).reshape(-1, c1)
            s2_flat   = s2_cloudy.transpose(1,2,0).reshape(-1, c2)
            mask_flat = cloud_mask.transpose(1,2,0).reshape(-1, cm)

        return np.hstack([s1_flat, s2_flat, mask_flat]), (b, h, w)

    def fit(self, train_loader, device='cpu', max_samples=100000):
        X_list, y_list, samples_collected = [], [], 0
        print(f"Collecting training pixels (target: {max_samples})...")
        for s1, s2_cloudy, s2_clean, cloud_mask in train_loader:
            X_batch, _ = self._prepare_features(s1, s2_cloudy, cloud_mask)
            c_target   = s2_clean.shape[1]
            y_batch    = s2_clean.cpu().numpy().transpose(0,2,3,1).reshape(-1, c_target)
            X_list.append(X_batch); y_list.append(y_batch)
            samples_collected += X_batch.shape[0]
            if samples_collected >= max_samples:
                break
        X = np.vstack(X_list)[:max_samples]
        y = np.vstack(y_list)[:max_samples]
        print(f"Fitting RandomForest on {X.shape[0]} pixels...")
        self.model.fit(X, y)
        self.is_fitted = True
        print("✓ RandomForest fitting complete")

    def predict(self, s1, s2_cloudy, cloud_mask, device='cpu'):
        features, (b, h, w) = self._prepare_features(s1, s2_cloudy, cloud_mask)
        if hasattr(self.model, "n_features_in_"):
            expected = self.model.n_features_in_
            if features.shape[1] != expected:
                if features.shape[1] < expected:
                    padding  = np.zeros((features.shape[0], expected - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    features = features[:, :expected]
        y_pred = self.model.predict(features)
        c_out  = y_pred.shape[1]
        return torch.from_numpy(
            y_pred.reshape(b, h, w, c_out).transpose(0, 3, 1, 2)
        ).float().to(device)


class RandomForestWrapper(nn.Module):
    """PyTorch-compatible wrapper for RandomForestCloudRemover."""

    def __init__(self, rf_model):
        super().__init__()
        self.rf_model = rf_model
        self._device  = 'cpu'

    def predict(self, s1, s2_cloudy, cloud_mask, device='cpu'):
        return self.rf_model.predict(s1, s2_cloudy, cloud_mask, device=device)

    def forward(self, x):
        C_s1     = 2
        s1       = x[:, :C_s1]
        cloud_mask = x[:, -1:]
        s2_cloudy  = x[:, C_s1:-1]
        return self.rf_model.predict(s1, s2_cloudy, cloud_mask, device=self._device)

    def to(self, device):
        self._device = device if isinstance(device, str) else str(device)
        return self

    def eval(self):   return self
    def train(self, mode=True): return self


def integrate_rf_with_pipeline(rf_model):
    return RandomForestWrapper(rf_model)