"""
Self-Supervised Pipeline for Cloud Detection & Image Reconstruction
===================================================================
Fully self-supervised approach - no manual labels required
Learns cloud masks and performs cloud removal using masked autoencoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import warnings
import yaml
from data_loader import SEN12MSCRDataset
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================

@dataclass
class SelfSupervisedConfig:
    """Central configuration for self-supervised pipeline"""

    # Model architecture
    use_s1: bool = True
    encoder_type: str = 'unet'  # 'unet', 'vit', 'hybrid'
    latent_dim: int = 512

    # Training
    patch_size: int = 256
    batch_size: int = 8
    epochs_pretrain: int = 5
    epochs_reconstruction: int = 5

    # Losses
    spectral_l1: bool = True
    sam_loss: bool = True
    temporal_consistency: bool = True
    uncertainty_weight: float = 0.3

    # Cloud detection
    cloud_threshold: float = 0.5
    min_cloud_fraction: float = 0.05
    max_cloud_fraction: float = 0.9

    # Masking (for pretraining)
    spatial_mask_ratio: float = 0.3
    spectral_mask_ratio: float = 0.2
    mask_block_size: int = 16

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5



# ==================== STAGE 1: MASKED AUTOENCODER (PRETRAINING) ====================

class MaskedMultispectralAutoencoder(nn.Module):
    """
    Self-supervised pretraining via masked reconstruction
    Learns spectral + spatial consistency without cloud labels
    """

    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()
        self.config = config

        # Determine input channels
        self.in_channels = 13  # S2 bands
        if config.use_s1:
            self.in_channels += 2  # Add S1 VV, VH

        # Shared encoder
        if config.encoder_type == 'unet':
            self.encoder = UNetEncoder(self.in_channels, config.latent_dim)
        elif config.encoder_type == 'vit':
            self.encoder = ViTEncoder(self.in_channels, config.latent_dim)
        elif config.encoder_type == 'hybrid':
            self.encoder = HybridEncoder(self.in_channels, config.latent_dim)

        # Decoder (reconstruction)
        self.decoder = UNetDecoder(config.latent_dim, 13)  # Only reconstruct S2

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, H, W) - S2 + optional S1
            mask: (B, 1, H, W) - binary mask (1=keep, 0=mask)

        Returns:
            reconstructed: (B, 13, H, W) - S2 reconstruction
            latent: (B, latent_dim, H', W') - encoded features
        """
        # Apply mask to input
        if mask is not None:
            x_masked = x * mask
        else:
            x_masked = x

        # Encode
        latent = self.encoder(x_masked)

        # Decode (reconstruct S2 only)
        reconstructed = self.decoder(latent)

        return reconstructed, latent

    def generate_masks(self, batch_size, H, W, device):
        """
        Generate random spatial and spectral masks

        Returns:
            spatial_mask: (B, 1, H, W)
            spectral_mask: (B, C, 1, 1)
        """
        # Spatial mask (block-wise)
        block_size = self.config.mask_block_size
        n_blocks_h = H // block_size
        n_blocks_w = W // block_size

        spatial_mask = torch.ones(batch_size, 1, H, W, device=device)

        for b in range(batch_size):
            # Randomly mask blocks
            n_mask = int(n_blocks_h * n_blocks_w * self.config.spatial_mask_ratio)
            mask_indices = torch.randperm(n_blocks_h * n_blocks_w)[:n_mask]

            for idx in mask_indices:
                i = idx // n_blocks_w
                j = idx % n_blocks_w
                spatial_mask[b, :,
                i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size] = 0

        # Spectral mask (drop random bands)
        n_bands = 13
        spectral_mask = torch.ones(batch_size, n_bands, 1, 1, device=device)

        for b in range(batch_size):
            n_mask = int(n_bands * self.config.spectral_mask_ratio)
            mask_bands = torch.randperm(n_bands)[:n_mask]
            spectral_mask[b, mask_bands, :, :] = 0

        # Combine masks
        combined_mask = spatial_mask * spectral_mask

        return combined_mask


class UNetEncoder(nn.Module):
    """U-Net style encoder"""

    def __init__(self, in_channels, latent_dim):
        super().__init__()

        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        self.bottleneck = self._conv_block(512, latent_dim)

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
        latent = self.bottleneck(self.pool(e4))

        return latent


class ViTEncoder(nn.Module):
    """Vision Transformer encoder"""

    def __init__(self, in_channels, latent_dim):
        super().__init__()

        # Patch embedding
        self.patch_size = 16
        self.patch_embed = nn.Conv2d(
            in_channels, latent_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=8,
                dim_feedforward=latent_dim * 4,
                batch_first=True
            ),
            num_layers=6
        )

    def forward(self, x):
        # Patch embedding
        patches = self.patch_embed(x)  # (B, latent_dim, H/16, W/16)

        B, C, H, W = patches.shape

        # Flatten patches
        patches = patches.flatten(2).transpose(1, 2)  # (B, H*W, latent_dim)

        # Transformer
        features = self.transformer(patches)

        # Reshape back
        features = features.transpose(1, 2).view(B, C, H, W)

        return features


class HybridEncoder(nn.Module):
    """Hybrid CNN + Transformer encoder"""

    def __init__(self, in_channels, latent_dim):
        super().__init__()

        # CNN stem
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU()
        )

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=8,
                dim_feedforward=latent_dim * 4,
                batch_first=True
            ),
            num_layers=4
        )

    def forward(self, x):
        # CNN feature extraction
        features = self.cnn_stem(x)  # (B, latent_dim, H/8, W/8)

        B, C, H, W = features.shape

        # Flatten and apply transformer
        flat = features.flatten(2).transpose(1, 2)
        transformed = self.transformer(flat)

        # Reshape
        output = transformed.transpose(1, 2).view(B, C, H, W)

        return output


class UNetDecoder(nn.Module):
    """U-Net style decoder"""

    def __init__(self, latent_dim, out_channels):
        super().__init__()

        self.up4 = nn.ConvTranspose2d(latent_dim, 512, 2, stride=2)
        self.dec4 = self._conv_block(512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(64, 64)

        self.out = nn.Conv2d(64, out_channels, 1)

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
        x = self.up4(x)
        x = self.dec4(x)
        x = self.up3(x)
        x = self.dec3(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up1(x)
        x = self.dec1(x)
        return self.out(x)


# ==================== STAGE 2: CLOUD UNCERTAINTY ESTIMATION ====================

class CloudUncertaintyEstimator:
    """
    Estimate cloud probability from reconstruction error
    High error → likely cloud
    """

    def __init__(self, config: SelfSupervisedConfig):
        self.config = config

    def estimate(self, original, reconstructed, smooth_sigma=2.0):
        """
        Compute cloud probability from reconstruction error

        Args:
            original: (B, 13, H, W) - S2 original
            reconstructed: (B, 13, H, W) - S2 reconstruction
            smooth_sigma: Gaussian smoothing sigma

        Returns:
            cloud_prob: (B, 1, H, W) - probability map [0, 1]
        """
        # Per-pixel reconstruction error
        error = torch.abs(original - reconstructed)

        # Aggregate across spectral bands
        error = error.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Normalize to [0, 1]
        error_normalized = self._normalize_batch(error)

        # Smooth
        if smooth_sigma > 0:
            error_normalized = self._gaussian_smooth(error_normalized, smooth_sigma)

        return error_normalized

    @staticmethod
    def _normalize_batch(x):
        """Normalize each sample in batch to [0, 1]"""
        B = x.shape[0]
        x_flat = x.view(B, -1)

        min_vals = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_vals = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)

        normalized = (x - min_vals) / (max_vals - min_vals + 1e-8)

        return normalized

    @staticmethod
    def _gaussian_smooth(x, sigma):
        """Apply Gaussian smoothing"""
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        coords -= kernel_size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        # 2D kernel
        kernel = g[:, None] * g[None, :]
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # Apply convolution
        padding = kernel_size // 2
        smoothed = F.conv2d(x, kernel, padding=padding)

        return smoothed


# ==================== STAGE 3: JOINT CLOUD DETECTION + RECONSTRUCTION MODEL ====================

class JointCloudModel(nn.Module):
    """
    Joint model for cloud detection and image reconstruction

    Architecture:
        Input (S2 + S1) → Encoder → ┬→ Cloud Head (P_cloud)
                                     └→ Decoder (S2_clean)
    """

    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()
        self.config = config

        # Determine input channels
        self.in_channels = 13
        if config.use_s1:
            self.in_channels += 2

        # Shared encoder
        if config.encoder_type == 'unet':
            self.encoder = UNetEncoder(self.in_channels, config.latent_dim)
        elif config.encoder_type == 'vit':
            self.encoder = ViTEncoder(self.in_channels, config.latent_dim)
        elif config.encoder_type == 'hybrid':
            self.encoder = HybridEncoder(self.in_channels, config.latent_dim)

        # Cloud detection head
        self.cloud_head = nn.Sequential(
            nn.Conv2d(config.latent_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

        # Image reconstruction decoder
        self.decoder = UNetDecoder(config.latent_dim, 13)

        # Upsampling for cloud head (match input resolution)
        self.cloud_upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - S2 + optional S1

        Returns:
            cloud_prob: (B, 1, H, W) - cloud probability
            reconstructed: (B, 13, H, W) - clean S2
            latent: (B, latent_dim, H', W')
        """
        # Encode
        latent = self.encoder(x)

        # Cloud detection
        cloud_prob = self.cloud_head(latent)
        cloud_prob = self.cloud_upsample(cloud_prob)

        # Image reconstruction
        reconstructed = self.decoder(latent)

        return cloud_prob, reconstructed, latent


# ==================== LOSSES ====================

class SpectralAngleLoss(nn.Module):
    """Spectral Angle Mapper (SAM) loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: (B, C, H, W)
            target: (B, C, H, W)
            mask: (B, 1, H, W) - optional mask
        """
        # Normalize
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)

        # Cosine similarity
        cos_sim = (pred_norm * target_norm).sum(dim=1, keepdim=True)
        cos_sim = torch.clamp(cos_sim, -1, 1)

        # Angle
        angle = torch.acos(cos_sim)

        # Apply mask
        if mask is not None:
            angle = angle * mask
            loss = angle.sum() / (mask.sum() + 1e-8)
        else:
            loss = angle.mean()

        return loss


class TemporalConsistencyLoss(nn.Module):
    """Enforce consistency across temporal samples"""

    def __init__(self):
        super().__init__()

    def forward(self, recon1, recon2):
        """
        Args:
            recon1, recon2: (B, 13, H, W) - reconstructions from different times
        """
        # L1 difference
        diff = torch.abs(recon1 - recon2)
        return diff.mean()


class SelfSupervisedLoss(nn.Module):
    """Combined loss for self-supervised training"""

    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()
        self.config = config

        self.l1_loss = nn.L1Loss()
        self.sam_loss = SpectralAngleLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred_dict, target_dict):
        """
        Args:
            pred_dict: {
                'cloud_prob': (B, 1, H, W),
                'reconstructed': (B, 13, H, W),
                'latent': ...
            }
            target_dict: {
                's2_clean': (B, 13, H, W),
                'cloud_mask': (B, 1, H, W) - pseudo labels,
                'mask_region': (B, 1, H, W) - where to apply loss
            }
        """
        losses = {}

        # Reconstruction loss (masked regions only)
        mask = target_dict.get('mask_region', None)

        if self.config.spectral_l1:
            l1 = self.l1_loss(
                pred_dict['reconstructed'] * mask,
                target_dict['s2_clean'] * mask
            ) if mask is not None else self.l1_loss(
                pred_dict['reconstructed'],
                target_dict['s2_clean']
            )
            losses['l1'] = l1

        if self.config.sam_loss:
            sam = self.sam_loss(
                pred_dict['reconstructed'],
                target_dict['s2_clean'],
                mask
            )
            losses['sam'] = sam

        # Cloud detection loss
        if 'cloud_mask' in target_dict:
            cloud_loss = self.bce_loss(
                pred_dict['cloud_prob'],
                target_dict['cloud_mask']
            )
            losses['cloud'] = cloud_loss * self.config.uncertainty_weight

        # Total loss
        total = sum(losses.values())
        losses['total'] = total

        return losses


# ==================== TRAINING PIPELINE ====================

class SelfSupervisedTrainer:
    """
    Complete self-supervised training pipeline

    Stage 1: Masked autoencoding (pretraining)
    Stage 2: Cloud uncertainty estimation
    Stage 3: Joint cloud detection + reconstruction
    """

    def __init__(self, config: SelfSupervisedConfig, device='cuda'):
        self.config = config
        self.device = device

        # Models
        self.pretrain_model = None
        self.joint_model = None

        # Optimizer
        self.optimizer = None

        # Losses
        self.criterion = SelfSupervisedLoss(config)

        # Cloud estimator
        self.cloud_estimator = CloudUncertaintyEstimator(config)

    def stage1_pretrain(self, train_loader, val_loader):
        """
        Stage 1: Self-supervised pretraining via masked reconstruction
        """
        print("\n" + "=" * 70)
        print("STAGE 1: MASKED AUTOENCODING PRETRAINING")
        print("=" * 70)

        # Initialize model
        self.pretrain_model = MaskedMultispectralAutoencoder(self.config).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.pretrain_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.config.epochs_pretrain):
            # Train
            train_loss = self._train_pretrain_epoch(train_loader)
            history['train_loss'].append(train_loss)

            # Validate
            val_loss = self._validate_pretrain_epoch(val_loader)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{self.config.epochs_pretrain} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.pretrain_model.state_dict(), 'pretrain_best.pth')

        # Load best
        self.pretrain_model.load_state_dict(torch.load('pretrain_best.pth'))

        return history

    def _train_pretrain_epoch(self, loader):
        """Single pretraining epoch"""
        self.pretrain_model.train()
        total_loss = 0

        for batch_idx, data in enumerate(loader):
            # Unpack data
            if len(data) == 4:
                s1, s2_cloudy, s2_clean, _ = data
            else:
                s2_cloudy, s2_clean = data
                s1 = None

            # Move to device
            s2_clean = s2_clean.to(self.device)

            # Prepare input
            if self.config.use_s1 and s1 is not None:
                s1 = s1.to(self.device)
                x = torch.cat([s2_clean, s1], dim=1)
            else:
                x = s2_clean

            # Generate masks
            masks = self.pretrain_model.generate_masks(
                x.shape[0], x.shape[2], x.shape[3], self.device
            )

            # Forward
            reconstructed, _ = self.pretrain_model(x, masks)

            # Loss (on S2 only)
            loss = F.l1_loss(reconstructed, s2_clean)

            # SAM loss
            sam = SpectralAngleLoss()(reconstructed, s2_clean)
            loss = loss + 0.1 * sam

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate_pretrain_epoch(self, loader):
        """Validation epoch"""
        self.pretrain_model.eval()
        total_loss = 0

        with torch.no_grad():
            for data in loader:
                if len(data) == 4:
                    s1, s2_cloudy, s2_clean, _ = data
                else:
                    s2_cloudy, s2_clean = data
                    s1 = None

                s2_clean = s2_clean.to(self.device)

                if self.config.use_s1 and s1 is not None:
                    s1 = s1.to(self.device)
                    x = torch.cat([s2_clean, s1], dim=1)
                else:
                    x = s2_clean

                masks = self.pretrain_model.generate_masks(
                    x.shape[0], x.shape[2], x.shape[3], self.device
                )

                reconstructed, _ = self.pretrain_model(x, masks)
                loss = F.l1_loss(reconstructed, s2_clean)

                total_loss += loss.item()

        return total_loss / len(loader)

    def stage2_generate_pseudo_labels(self, dataset):
        """
        Stage 2: Generate pseudo cloud labels using reconstruction error

        Returns:
            pseudo_labels: List of cloud masks
        """
        print("\n" + "=" * 70)
        print("STAGE 2: GENERATING PSEUDO CLOUD LABELS")
        print("=" * 70)

        self.pretrain_model.eval()
        pseudo_labels = []

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

        with torch.no_grad():
            for data in loader:
                if len(data) == 4:
                    s1, s2_cloudy, s2_clean, _ = data
                else:
                    s2_cloudy, s2_clean = data
                    s1 = None

                s2_cloudy = s2_cloudy.to(self.device)

                # Prepare input
                if self.config.use_s1 and s1 is not None:
                    s1 = s1.to(self.device)
                    x = torch.cat([s2_cloudy, s1], dim=1)
                else:
                    x = s2_cloudy

                # Reconstruct
                reconstructed, _ = self.pretrain_model(x, mask=None)

                # Estimate cloud uncertainty
                cloud_prob = self.cloud_estimator.estimate(s2_cloudy, reconstructed)

                # Store
                for i in range(cloud_prob.shape[0]):
                    pseudo_labels.append(cloud_prob[i].cpu().numpy())

        print(f"✓ Generated {len(pseudo_labels)} pseudo cloud masks")

        return pseudo_labels

    def stage3_joint_training(self, train_loader, val_loader, pseudo_labels):
        """
        Stage 3: Train joint cloud detection + reconstruction model
        """
        print("\n" + "=" * 70)
        print("STAGE 3: JOINT CLOUD DETECTION + RECONSTRUCTION")
        print("=" * 70)
        if self.pretrain_model is None:
            # We assume the default output directory structure
            checkpoint_path = Path("results/selfsupervised/pretrain_best.pth")
            if not checkpoint_path.exists():
                # Fallback to current directory if not in results
                checkpoint_path = Path("pretrain_best.pth")

            if checkpoint_path.exists():
                print(f"Restoring pretrained model from {checkpoint_path}...")
                self.pretrain_model = MaskedMultispectralAutoencoder(self.config).to(self.device)
                self.pretrain_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            else:
                raise RuntimeError(
                    "Stage 3 requires a pretrained model. Run Stage 1 first or "
                    "ensure 'pretrain_best.pth' exists."
                )
        # Initialize joint model
        self.joint_model = JointCloudModel(self.config).to(self.device)

        # Transfer encoder weights from pretrained model
        self.joint_model.encoder.load_state_dict(
            self.pretrain_model.encoder.state_dict()
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.joint_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.config.epochs_reconstruction):
            train_loss = self._train_joint_epoch(train_loader, pseudo_labels)
            history['train_loss'].append(train_loss)

            val_loss = self._validate_joint_epoch(val_loader, pseudo_labels)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{self.config.epochs_reconstruction} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.joint_model.state_dict(), 'joint_model_best.pth')

        self.joint_model.load_state_dict(torch.load('joint_model_best.pth'))

        return history

    def _train_joint_epoch(self, loader, pseudo_labels):
        self.joint_model.train()
        total_loss = 0
        for batch_idx, data in enumerate(loader):
            if len(data) == 4:
                s1, s2_cloudy, s2_clean, _ = data
            else:
                s2_cloudy, s2_clean = data
                s1 = None

            s2_cloudy = s2_cloudy.to(self.device)
            s2_clean = s2_clean.to(self.device)

            # Prepare input
            if self.config.use_s1 and s1 is not None:
                s1 = s1.to(self.device)
                x = torch.cat([s2_cloudy, s1], dim=1)
            else:
                x = s2_cloudy

            # Get pseudo labels for this batch
            batch_start = batch_idx * loader.batch_size
            batch_pseudo = torch.from_numpy(
                np.stack(pseudo_labels[batch_start:batch_start + x.shape[0]])
            ).to(self.device)

            # Forward
            cloud_prob, reconstructed, _ = self.joint_model(x)

            # Prepare targets
            target_dict = {
                's2_clean': s2_clean,
                'cloud_mask': batch_pseudo,
                'mask_region': batch_pseudo > 0.3  # Focus on cloudy regions
            }

            pred_dict = {
                'cloud_prob': cloud_prob,
                'reconstructed': reconstructed
            }

            # Compute loss
            losses = self.criterion(pred_dict, target_dict)
            loss = losses['total']

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate_joint_epoch(self, loader, pseudo_labels):
        """Validation epoch for joint model"""
        self.joint_model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                if len(data) == 4:
                    s1, s2_cloudy, s2_clean, _ = data
                else:
                    s2_cloudy, s2_clean = data
                    s1 = None

                s2_cloudy = s2_cloudy.to(self.device)
                s2_clean = s2_clean.to(self.device)

                if self.config.use_s1 and s1 is not None:
                    s1 = s1.to(self.device)
                    x = torch.cat([s2_cloudy, s1], dim=1)
                else:
                    x = s2_cloudy

                batch_start = batch_idx * loader.batch_size
                batch_pseudo = torch.from_numpy(
                    np.stack(pseudo_labels[batch_start:batch_start + x.shape[0]])
                ).to(self.device)

                cloud_prob, reconstructed, _ = self.joint_model(x)

                target_dict = {
                    's2_clean': s2_clean,
                    'cloud_mask': batch_pseudo,
                    'mask_region': batch_pseudo > 0.3
                }

                pred_dict = {
                    'cloud_prob': cloud_prob,
                    'reconstructed': reconstructed
                }

                losses = self.criterion(pred_dict, target_dict)
                total_loss += losses['total'].item()

        return total_loss / len(loader)

    def inference(self, s1, s2_cloudy):
        """
        Inference pipeline

        Args:
            s1: (B, 2, H, W) or None
            s2_cloudy: (B, 13, H, W)

        Returns:
            cloud_mask: (B, 1, H, W)
            reconstructed: (B, 13, H, W)
        """
        self.joint_model.eval()

        with torch.no_grad():
            # Prepare input
            if self.config.use_s1 and s1 is not None:
                x = torch.cat([s2_cloudy, s1], dim=1)
            else:
                x = s2_cloudy

            # Forward
            cloud_prob, reconstructed, _ = self.joint_model(x)

        return cloud_prob, reconstructed
def compute_selfsupervised_cloud_mask(s1, s2_cloudy, model_path, config, device='cuda'):
    """
    Args:
    s1: (2, H, W) or None
    s2_cloudy: (13, H, W)
    model_path: Path
    to
    trained
    model
    config: SelfSupervisedConfig
    device: Device


    Returns:
    cloud_mask: (1, H, W)
    reconstructed: (13, H, W)
    """
    # Load model
    model = JointCloudModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare input
    if isinstance(s2_cloudy, np.ndarray):
        s2_cloudy = torch.from_numpy(s2_cloudy).float()
    if s1 is not None and isinstance(s1, np.ndarray):
        s1 = torch.from_numpy(s1).float()

    # Add batch dimension
    s2_cloudy = s2_cloudy.unsqueeze(0).to(device)

    if config.use_s1 and s1 is not None:
        s1 = s1.unsqueeze(0).to(device)
        x = torch.cat([s2_cloudy, s1], dim=1)
    else:
        x = s2_cloudy

    # Inference
    with torch.no_grad():
        cloud_prob, reconstructed, _ = model(x)

    # Remove batch dimension
    cloud_mask = cloud_prob[0].cpu().numpy()
    reconstructed = reconstructed[0].cpu().numpy()

    return cloud_mask, reconstructed

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data_loader import SEN12MSCRDataset


def _build_loaders(
    dataset_root: str,
    patch_size: int,
    batch_size: int,
    data_fraction: float,
    val_split: float,
    seed: int,
    num_workers: int,
):
    """
    Create dataset + reproducible train/val split + dataloaders.
    Stage-3 loaders use shuffle=False to keep pseudo-label alignment stable.
    """
    dataset = SEN12MSCRDataset(
        root_dir=dataset_root,
        patch_size=patch_size,
        data_fraction=data_fraction,
        random_seed=seed,
        cloud_mask_mode="simple",  # keep robust; deep mask can be added later if needed
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = max(1, n_total - n_val)

    # Reproducible split
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    # Stage 1: you can shuffle training
    train_loader_s1 = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader_s1 = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    # Stage 3: keep order stable (shuffle=False) for pseudo-label indexing logic
    train_loader_s3 = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    val_loader_s3 = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    return dataset, train_ds, val_ds, train_loader_s1, val_loader_s1, train_loader_s3, val_loader_s3


def run_pipeline(
    dataset_root: str,
    out_dir: str,
    device: str,
    seed: int,
    data_fraction: float,
    val_split: float,
    num_workers: int,
    config: "SelfSupervisedConfig",
    run_stage1: bool,
    run_stage2: bool,
    run_stage3: bool,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build data
    dataset, train_ds, val_ds, train_loader_s1, val_loader_s1, train_loader_s3, val_loader_s3 = _build_loaders(
        dataset_root=dataset_root,
        patch_size=config.patch_size,
        batch_size=config.batch_size,
        data_fraction=data_fraction,
        val_split=val_split,
        seed=seed,
        num_workers=num_workers,
    )

    print("\n" + "=" * 70)
    print("SELF-SUPERVISED PIPELINE RUNNER")
    print("=" * 70)
    print(f"Device:        {device}")
    print(f"Dataset root:  {dataset_root}")
    print(f"Samples:       total={len(dataset)} train={len(train_ds)} val={len(val_ds)}")
    print(f"Data fraction: {data_fraction}")
    print(f"Val split:     {val_split}")
    print(f"Output dir:    {out_dir}")
    print("=" * 70)

    # Save config for reproducibility

    trainer = SelfSupervisedTrainer(config, device=device)

    # --- Stage 1 ---
    if run_stage1:
        history1 = trainer.stage1_pretrain(train_loader_s1, val_loader_s1)
        # Move checkpoint into out_dir (stage code saves to fixed name)
        src = Path("pretrain_best.pth")
        if src.exists():
            dst = out_dir / "pretrain_best.pth"
            src.replace(dst)
            print(f"✓ Saved Stage-1 checkpoint: {dst}")
        else:
            print("⚠ Stage-1 checkpoint not found (pretrain_best.pth).")

    # --- Stage 2 ---
    pseudo_labels = None
    if run_stage2:
        # Generate pseudo labels for the TRAIN split (must match stage-3 loader order)
        pseudo_labels = trainer.stage2_generate_pseudo_labels(train_ds)

        pseudo_path = out_dir / "pseudo_labels_train.npy"
        import numpy as np
        np.save(pseudo_path, np.stack(pseudo_labels))
        print(f"✓ Saved pseudo labels: {pseudo_path}")

    # --- Stage 3 ---
    if run_stage3:
        if pseudo_labels is None:
            # Try loading from disk if Stage 2 wasn't run in this session
            pseudo_path = out_dir / "pseudo_labels_train.npy"
            if not pseudo_path.exists():
                raise RuntimeError(
                    "Stage 3 requires pseudo labels. Run with --stage2 first "
                    "or ensure pseudo_labels_train.npy exists in out_dir."
                )
            import numpy as np
            pseudo_labels = list(np.load(pseudo_path))
            print(f"✓ Loaded pseudo labels: {pseudo_path}")

        history3 = trainer.stage3_joint_training(train_loader_s3, val_loader_s3, pseudo_labels)

        src = Path("joint_model_best.pth")
        if src.exists():
            dst = out_dir / "joint_model_best.pth"
            src.replace(dst)
            print(f"✓ Saved Stage-3 checkpoint: {dst}")
        else:
            print("⚠ Stage-3 checkpoint not found (joint_model_best.pth).")

    print("\n✓ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run self-supervised stages (1/2/3) from SelfSupervisedTest.py")
    parser.add_argument("--dataset-root", type=str, default="./sen12mscr_dataset")
    parser.add_argument("--out-dir", type=str, default="./results/selfsupervised")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-fraction", type=float, default=0.05)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 first on Windows for stability")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu; default auto-detect")

    # Stage switches
    parser.add_argument("--stage1", action="store_true", help="Run Stage 1 (pretrain)")
    parser.add_argument("--stage2", action="store_true", help="Run Stage 2 (pseudo labels)")
    parser.add_argument("--stage3", action="store_true", help="Run Stage 3 (joint training)")
    parser.add_argument("--all", action="store_true", help="Run all stages in order")

    # Lightweight config overrides
    parser.add_argument("--epochs-pretrain", type=int, default=3)
    parser.add_argument("--epochs-reconstruction", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--encoder-type", type=str, default="unet", choices=["unet", "vit", "hybrid"])
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--use-s1", action="store_true", help="Use S1 (VV/VH) along with S2")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Decide which stages to run
    run_stage1 = args.all or args.stage1
    run_stage2 = args.all or args.stage2
    run_stage3 = args.all or args.stage3
    if not (run_stage1 or run_stage2 or run_stage3):
        # Default: run everything if user didn't specify
        run_stage1 = run_stage2 = run_stage3 = True

    config = SelfSupervisedConfig(
        use_s1=args.use_s1,
        encoder_type=args.encoder_type,
        latent_dim=args.latent_dim,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs_pretrain=args.epochs_pretrain,
        epochs_reconstruction=args.epochs_reconstruction,
    )

    run_pipeline(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        device=device,
        seed=args.seed,
        data_fraction=args.data_fraction,
        val_split=args.val_split,
        num_workers=args.num_workers,
        config=config,
        run_stage1=run_stage1,
        run_stage2=run_stage2,
        run_stage3=run_stage3,
    )
