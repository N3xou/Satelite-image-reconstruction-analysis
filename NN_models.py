# ==================== 3. MODEL IMPLEMENTATIONS ====================
import torch.nn as nn
import torch
import numpy as np
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

