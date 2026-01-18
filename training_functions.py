import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import Dataset, DataLoader
from NN_models import UNet, DiffusionModel, SimpleCNN, Generator, LSTMCloudRemover, Discriminator
from collections import defaultdict


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
    """Handles training with K-Fold cross-validation (FIXED: scene-level splitting)"""

    def __init__(self, model_type, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_type = model_type
        self.device = device
        self.models = []
        self.histories = []
        self.training_time = 0  # Track total training time

        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _get_scene_level_splits(self, dataset, k=5):
        """
        Split dataset at SCENE level to prevent data leakage

        Returns:
            List of (train_indices, val_indices) tuples for each fold
        """
        # Group samples by scene
        scene_to_indices = defaultdict(list)

        for idx, sample in enumerate(dataset.samples):
            # Extract season and scene_id from sample paths
            s2_clean_path = sample['s2_clean']
            # Path format: .../ROIsXXXX_season_s2/s2_X/patch.tif
            parts = s2_clean_path.parts
            season = parts[-3].split('_s2')[0]  # ROIsXXXX_season
            scene_id = parts[-2].split('_')[1]  # s2_X -> X
            scene_key = f"{season}_{scene_id}"
            scene_to_indices[scene_key].append(idx)

        # Get scene keys
        scene_keys = list(scene_to_indices.keys())

        print(f"\nDataset scene-level statistics:")
        print(f"  Total scenes: {len(scene_keys)}")
        print(f"  Total samples: {len(dataset.samples)}")
        print(f"  Avg samples per scene: {len(dataset.samples) / len(scene_keys):.1f}")

        # K-Fold on scenes
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)

        splits = []
        for train_scene_idx, val_scene_idx in kfold.split(scene_keys):
            # Get scene keys for this fold
            train_scenes = [scene_keys[i] for i in train_scene_idx]
            val_scenes = [scene_keys[i] for i in val_scene_idx]

            # Get sample indices
            train_indices = [idx for scene in train_scenes for idx in scene_to_indices[scene]]
            val_indices = [idx for scene in val_scenes for idx in scene_to_indices[scene]]

            splits.append((train_indices, val_indices))

        return splits

    def train_kfold(self, dataset, k=5, epochs=20, batch_size=8, lr=0.001,
                    patience=7, use_amp=False, num_workers=4, persistent_workers=True,
                    prefetch_factor=3):
        """
        Train model with K-fold cross-validation at SCENE level

        FIXED: Splits scenes (not individual patches) to prevent data leakage

        Args:
            dataset: SEN12MSCRDataset instance
            k: Number of folds
            epochs: Training epochs per fold
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
            use_amp: Use Automatic Mixed Precision
            num_workers: Data loading workers
            persistent_workers: Keep workers alive between epochs
            prefetch_factor: Batches to prefetch per worker
        """
        import time
        start_time = time.time()

        print(f"\n{'=' * 60}")
        print(f"Training {self.model_type} with {k}-Fold CV (Scene-Level Split)")
        print(f"{'=' * 60}")

        # Get scene-level splits
        splits = self._get_scene_level_splits(dataset, k=k)

        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\nFold {fold + 1}/{k}")
            print(f"  Training samples: {len(train_idx)}")
            print(f"  Validation samples: {len(val_idx)}")
            print("-" * 40)

            # Create subsets
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            # Create data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                drop_last=True
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                num_workers=num_workers // 2,
                pin_memory=True,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=prefetch_factor if num_workers > 0 else None
            )

            # Initialize model
            if self.model_type == 'UNet':
                model = UNet(in_channels=len(dataset.s2_bands),
                             out_channels=len(dataset.s2_bands)).to(self.device)
            elif self.model_type == 'SimpleCNN':
                model = SimpleCNN(in_channels=len(dataset.s2_bands),
                                  out_channels=len(dataset.s2_bands)).to(self.device)
            elif self.model_type == 'GAN':
                model = self._train_gan_fold(train_loader, val_loader, epochs, lr,
                                             patience, use_amp, len(dataset.s2_bands))
                self.models.append(model)
                continue
            elif self.model_type == 'LSTM':
                model = LSTMCloudRemover(in_channels=len(dataset.s2_bands)).to(self.device)
            elif self.model_type == 'Diffusion':
                model = self._train_diffusion_fold(train_loader, val_loader, epochs,
                                                   lr, patience, use_amp, len(dataset.s2_bands))
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
        print(f"\n{'=' * 60}")
        print(f"Total training time: {self.training_time / 60:.2f} minutes")
        print(f"Average time per fold: {self.training_time / k / 60:.2f} minutes")
        print(f"{'=' * 60}")

        return self.models, self.histories

    def _train_fold(self, model, train_loader, val_loader, epochs, lr,
                    patience, use_amp):
        """Train single fold with early stopping"""
        import time
        fold_start = time.time()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None

        history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}

        for epoch in range(epochs):
            epoch_start = time.time()

            # Training
            model.train()
            train_loss = 0
            for s1, s2_cloudy, s2_clean, cloud_mask in train_loader:
                s2_cloudy = s2_cloudy.to(self.device, non_blocking=True)
                s2_clean = s2_clean.to(self.device, non_blocking=True)

                # Handle LSTM input
                if self.model_type == 'LSTM':
                    s2_cloudy = s2_cloudy.unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)

                # Mixed precision forward pass
                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(s2_cloudy)
                        loss = criterion(output, s2_clean)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(s2_cloudy)
                    loss = criterion(output, s2_clean)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for s1, s2_cloudy, s2_clean, cloud_mask in val_loader:
                    s2_cloudy = s2_cloudy.to(self.device, non_blocking=True)
                    s2_clean = s2_clean.to(self.device, non_blocking=True)

                    if self.model_type == 'LSTM':
                        s2_cloudy = s2_cloudy.unsqueeze(1)

                    if use_amp and self.device == 'cuda':
                        with torch.cuda.amp.autocast():
                            output = model(s2_cloudy)
                            loss = criterion(output, s2_clean)
                    else:
                        output = model(s2_cloudy)
                        loss = criterion(output, s2_clean)

                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            epoch_time = time.time() - epoch_start

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['epoch_times'].append(epoch_time)

            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s")

            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        # Load best model
        early_stopping.load_best_model(model)
        fold_time = time.time() - fold_start
        print(f"Fold training time: {fold_time / 60:.2f} minutes")
        print(f"Loaded best model with val_loss: {early_stopping.best_loss:.6f}")

        return history

    def _train_gan_fold(self, train_loader, val_loader, epochs, lr, patience, use_amp, n_channels):
        """Train GAN for single fold with early stopping"""
        generator = Generator(in_channels=n_channels, out_channels=n_channels).to(self.device)
        discriminator = Discriminator(in_channels=n_channels * 2).to(self.device)

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

            for s1, s2_cloudy, s2_clean, cloud_mask in train_loader:
                s2_cloudy = s2_cloudy.to(self.device)
                s2_clean = s2_clean.to(self.device)
                batch_size = s2_cloudy.size(0)

                # Labels for discriminator
                real_label = torch.ones(batch_size, 1, 30, 30).to(self.device)
                fake_label = torch.zeros(batch_size, 1, 30, 30).to(self.device)

                # Train Discriminator
                optimizer_d.zero_grad()

                if use_amp and scaler_d is not None:
                    with torch.cuda.amp.autocast():
                        fake_clean = generator(s2_cloudy)
                        real_loss = criterion_gan(discriminator(s2_cloudy, s2_clean), real_label)
                        fake_loss = criterion_gan(discriminator(s2_cloudy, fake_clean.detach()), fake_label)
                        d_loss = (real_loss + fake_loss) / 2
                    scaler_d.scale(d_loss).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    fake_clean = generator(s2_cloudy)
                    real_loss = criterion_gan(discriminator(s2_cloudy, s2_clean), real_label)
                    fake_loss = criterion_gan(discriminator(s2_cloudy, fake_clean.detach()), fake_label)
                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()

                if use_amp and scaler_g is not None:
                    with torch.cuda.amp.autocast():
                        fake_clean = generator(s2_cloudy)
                        gan_loss = criterion_gan(discriminator(s2_cloudy, fake_clean), real_label)
                        l1_loss = criterion_l1(fake_clean, s2_clean)
                        g_loss = gan_loss + 100 * l1_loss
                    scaler_g.scale(g_loss).backward()
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    fake_clean = generator(s2_cloudy)
                    gan_loss = criterion_gan(discriminator(s2_cloudy, fake_clean), real_label)
                    l1_loss = criterion_l1(fake_clean, s2_clean)
                    g_loss = gan_loss + 100 * l1_loss
                    g_loss.backward()
                    optimizer_g.step()

            # Validation
            generator.eval()
            val_loss = 0
            with torch.no_grad():
                for s1, s2_cloudy, s2_clean, cloud_mask in val_loader:
                    s2_cloudy = s2_cloudy.to(self.device)
                    s2_clean = s2_clean.to(self.device)
                    fake_clean = generator(s2_cloudy)
                    val_loss += criterion_l1(fake_clean, s2_clean).item()

            val_loss /= len(val_loader)

            print(
                f"Epoch {epoch + 1}/{epochs} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if early_stopping(val_loss, generator):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        early_stopping.load_best_model(generator)
        return generator

    def _train_diffusion_fold(self, train_loader, val_loader, epochs, lr, patience, use_amp, n_channels):
        """Train Diffusion model for single fold"""
        model = DiffusionModel(in_channels=n_channels).to(self.device)
        diffusion_trainer = DiffusionTrainer(model, self.device, num_timesteps=1000)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for s1, s2_cloudy, s2_clean, cloud_mask in train_loader:
                s2_clean = s2_clean.to(self.device)

                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = diffusion_trainer.train_step(s2_clean, optimizer)
                else:
                    loss = diffusion_trainer.train_step(s2_clean, optimizer)

                train_loss += loss

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for s1, s2_cloudy, s2_clean, cloud_mask in val_loader:
                    s2_cloudy = s2_cloudy.to(self.device)
                    s2_clean = s2_clean.to(self.device)
                    denoised = diffusion_trainer.denoise(s2_cloudy, num_inference_steps=20)
                    val_loss += nn.functional.mse_loss(denoised, s2_clean).item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if early_stopping(val_loss, model):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        early_stopping.load_best_model(model)
        return model


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
        """Full denoising process (cloud removal)"""
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