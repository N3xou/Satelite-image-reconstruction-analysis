import time
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader
from collections import defaultdict
from Models import (
    UNet, DiffusionModel, SimpleCNN,
    Generator, DSen2CR,
    Discriminator, RandomForestCloudRemover, RandomForestWrapper
)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=3, min_delta=0.0, verbose=True):
        self.patience          = patience
        self.min_delta         = min_delta
        self.verbose           = verbose
        self.counter           = 0
        self.best_loss         = None
        self.early_stop        = False
        self.best_model_state  = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss        = val_loss
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
            self.best_loss        = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter          = 0
        return self.early_stop

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ---------------------------------------------------------------------------
# CARL loss  (Cloud-Adaptive Regularized Loss)
# Reference: Meraner et al. (2020), Eq. in Section 4.3
# ---------------------------------------------------------------------------

def carl_loss(
    predicted:   torch.Tensor,   # [B, C, H, W]  model output
    s2_clean:    torch.Tensor,   # [B, C, H, W]  ground-truth
    s2_cloudy:   torch.Tensor,   # [B, C, H, W]  cloudy input
    cloud_mask:  torch.Tensor,   # [B, 1, H, W]  in [0,1]
    lambda_reg:  float = 1.0,
) -> torch.Tensor:
    """
    CARL = cloud_adaptive_L1 + lambda_reg * target_L1

    cloud_adaptive_L1:
        cloud pixels   → |pred - clean|   (penalise reconstruction error)
        clear pixels   → |pred - cloudy|  (penalise unnecessary modification)

    target_L1:
        full image     → |pred - clean|   (global regularisation)
    """
    N_tot = predicted.numel()
    clear_mask = 1.0 - cloud_mask                         # [B,1,H,W]

    cloud_adaptive = (
        cloud_mask  * torch.abs(predicted - s2_clean) +
        clear_mask  * torch.abs(predicted - s2_cloudy)
    ).sum() / N_tot

    target_reg = torch.abs(predicted - s2_clean).sum() / N_tot

    return cloud_adaptive + lambda_reg * target_reg


class ModelTrainer:
    """Handles training with K-Fold cross-validation (scene-level splitting)."""

    def __init__(self, model_type, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_type   = model_type
        self.device       = device
        self.models       = []
        self.histories    = []
        self.training_time = 0

        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            torch.backends.cudnn.benchmark    = True
            torch.backends.cuda.matmul.allow_tf32  = True
            torch.backends.cudnn.allow_tf32   = True

    # ------------------------------------------------------------------
    # Scene-level K-fold split
    # ------------------------------------------------------------------

    def _get_scene_level_splits(self, dataset, k=5):
        actual_dataset = dataset
        is_subset      = isinstance(dataset, torch.utils.data.Subset)
        if is_subset:
            actual_dataset = dataset.dataset
            subset_indices = dataset.indices

        scene_to_indices = defaultdict(list)
        indices_to_iterate = range(len(dataset)) if not is_subset else range(len(subset_indices))

        for idx in indices_to_iterate:
            sample_idx = idx if not is_subset else subset_indices[idx]
            sample     = actual_dataset.samples[sample_idx]
            parts      = sample['s2_clean'].parts
            season     = parts[-3].split('_s2')[0]
            scene_id   = parts[-2].split('_')[1]
            scene_to_indices[f"{season}_{scene_id}"].append(idx)

        scene_keys = list(scene_to_indices.keys())
        print(f"\nDataset scene-level statistics:")
        print(f"  Total scenes:           {len(scene_keys)}")
        print(f"  Total samples:          {len(actual_dataset.samples)}")
        print(f"  Avg samples per scene:  {len(actual_dataset.samples)/len(scene_keys):.1f}")

        kfold  = KFold(n_splits=k, shuffle=True, random_state=42)
        splits = []
        for train_si, val_si in kfold.split(scene_keys):
            train_scenes = [scene_keys[i] for i in train_si]
            val_scenes   = [scene_keys[i] for i in val_si]
            splits.append((
                [idx for s in train_scenes for idx in scene_to_indices[s]],
                [idx for s in val_scenes   for idx in scene_to_indices[s]],
            ))
        return splits

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train_kfold(self, dataset, k=5, epochs=20, batch_size=8, lr=0.001,
                    patience=7, use_amp=False, num_workers=4,
                    persistent_workers=True, prefetch_factor=3):

        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Training {self.model_type} with {k}-Fold CV (Scene-Level Split)")
        print(f"{'='*60}")

        splits = self._get_scene_level_splits(dataset, k=k)

        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\nFold {fold+1}/{k}  |  train={len(train_idx)}  val={len(val_idx)}")
            print("-" * 40)

            train_loader = DataLoader(
                torch.utils.data.Subset(dataset, train_idx),
                batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                drop_last=True,
            )
            val_loader = DataLoader(
                torch.utils.data.Subset(dataset, val_idx),
                batch_size=batch_size,
                num_workers=num_workers // 2, pin_memory=True,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )

            base_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
            in_channels  = 2 + len(base_dataset.s2_bands)   # S1(2) + S2(n)
            out_channels = len(base_dataset.s2_bands)

            # ---- model construction ----
            if self.model_type == 'UNet':
                model   = UNet(in_channels=in_channels, out_channels=out_channels).to(self.device)
                history = self._train_fold(model, train_loader, val_loader, epochs, lr, patience, use_amp)

            elif self.model_type == 'SimpleCNN':
                model   = SimpleCNN(in_channels=in_channels, out_channels=out_channels).to(self.device)
                history = self._train_fold(model, train_loader, val_loader, epochs, lr, patience, use_amp)

            elif self.model_type == 'GAN':
                model   = self._train_gan_fold(train_loader, val_loader, epochs, lr,
                                               patience, use_amp, in_channels, out_channels)
                self.models.append(model); continue

            elif self.model_type == 'DSen2CR':
                model   = self._train_dsen2cr_fold(train_loader, val_loader, epochs, lr,
                                                   patience, use_amp, in_channels, out_channels)
                self.models.append(model); continue

            elif self.model_type == 'Diffusion':
                model   = self._train_diffusion_fold(train_loader, val_loader, epochs,
                                                     lr, patience, use_amp, in_channels, out_channels)
                self.models.append(model); continue

            elif self.model_type == 'RandomForest':
                model, history = self._train_rf_fold(train_loader, val_loader)
                self.models.append(model); self.histories.append(history); continue

            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            self.models.append(model)
            self.histories.append(history)

        self.training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Total training time:      {self.training_time/60:.2f} min")
        print(f"Average time per fold:    {self.training_time/k/60:.2f} min")
        print(f"{'='*60}")
        return self.models, self.histories

    # ------------------------------------------------------------------
    # Generic fold trainer  (UNet / SimpleCNN)
    # ------------------------------------------------------------------

    def _train_fold(self, model, train_loader, val_loader, epochs, lr, patience, use_amp):
        fold_start = time.time()
        criterion  = nn.MSELoss()
        optimizer  = optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None
        history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}

        for epoch in range(epochs):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            for s1, s2_cloudy, s2_clean, _ in train_loader:
                s1        = s1.to(self.device,       non_blocking=True)
                s2_cloudy = s2_cloudy.to(self.device, non_blocking=True)
                s2_clean  = s2_clean.to(self.device,  non_blocking=True)
                model_input = torch.cat([s1, s2_cloudy], dim=1)
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler:
                    with torch.cuda.amp.autocast():
                        loss = criterion(model(model_input), s2_clean)
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss = criterion(model(model_input), s2_clean)
                    loss.backward(); optimizer.step()
                train_loss += loss.item()

            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for s1, s2_cloudy, s2_clean, _ in val_loader:
                    s1        = s1.to(self.device,        non_blocking=True)
                    s2_cloudy = s2_cloudy.to(self.device,  non_blocking=True)
                    s2_clean  = s2_clean.to(self.device,   non_blocking=True)
                    mi = torch.cat([s1, s2_cloudy], dim=1)
                    if use_amp and self.device == 'cuda':
                        with torch.cuda.amp.autocast():
                            val_loss += criterion(model(mi), s2_clean).item()
                    else:
                        val_loss += criterion(model(mi), s2_clean).item()

            train_loss /= len(train_loader); val_loss /= len(val_loader)
            epoch_time  = time.time() - t0
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['epoch_times'].append(epoch_time)
            print(f"Epoch {epoch+1}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}  {epoch_time:.1f}s")
            if early_stopping(val_loss, model):
                print(f"Early stopping at epoch {epoch+1}"); break

        early_stopping.load_best_model(model)
        print(f"Fold done in {(time.time()-fold_start)/60:.2f} min  |  best_val={early_stopping.best_loss:.6f}")
        return history

    # ------------------------------------------------------------------
    # DSen2-CR fold trainer  (uses CARL loss)
    # ------------------------------------------------------------------

    def _train_dsen2cr_fold(self, train_loader, val_loader, epochs, lr,
                             patience, use_amp, in_channels, out_channels):
        """
        Train DSen2-CR with the CARL (Cloud-Adaptive Regularized Loss).

        CARL differentiates cloudy pixels (penalise reconstruction error vs clean)
        from clear pixels (penalise unnecessary modification of already-correct values).
        Reference: Meraner et al. (2020), Section 4.3.
        """
        model = DSen2CR(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=16,        # B=16 as in the paper
            feature_size=256,     # F=256 as in the paper
        ).to(self.device)

        # Adam with paper lr; gradient clipping handled manually (Section 6.6)
        optimizer      = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None
        history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}

        for epoch in range(epochs):
            t0 = time.time()
            model.train(); train_loss = 0.0

            for s1, s2_cloudy, s2_clean, cloud_mask in train_loader:
                s1         = s1.to(self.device,          non_blocking=True)
                s2_cloudy  = s2_cloudy.to(self.device,   non_blocking=True)
                s2_clean   = s2_clean.to(self.device,    non_blocking=True)
                cloud_mask = cloud_mask.to(self.device,  non_blocking=True)

                model_input = torch.cat([s1, s2_cloudy], dim=1)
                optimizer.zero_grad(set_to_none=True)

                if use_amp and scaler:
                    with torch.cuda.amp.autocast():
                        predicted = model(model_input)
                        loss = carl_loss(predicted, s2_clean, s2_cloudy, cloud_mask)
                    scaler.scale(loss).backward()
                    # Gradient clipping (Section 6.6 of paper: clip norm to 5)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    predicted = model(model_input)
                    loss = carl_loss(predicted, s2_clean, s2_cloudy, cloud_mask)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                train_loss += loss.item()

            # Validation with CARL loss
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for s1, s2_cloudy, s2_clean, cloud_mask in val_loader:
                    s1         = s1.to(self.device,         non_blocking=True)
                    s2_cloudy  = s2_cloudy.to(self.device,  non_blocking=True)
                    s2_clean   = s2_clean.to(self.device,   non_blocking=True)
                    cloud_mask = cloud_mask.to(self.device, non_blocking=True)
                    mi = torch.cat([s1, s2_cloudy], dim=1)
                    if use_amp and self.device == 'cuda':
                        with torch.cuda.amp.autocast():
                            val_loss += carl_loss(model(mi), s2_clean, s2_cloudy, cloud_mask).item()
                    else:
                        val_loss += carl_loss(model(mi), s2_clean, s2_cloudy, cloud_mask).item()

            train_loss /= len(train_loader); val_loss /= len(val_loader)
            epoch_time  = time.time() - t0
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['epoch_times'].append(epoch_time)
            print(f"Epoch {epoch+1}/{epochs}  CARL_train={train_loss:.6f}  CARL_val={val_loss:.6f}  {epoch_time:.1f}s")
            if early_stopping(val_loss, model):
                print(f"Early stopping at epoch {epoch+1}"); break

        early_stopping.load_best_model(model)
        print(f"DSen2-CR fold done  |  best_val={early_stopping.best_loss:.6f}")
        return model

    # ------------------------------------------------------------------
    # GAN fold trainer
    # ------------------------------------------------------------------

    def _train_gan_fold(self, train_loader, val_loader, epochs, lr,
                         patience, use_amp, n_channels, out_channels):
        generator     = Generator(in_channels=n_channels, out_channels=out_channels).to(self.device)
        discriminator = Discriminator(in_channels=out_channels * 2).to(self.device)
        criterion_gan = nn.BCEWithLogitsLoss()
        criterion_l1  = nn.L1Loss()
        optimizer_g   = optim.Adam(generator.parameters(),     lr=lr, betas=(0.5, 0.999))
        optimizer_d   = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        scaler_g = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None
        scaler_d = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None

        for epoch in range(epochs):
            generator.train(); discriminator.train()
            for s1, s2_cloudy, s2_clean, _ in train_loader:
                s1         = s1.to(self.device)
                s2_cloudy  = s2_cloudy.to(self.device)
                s2_clean   = s2_clean.to(self.device)
                bs         = s2_cloudy.size(0)
                mi         = torch.cat([s1, s2_cloudy], dim=1)
                real_label = torch.ones(bs, 1, 30, 30).to(self.device)
                fake_label = torch.zeros(bs, 1, 30, 30).to(self.device)

                optimizer_d.zero_grad()
                if use_amp and scaler_d:
                    with torch.cuda.amp.autocast():
                        fake    = generator(mi)
                        d_loss  = (criterion_gan(discriminator(s2_cloudy, s2_clean), real_label) +
                                   criterion_gan(discriminator(s2_cloudy, fake.detach()), fake_label)) / 2
                    scaler_d.scale(d_loss).backward(); scaler_d.step(optimizer_d); scaler_d.update()
                else:
                    fake   = generator(mi)
                    d_loss = (criterion_gan(discriminator(s2_cloudy, s2_clean), real_label) +
                              criterion_gan(discriminator(s2_cloudy, fake.detach()), fake_label)) / 2
                    d_loss.backward(); optimizer_d.step()

                optimizer_g.zero_grad()
                if use_amp and scaler_g:
                    with torch.cuda.amp.autocast():
                        fake   = generator(mi)
                        g_loss = criterion_gan(discriminator(s2_cloudy, fake), real_label) + 100 * criterion_l1(fake, s2_clean)
                    scaler_g.scale(g_loss).backward(); scaler_g.step(optimizer_g); scaler_g.update()
                else:
                    fake   = generator(mi)
                    g_loss = criterion_gan(discriminator(s2_cloudy, fake), real_label) + 100 * criterion_l1(fake, s2_clean)
                    g_loss.backward(); optimizer_g.step()

            generator.eval(); val_loss = 0.0
            with torch.no_grad():
                for s1, s2_cloudy, s2_clean, _ in val_loader:
                    s1 = s1.to(self.device); s2_cloudy = s2_cloudy.to(self.device); s2_clean = s2_clean.to(self.device)
                    val_loss += criterion_l1(generator(torch.cat([s1, s2_cloudy], dim=1)), s2_clean).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}  D={d_loss.item():.4f}  G={g_loss.item():.4f}  val={val_loss:.6f}")
            if early_stopping(val_loss, generator):
                print(f"Early stopping at epoch {epoch+1}"); break

        early_stopping.load_best_model(generator)
        return generator

    # ------------------------------------------------------------------
    # Diffusion fold trainer
    # ------------------------------------------------------------------

    def _train_diffusion_fold(self, train_loader, val_loader, epochs, lr,
                               patience, use_amp, n_channels, out_channels):
        n_channels += 1  # +1 for cloud mask conditioning channel
        model = DiffusionModel(in_channels=n_channels, out_channels=out_channels).to(self.device)
        optimizer      = optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None

        T             = 1000
        betas         = torch.linspace(1e-4, 0.02, T).to(self.device)
        alphas        = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        for epoch in range(epochs):
            model.train(); train_loss = 0.0
            for s1, s2_cloudy, s2_clean, cloud_mask in train_loader:
                s1         = s1.to(self.device);         s2_cloudy  = s2_cloudy.to(self.device)
                s2_clean   = s2_clean.to(self.device);   cloud_mask = cloud_mask.to(self.device)
                bs         = s2_clean.shape[0]
                x_cond     = torch.cat([s1, cloud_mask], dim=1)
                t          = torch.randint(0, T, (bs,), device=self.device)
                noise      = torch.randn_like(s2_clean)
                x_noisy    = (alphas_cumprod[t][:,None,None,None]**0.5 * s2_clean +
                               (1-alphas_cumprod[t][:,None,None,None])**0.5 * noise)
                x_input    = torch.cat([x_noisy, x_cond], dim=1)
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler:
                    with torch.cuda.amp.autocast():
                        loss = nn.functional.mse_loss(model(x_input, t), noise)
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss = nn.functional.mse_loss(model(x_input, t), noise)
                    loss.backward(); optimizer.step()
                train_loss += loss.item()

            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for s1, s2_cloudy, s2_clean, cloud_mask in val_loader:
                    s1 = s1.to(self.device); s2_clean = s2_clean.to(self.device); cloud_mask = cloud_mask.to(self.device)
                    x_cond  = torch.cat([s1, cloud_mask], dim=1)
                    x_noisy = torch.randn_like(s2_clean)
                    x_input = torch.cat([x_noisy, x_cond], dim=1)
                    pred_n  = model(x_input, torch.zeros(bs, dtype=torch.long, device=self.device))
                    val_loss += nn.functional.mse_loss(x_noisy - pred_n, s2_clean).item()
            train_loss /= len(train_loader); val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")
            if early_stopping(val_loss, model): break

        early_stopping.load_best_model(model)
        return model

    # ------------------------------------------------------------------
    # Random Forest fold trainer
    # ------------------------------------------------------------------

    def _train_rf_fold(self, train_loader, val_loader):
        print("\nTraining Random Forest model...")
        start = time.time()
        rf = RandomForestCloudRemover(n_estimators=200, max_depth=15)
        rf.fit(train_loader, device=self.device, max_samples=100000)

        print("\nValidating Random Forest...")
        val_loss, n_batches = 0.0, 0
        for s1, s2_cloudy, s2_clean, cloud_mask in val_loader:
            s1 = s1.to(self.device); s2_cloudy = s2_cloudy.to(self.device)
            s2_clean = s2_clean.to(self.device); cloud_mask = cloud_mask.to(self.device)
            pred     = rf.predict(s1, s2_cloudy, cloud_mask, device=self.device).to(self.device)
            val_loss += nn.functional.mse_loss(pred, s2_clean).item()
            n_batches += 1
            if n_batches >= 10: break
        val_loss /= n_batches
        print(f"Random Forest Validation Loss: {val_loss:.6f}")
        return RandomForestWrapper(rf), {
            'train_loss': [], 'val_loss': [val_loss],
            'epoch_times': [time.time() - start]
        }