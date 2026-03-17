import time
import torch
import torch.nn as nn
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
# Cloud-aware loss functions
# ---------------------------------------------------------------------------

def masked_reconstruction_loss(
    predicted:      torch.Tensor,
    s2_clean:       torch.Tensor,
    s2_cloudy:      torch.Tensor,
    cloud_mask:     torch.Tensor,
    cloud_weight:   float = 3.0,
    clear_preserve: float = 2.0,
) -> torch.Tensor:
    """
    Masked Reconstruction Loss — used by UNet, SimpleCNN, GAN.

    Two competing objectives expressed as weighted MSE:

    1. Cloud reconstruction (reward):
       cloud_mask * (pred - clean)^2  weighted by cloud_weight
       Pushes the model to correct cloudy pixels toward the ground truth.

    2. Clear-area preservation (punishment):
       (1-cloud_mask) * (pred - cloudy)^2  weighted by clear_preserve
       In clear areas s2_cloudy == s2_clean, so this directly punishes
       any unnecessary modification of already-correct pixels.

    Default weights (cloud_weight=3, clear_preserve=2):
       Cloud pixels count 3x, clear pixels 2x a neutral pixel.
       The model is moderately more rewarded for fixing clouds than
       punished for touching clear areas (ratio 3:2).
       Raise clear_preserve to make the model more conservative.
    """
    N          = predicted.numel()
    clear_mask = 1.0 - cloud_mask

    cloud_recon = cloud_weight  * cloud_mask  * (predicted - s2_clean ).pow(2)
    clear_pres  = clear_preserve * clear_mask * (predicted - s2_cloudy).pow(2)

    return (cloud_recon + clear_pres).sum() / N


def carl_loss(
    predicted:   torch.Tensor,
    s2_clean:    torch.Tensor,
    s2_cloudy:   torch.Tensor,
    cloud_mask:  torch.Tensor,
    lambda_reg:  float = 1.0,
) -> torch.Tensor:
    """
    Cloud-Adaptive Regularized Loss (CARL) — used by DSen2-CR.
    Reference: Meraner et al. (2020), Section 4.3.

    CARL = cloud_adaptive_L1 + lambda_reg * target_L1

        cloud_adaptive_L1:
            cloud pixels  → |pred - clean|   (fix the cloud)
            clear pixels  → |pred - cloudy|  (don't touch what's correct)

        target_L1:
            full image    → |pred - clean|   (global regularisation)
    """
    N_tot      = predicted.numel()
    clear_mask = 1.0 - cloud_mask

    cloud_adaptive = (
        cloud_mask * torch.abs(predicted - s2_clean) +
        clear_mask * torch.abs(predicted - s2_cloudy)
    ).sum() / N_tot

    target_reg = torch.abs(predicted - s2_clean).sum() / N_tot

    return cloud_adaptive + lambda_reg * target_reg


def diffusion_noise_loss(
    predicted_noise: torch.Tensor,
    true_noise:      torch.Tensor,
    cloud_mask:      torch.Tensor,
    cloud_weight:    float = 3.0,
) -> torch.Tensor:
    """
    Cloud-weighted noise prediction loss — used by Diffusion Model.

    Standard DDPM uses MSE(predicted_noise, true_noise) uniformly.
    Here cloud pixels are upweighted: getting the noise right in cloud
    regions matters more because those pixels need to be reconstructed
    from scratch during the reverse diffusion process.

    Clear pixels still contribute at weight 1.0 so the model does not
    corrupt them, but cloud pixels contribute at cloud_weight.
    """
    N          = predicted_noise.numel()
    clear_mask = 1.0 - cloud_mask
    error_sq   = (predicted_noise - true_noise).pow(2)

    weighted = cloud_weight * cloud_mask * error_sq + clear_mask * error_sq
    return weighted.sum() / N


class ModelTrainer:
    """
    Trains a single model on a pre-built train/val split.

    The caller (main.py) is responsible for creating the split once and
    passing ready-made DataLoaders.  There is no K-fold logic here.
    """

    VALID_LOSS_TYPES = {"basic", "MRL"}

    def __init__(self, model_type,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 loss_type: str = "MRL"):
        """
        Parameters
        ----------
        model_type : str    Name of the model to train.
        device     : str    "cuda" or "cpu".
        loss_type  : str    "basic" — plain MSE / L1 losses (original behaviour).
                            "MRL"   — Masked Reconstruction Loss family:
                                      cloud pixels rewarded, clear pixels preserved.
        """
        if loss_type not in self.VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type='{loss_type}' is invalid. "
                f"Choose from {self.VALID_LOSS_TYPES}."
            )
        self.model_type    = model_type
        self.loss_type     = loss_type
        self.device        = device
        self.training_time = 0

        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            torch.backends.cudnn.benchmark       = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True

    # ------------------------------------------------------------------
    # Public entry point — static train/val split
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader,
              epochs=20, lr=0.001, patience=7, use_amp=False):
        """
        Train the model on a fixed train/val split.

        The split is created once in main.py and shared across all models,
        so every model sees exactly the same training and validation patches.

        Parameters
        ----------
        train_loader : DataLoader   Pre-built training loader.
        val_loader   : DataLoader   Pre-built validation loader.
        epochs       : int          Maximum training epochs.
        lr           : float        Initial learning rate.
        patience     : int          Early-stopping patience.
        use_amp      : bool         Enable AMP (fp16) training.

        Returns
        -------
        model    : trained PyTorch model (or RandomForestWrapper)
        history  : dict with train_loss / val_loss / epoch_times lists
        """
        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Training {self.model_type}  [loss={self.loss_type}]")
        print(f"  train batches: {len(train_loader)}  |  val batches: {len(val_loader)}")
        print(f"{'='*60}")

        # Resolve channel counts from the loader's dataset
        base_ds      = train_loader.dataset
        while isinstance(base_ds, torch.utils.data.Subset):
            base_ds  = base_ds.dataset
        in_channels  = 2 + len(base_ds.s2_bands)   # S1(2) + S2(n)
        out_channels = len(base_ds.s2_bands)

        # ---- dispatch to model-specific trainer ----
        if self.model_type in ("UNet", "SimpleCNN"):
            if self.model_type == "UNet":
                model = UNet(in_channels=in_channels, out_channels=out_channels).to(self.device)
            else:
                model = SimpleCNN(in_channels=in_channels, out_channels=out_channels).to(self.device)
            history = self._train_fold(model, train_loader, val_loader,
                                       epochs, lr, patience, use_amp, self.loss_type)

        elif self.model_type == "GAN":
            model   = self._train_gan_fold(train_loader, val_loader, epochs, lr,
                                           patience, use_amp, in_channels, out_channels,
                                           self.loss_type)
            history = {}

        elif self.model_type == "DSen2CR":
            model   = self._train_dsen2cr_fold(train_loader, val_loader, epochs, lr,
                                               patience, use_amp, in_channels, out_channels)
            history = {}

        elif self.model_type == "Diffusion":
            model   = self._train_diffusion_fold(train_loader, val_loader, epochs,
                                                 lr, patience, use_amp,
                                                 in_channels, out_channels, self.loss_type)
            history = {}

        elif self.model_type == "RandomForest":
            model, history = self._train_rf_fold(train_loader, val_loader)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.training_time = time.time() - start_time
        print(f"\n✓ Training complete: {self.training_time / 60:.2f} min")
        return model, history


    # ------------------------------------------------------------------
    # Generic fold trainer  (UNet / SimpleCNN)
    # ------------------------------------------------------------------

    def _train_fold(self, model, train_loader, val_loader, epochs, lr, patience,
                    use_amp, loss_type: str = "MRL"):
        """
        Generic fold trainer for UNet and SimpleCNN.

        loss_type="basic" : plain nn.MSELoss on all pixels (original behaviour)
        loss_type="MRL"   : masked_reconstruction_loss — rewards fixing clouds,
                            penalises modifying already-correct clear pixels
        """
        fold_start = time.time()
        optimizer  = optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None
        history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}
        mse     = nn.MSELoss()
        use_mrl = (loss_type == "MRL")
        print(f"  Loss: {'masked_reconstruction_loss (MRL)' if use_mrl else 'MSELoss (basic)'}")

        for epoch in range(epochs):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            for s1, s2_cloudy, s2_clean, cloud_mask in train_loader:
                s1         = s1.to(self.device,        non_blocking=True)
                s2_cloudy  = s2_cloudy.to(self.device,  non_blocking=True)
                s2_clean   = s2_clean.to(self.device,   non_blocking=True)
                cloud_mask = cloud_mask.to(self.device, non_blocking=True)
                model_input = torch.cat([s1, s2_cloudy], dim=1)
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler:
                    with torch.cuda.amp.autocast():
                        pred = model(model_input)
                        loss = (masked_reconstruction_loss(pred, s2_clean, s2_cloudy, cloud_mask)
                                if use_mrl else mse(pred, s2_clean))
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    pred = model(model_input)
                    loss = (masked_reconstruction_loss(pred, s2_clean, s2_cloudy, cloud_mask)
                            if use_mrl else mse(pred, s2_clean))
                    loss.backward(); optimizer.step()
                train_loss += loss.item()

            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for s1, s2_cloudy, s2_clean, cloud_mask in val_loader:
                    s1         = s1.to(self.device,        non_blocking=True)
                    s2_cloudy  = s2_cloudy.to(self.device,  non_blocking=True)
                    s2_clean   = s2_clean.to(self.device,   non_blocking=True)
                    cloud_mask = cloud_mask.to(self.device, non_blocking=True)
                    mi   = torch.cat([s1, s2_cloudy], dim=1)
                    pred = model(mi)
                    if use_amp and self.device == 'cuda':
                        with torch.cuda.amp.autocast():
                            v = (masked_reconstruction_loss(pred, s2_clean, s2_cloudy, cloud_mask)
                                 if use_mrl else mse(pred, s2_clean))
                    else:
                        v = (masked_reconstruction_loss(pred, s2_clean, s2_cloudy, cloud_mask)
                             if use_mrl else mse(pred, s2_clean))
                    val_loss += v.item()

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
                         patience, use_amp, n_channels, out_channels,
                         loss_type: str = "MRL"):
        """
        loss_type="basic" : generator pixel loss = 100 × L1(fake, clean)
        loss_type="MRL"   : generator pixel loss = 100 × masked_reconstruction_loss
        """
        use_mrl = (loss_type == "MRL")
        print(f"  Loss: {'masked_reconstruction_loss (MRL)' if use_mrl else 'L1 (basic)'}")
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
            for s1, s2_cloudy, s2_clean, cloud_mask in train_loader:
                s1         = s1.to(self.device)
                s2_cloudy  = s2_cloudy.to(self.device)
                s2_clean   = s2_clean.to(self.device)
                cloud_mask = cloud_mask.to(self.device)
                bs         = s2_cloudy.size(0)
                mi         = torch.cat([s1, s2_cloudy], dim=1)
                real_label = torch.ones(bs, 1, 30, 30).to(self.device)
                fake_label = torch.zeros(bs, 1, 30, 30).to(self.device)

                # ---- Discriminator step (unchanged — standard real/fake) ----
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

                # ---- Generator step (masked pixel loss replaces plain L1) ----
                # The adversarial term keeps outputs perceptually realistic.
                # masked_reconstruction_loss replaces plain L1 so the generator
                # is rewarded for fixing cloud pixels and penalised for touching
                # pixels that were already correct in clear areas.
                optimizer_g.zero_grad()
                if use_amp and scaler_g:
                    with torch.cuda.amp.autocast():
                        fake = generator(mi)
                        adv  = criterion_gan(discriminator(s2_cloudy, fake), real_label)
                        pix  = (masked_reconstruction_loss(fake, s2_clean, s2_cloudy, cloud_mask)
                                if use_mrl else criterion_l1(fake, s2_clean))
                        g_loss = adv + 100 * pix
                    scaler_g.scale(g_loss).backward(); scaler_g.step(optimizer_g); scaler_g.update()
                else:
                    fake = generator(mi)
                    adv  = criterion_gan(discriminator(s2_cloudy, fake), real_label)
                    pix  = (masked_reconstruction_loss(fake, s2_clean, s2_cloudy, cloud_mask)
                            if use_mrl else criterion_l1(fake, s2_clean))
                    g_loss = adv + 100 * pix
                    g_loss.backward(); optimizer_g.step()

            generator.eval(); val_loss = 0.0
            with torch.no_grad():
                for s1, s2_cloudy, s2_clean, cloud_mask in val_loader:
                    s1         = s1.to(self.device);  s2_cloudy = s2_cloudy.to(self.device)
                    s2_clean   = s2_clean.to(self.device);  cloud_mask = cloud_mask.to(self.device)
                    fake = generator(torch.cat([s1, s2_cloudy], dim=1))
                    v    = (masked_reconstruction_loss(fake, s2_clean, s2_cloudy, cloud_mask)
                            if use_mrl else criterion_l1(fake, s2_clean))
                    val_loss += v.item()
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
                               patience, use_amp, n_channels, out_channels,
                               loss_type: str = "MRL"):
        """
        loss_type="basic" : uniform MSE on noise prediction — nn.functional.mse_loss
        loss_type="MRL"   : diffusion_noise_loss — upweights noise error on cloud pixels
        """
        use_mrl    = (loss_type == "MRL")
        print(f"  Loss: {'diffusion_noise_loss (MRL)' if use_mrl else 'uniform MSE (basic)'}")
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
                        pred_n = model(x_input, t)
                        loss   = (diffusion_noise_loss(pred_n, noise, cloud_mask)
                                  if use_mrl else nn.functional.mse_loss(pred_n, noise))
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    pred_n = model(x_input, t)
                    loss   = (diffusion_noise_loss(pred_n, noise, cloud_mask)
                              if use_mrl else nn.functional.mse_loss(pred_n, noise))
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