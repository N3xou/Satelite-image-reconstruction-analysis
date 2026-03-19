"""
config.py — Central configuration for the satellite cloud removal pipeline.

All settings that were previously scattered across main.py are defined here.
Import with:
    from config import Config
"""

from pathlib import Path


class Config:
    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    DATASET_ROOT       = "./sen12mscr_dataset"
    SEASONS            = None              # None = all four seasons
    S2_BANDS           = list(range(1, 14))  # all 13 Sentinel-2 bands
    PATCH_SIZE         = 256
    DATA_FRACTION      = 0.15             # fraction of scenes to load
    MIN_CLOUD_FRACTION = 0.2              # discard patches with less cloud
    MAX_CLOUD_FRACTION = 0.7              # discard patches with more cloud

    # Cloud mask mode fed to SEN12MSCRDataset
    # Options: "combined" | "gt_diff" | "gt_threshold" | "spectral" | "sar_optical"
    CLOUD_MASK_MODE    = "combined"

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------
    # "basic" — original plain losses for each model:
    #     UNet/SimpleCNN : MSE(pred, clean)
    #     GAN generator  : adversarial + 100 × L1(fake, clean)
    #     DSen2-CR       : CARL  (unchanged — always cloud-aware by design)
    #     Diffusion      : uniform MSE on noise prediction
    #
    # "MRL"  — Masked Reconstruction Loss (cloud-aware) for each model:
    #     UNet/SimpleCNN : masked_reconstruction_loss
    #                      rewards fixing clouds, penalises touching clear pixels
    #     GAN generator  : adversarial + 100 × masked_reconstruction_loss
    #     DSen2-CR       : CARL  (unchanged — already optimal)
    #     Diffusion      : diffusion_noise_loss
    #                      upweights noise-prediction error on cloud pixels
    LOSS_TYPE          = "MRL"            # "basic" | "MRL"

    # MRL-specific weights (only used when LOSS_TYPE == "MRL")
    MRL_CLOUD_WEIGHT   = 3.0   # multiplier on cloud-pixel reconstruction error
    MRL_CLEAR_PRESERVE = 2.0   # multiplier on clear-pixel modification penalty
    MRL_DIFF_CLOUD_W   = 3.0   # cloud upweight for diffusion noise loss

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    BATCH_SIZE     = 6
    EPOCHS         = 5
    LEARNING_RATE  = 0.001
    PATIENCE       = 2
    USE_AMP        = True      # Automatic Mixed Precision (mandatory on Pascal)
    NUM_WORKERS    = 4

    # ------------------------------------------------------------------
    # Models to train
    # Choices: "SimpleCNN" | "UNet" | "GAN" | "DSen2CR" | "Diffusion" | "RandomForest"
    # ------------------------------------------------------------------
    MODELS = ["GAN", "UNet", "RandomForest"]
    # MODELS = ["SimpleCNN", "UNet", "RandomForest"]
    # MODELS = ["SimpleCNN", "UNet", "GAN", "DSen2CR", "Diffusion", "RandomForest"]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    OUTPUT_DIR   = Path("./MRLresults")
    SAVE_MODELS  = True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @classmethod
    def validate(cls):
        """Raise early with a clear message if settings are inconsistent."""
        valid_losses = {"basic", "MRL"}
        if cls.LOSS_TYPE not in valid_losses:
            raise ValueError(
                f"Config.LOSS_TYPE='{cls.LOSS_TYPE}' is not valid. "
                f"Choose from {valid_losses}."
            )

        valid_masks = {"combined", "gt_diff", "gt_threshold", "spectral", "sar_optical"}
        if cls.CLOUD_MASK_MODE not in valid_masks:
            raise ValueError(
                f"Config.CLOUD_MASK_MODE='{cls.CLOUD_MASK_MODE}' is not valid. "
                f"Choose from {valid_masks}."
            )

        valid_models = {"SimpleCNN", "UNet", "GAN", "DSen2CR", "Diffusion", "RandomForest"}
        bad = set(cls.MODELS) - valid_models
        if bad:
            raise ValueError(
                f"Unknown model(s) in Config.MODELS: {bad}. "
                f"Valid choices: {valid_models}."
            )

        if not (0.0 < cls.DATA_FRACTION <= 1.0):
            raise ValueError(
                f"Config.DATA_FRACTION={cls.DATA_FRACTION} must be in (0, 1]."
            )

        if not (0.0 <= cls.MIN_CLOUD_FRACTION < cls.MAX_CLOUD_FRACTION <= 1.0):
            raise ValueError(
                f"Cloud fraction bounds invalid: "
                f"MIN={cls.MIN_CLOUD_FRACTION}, MAX={cls.MAX_CLOUD_FRACTION}. "
                f"Must satisfy 0 <= MIN < MAX <= 1."
            )

    @classmethod
    def summary(cls):
        """Print a formatted summary of the active configuration."""
        print("\n" + "=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"  Dataset root:      {cls.DATASET_ROOT}")
        print(f"  S2 bands:          {len(cls.S2_BANDS)}  {cls.S2_BANDS}")
        print(f"  Data fraction:     {cls.DATA_FRACTION * 100:.0f}%")
        print(f"  Patch size:        {cls.PATCH_SIZE}×{cls.PATCH_SIZE}")
        print(f"  Cloud mask mode:   {cls.CLOUD_MASK_MODE}")
        print(f"  Cloud fraction:    {cls.MIN_CLOUD_FRACTION:.0%} – {cls.MAX_CLOUD_FRACTION:.0%}")
        print(f"")
        print(f"  Loss type:         {cls.LOSS_TYPE}")
        if cls.LOSS_TYPE == "MRL":
            print(f"    cloud_weight:    {cls.MRL_CLOUD_WEIGHT}")
            print(f"    clear_preserve:  {cls.MRL_CLEAR_PRESERVE}")
            print(f"    diff_cloud_w:    {cls.MRL_DIFF_CLOUD_W}")
        print(f"")
        print(f"  Models:            {', '.join(cls.MODELS)}")
        print(f"  Batch size:        {cls.BATCH_SIZE}")
        print(f"  Epochs:            {cls.EPOCHS}")
        print(f"  Learning rate:     {cls.LEARNING_RATE}")
        print(f"  Patience:          {cls.PATIENCE}")
        print(f"  AMP:               {cls.USE_AMP}")
        print(f"  Workers:           {cls.NUM_WORKERS}")
        print(f"  Output dir:        {cls.OUTPUT_DIR}")
        print(f"  Save models:       {cls.SAVE_MODELS}")
        print("=" * 60)