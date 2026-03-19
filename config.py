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
    DATA_FRACTION      = 0.1             # fraction of scenes to load
    MIN_CLOUD_FRACTION = 0.1              # discard patches with less cloud
    MAX_CLOUD_FRACTION = 0.5              # discard patches with more cloud

    # Cloud mask mode fed to SEN12MSCRDataset.
    #
    # "gt_diff"          — supervised soft mask from |cloudy - clean|.
    #                      Most accurate signal for training.
    # "gt_threshold"     — hard binary version of gt_diff.
    # "feature_detector" — DSen2-CR reference algorithm (progressive
    #                      minimum-score elimination). Does not need the
    #                      clean image; best choice for inference.
    # "spectral"         — simplified brightness + haze + cirrus blend.
    #                      Kept for ablation; prefer feature_detector.
    # "sar_optical"      — legacy SAR vs optical disagreement.
    # "combined"         — gt_diff + feature_detector weighted blend.
    #                      Best of both worlds for training.
    CLOUD_MASK_MODE    = "gt_diff"

    # ------------------------------------------------------------------
    # feature_detector / combined mask parameters
    # Only used when CLOUD_MASK_MODE is "feature_detector" or "combined".
    # ------------------------------------------------------------------

    # Binarisation threshold for feature_detector (0–1).
    # Pixels with score >= this are considered cloud when counting coverage.
    # Does not affect the soft mask used as model input.
    FD_CLOUD_THRESHOLD  = 0.35

    # Enable NDMI moisture check (B08 + B11).
    # Slightly more accurate but requires both bands to be loaded.
    FD_USE_MOIST_CHECK  = False

    # If True, shadow pixels detected by the CSI shadow detector are given
    # the same mask weight as cloud pixels.  Set True if you want the model
    # to reconstruct shadowed areas as well as cloudy ones.
    FD_SHADOW_AS_CLOUD  = False

    # Weight of gt_diff component in "combined" mode (0–1).
    # The physics component (feature_detector) receives weight 1 - this value.
    GT_DIFF_WEIGHT      = 0.7

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
    LOSS_TYPE          = "basic"            # "basic" | "MRL"

    # MRL-specific weights (only used when LOSS_TYPE == "MRL")
    MRL_CLOUD_WEIGHT   = 3.0   # multiplier on cloud-pixel reconstruction error
    MRL_CLEAR_PRESERVE = 5.0   # multiplier on clear-pixel modification penalty
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
    MODELS = ["UNet"]
    # MODELS = ["GAN", "UNet", "RandomForest"]
    # MODELS = ["SimpleCNN", "UNet", "GAN", "DSen2CR", "Diffusion", "RandomForest"]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    OUTPUT_DIR   = Path("./MRLresults2")
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

        valid_masks = {
            "combined", "gt_diff", "gt_threshold",
            "spectral", "sar_optical", "feature_detector",
        }
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

        if not (0.0 < cls.FD_CLOUD_THRESHOLD < 1.0):
            raise ValueError(
                f"Config.FD_CLOUD_THRESHOLD={cls.FD_CLOUD_THRESHOLD} must be in (0, 1)."
            )

        if not (0.0 <= cls.GT_DIFF_WEIGHT <= 1.0):
            raise ValueError(
                f"Config.GT_DIFF_WEIGHT={cls.GT_DIFF_WEIGHT} must be in [0, 1]."
            )

        # Warn if feature_detector is used with a reduced band set that lacks
        # the aerosol or cirrus bands (best-effort, not a hard error).
        if cls.CLOUD_MASK_MODE in ("feature_detector", "combined"):
            full_bands = set(range(1, 14))
            loaded     = set(cls.S2_BANDS)
            missing    = {1, 3, 8, 11, 12} - loaded   # bands used by FD tests
            if missing:
                import warnings
                warnings.warn(
                    f"CLOUD_MASK_MODE='{cls.CLOUD_MASK_MODE}' works best with all 13 "
                    f"Sentinel-2 bands.  Missing bands {sorted(missing)} will cause "
                    f"some detection tests to be skipped (graceful fallback).",
                    UserWarning,
                    stacklevel=2,
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
        if cls.CLOUD_MASK_MODE in ("feature_detector", "combined"):
            print(f"    fd_threshold:    {cls.FD_CLOUD_THRESHOLD}")
            print(f"    moist_check:     {cls.FD_USE_MOIST_CHECK}")
            print(f"    shadow_as_cloud: {cls.FD_SHADOW_AS_CLOUD}")
        if cls.CLOUD_MASK_MODE == "combined":
            print(f"    gt_diff_weight:  {cls.GT_DIFF_WEIGHT}")
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