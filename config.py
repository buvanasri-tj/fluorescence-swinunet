class Config:
    # Dataset root
    DATA_ROOT = "data"

    # Training parameters
    IMAGE_SIZE = 256
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LR = 1e-4

    # Segmentation model
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # Output directories
    CHECKPOINT_DIR = "checkpoints"
    PRED_DIR = "predictions"
    OVERLAY_DIR = "overlays"

    # Detection
    DET_IMAGE_SIZE = 512

    # Counting CSV
    COUNT_CSV = "data/counts.csv"


cfg = Config()
