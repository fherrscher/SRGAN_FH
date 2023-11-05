import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

#####################
# Start of user settings
#####################

LOAD_MODEL = False
SAVE_MODEL = True
INPUT_DIR = "data_copy"
CHECKPOINT_GEN = "checkpoints/gen.pth.tar"
CHECKPOINT_DISC = "checkpoints/disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 32
NUM_WORKERS = 4
HIGH_RES = 96
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3


TEST_DIR = "test/inp/"
TEST_OUT_DIR = "test/out/"

## Tensorboard implementation
USE_TENSORBOARD = True
TB_LOG_DIR = "runs/mnist/run2"

#####################
# End of user settings
#####################




highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)
