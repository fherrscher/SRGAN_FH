import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

#####################
# Start of user settings
#####################

LOAD_MODEL = False # Set to True to load the model from checkpoint and continue training
SAVE_MODEL = False # Set to True to save the model every epoch
INPUT_DIR = "data_copy" # Input directory (img data has to be in a subdirectory, e.g. data/img/)
CHECKPOINT_GEN = "checkpoints/gen.pth.tar" # Generator checkpoint file
CHECKPOINT_DISC = "checkpoints/disc.pth.tar" # Discriminator checkpoint file
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
DEVICE = "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 64
NUM_WORKERS = 4
HIGH_RES = 96
LOW_RES = HIGH_RES // 4 # Set the upscaling factor
IMG_CHANNELS = 3


USE_TENSORBOARD = True # Set to True to use tensorboard
TB_LOG_DIR = "runs/mnist/local_4" # Tensorboard log dir
PLOT_EPOCHS = 10 # Every X epochs plot the examples
EXAMPLE_IMAGE = "test/inp/input.jpg" # Example image for Tensorboard

TEST_DIR = "test/inp/" # Test directory (input images)
TEST_OUT_DIR = "test/out/" # Test directory (output images)

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
