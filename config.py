from dataset import MaskingMethod

DATASET_SELECT = 'photo'

DATASET_PATH = 'c:/Users/keller/ran/ContextEncoder4Inpainting/data/' + DATASET_SELECT

TRAIN_SET_RATIO = 0.1#0.8
VALID_SET_RATIO = 0.1#1 - TRAIN_SET_RATIO


NUM_OF_WORKERS_DATALOADER = 0#4

NUM_EPOCHS = 100

GEN_LR = 0.0001 #0.002

DISC_LR = 0.0002
DISC_BETA1 = 0.5
DISC_BETA2 = 0.999

LAMBDA_REC = 0.999
LAMBDA_ADV = 0.001

USE_GPU = True

IMAGE_SIZE = 256


FIXED_RANDOM = True
RANDOM_SEED = 42

MASKING_METHOD = "RandomBlock" #"CentralRegion" "RandomBlock" "RandomRegion"

if MASKING_METHOD == "RandomRegion":
    MASK_SIZE = 1000
else:
    MASK_SIZE = 128


if MASKING_METHOD == "CentralRegion":
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 64




MAX_BLOCKS = 10

USE_GPU = True

SHOW_IMAGE = False#True
