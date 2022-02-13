from dataset import MaskingMethod

# Dataset selection
DATASET_SELECT = 'photo' # 'monet'
BASE_PROJECT_PATH = '/home/ranz/projects/github_projects/dl4vision_course/context_encoders/'

# Train/validation set size
TRAIN_SET_RATIO = 0.8
VALID_SET_RATIO = 1 - TRAIN_SET_RATIO

# Models Load / Save
ENABLE_MODEL_SAVE = True
if DATASET_SELECT == 'photo':
    MODEL_SAVE_PATH = BASE_PROJECT_PATH + 'models/photo'
    ENABLE_PRETRAINED_MODEL_LOAD = False
else:
    MODEL_SAVE_PATH = BASE_PROJECT_PATH + 'models/monet'
    ENABLE_PRETRAINED_MODEL_LOAD = True
    # for Monet dataset, we use transfer learning - we load the big dataset (photo) pretrained model
    PRETRIANED_MODEL_PATH = BASE_PROJECT_PATH + 'models/photo'

# Inpainting configuration
MASKING_METHOD = "CentralRegion" #"CentralRegion" "RandomBlock" "RandomRegion"
MASK_SIZE = 128
RANDOM_REGION_MASK_MAX_PIXELS = 2000 # Used only for the fully random mode
MAX_BLOCKS = 10 # Used for the RandomBlock masking method

# Training hyper-parameters
NUM_EPOCHS = 120

BATCH_SIZE = 32

DISC_LR = 0.0002
DISC_BETA1 = 0.5
DISC_BETA2 = 0.999
LAMBDA_REC = 0.999
LAMBDA_ADV = 0.001

if MASKING_METHOD == "RandomRegion":
    GEN_LR = 0.0001 #0.002
else:
    GEN_LR = 0.002 

# Derived constants on dataset (do not change)
DATASET_PATH = BASE_PROJECT_PATH + 'data/' + DATASET_SELECT
RANDOM_REGION_TEMPLATES_PATH = BASE_PROJECT_PATH + 'pascal'

# General configuration (not relevant to change)
IMAGE_SIZE = 256
FIXED_RANDOM = True
RANDOM_SEED = 42

# Infrastructure configuration
NUM_OF_WORKERS_DATALOADER = 0#4
USE_GPU = True     

# Debug flags
SHOW_IMAGE = False#True
SHOW_EXAMPLES_RESULTS_ON_VALID_SET = True
ENABLE_TENSORBOARD = True
