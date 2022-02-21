from dataset import MaskingMethod

# Dataset selection
DATASET_SELECT = 'photo' # 'monet'
BASE_PROJECT_PATH = 'C:/Users/keller/ran/ContextEncoder4Inpainting/'
#'/home/ranz/projects/github_projects/dl4vision_course/context_encoders/'

# Train/validation set size
TRAIN_SET_RATIO = 0.8
VALID_SET_RATIO = 1 - TRAIN_SET_RATIO

# Transforms
TO_RESIZE = True
RESIZE_DIM = 128
TO_ADD_NOISE_TO_TRAIN_SET = False #True
TO_NORMALIZE = True #True

# Models Load / Save
ENABLE_MODEL_SAVE = True
if DATASET_SELECT == 'photo':
    MODEL_SAVE_PATH = BASE_PROJECT_PATH + 'models/photo'
    ENABLE_PRETRAINED_MODEL_LOAD = False#True
    PRETRAINED_MODEL_PATH = BASE_PROJECT_PATH + 'models/photo'
else:
    MODEL_SAVE_PATH = BASE_PROJECT_PATH + 'models/monet'
    ENABLE_PRETRAINED_MODEL_LOAD = True
    # for Monet dataset, we use transfer learning - we load the big dataset (photo) pretrained model
    PRETRAINED_MODEL_PATH = BASE_PROJECT_PATH + 'models/photo'

# Inpainting configuration
MASKING_METHOD = "CentralRegion" #"CentralRegion" "RandomBlock" "RandomRegion"
MASK_SIZE = 64
MASK_OVERLAP = 7
RANDOM_REGION_MASK_MAX_PIXELS = 2000 # Used only for the fully random mode
MAX_BLOCKS = 10 # Used for the RandomBlock masking method

# Training hyper-parameters
NUM_EPOCHS = 800 # 500

BATCH_SIZE = 64

DISC_LR = 0.00002#0.0001
DISC_BETA1 = 0.5
DISC_BETA2 = 0.999
LAMBDA_REC = 0.999
LAMBDA_ADV = 0.001#0.001

if MASKING_METHOD == "RandomRegion":
    GEN_LR = 0.002
else:
    GEN_LR = 0.002 

APPLY_GAUSSIAN_WEIGHT_INIT = True # only relevant in case there's no pretraining

# Derived constants on dataset (do not change)
DATASET_PATH = BASE_PROJECT_PATH + 'data/' + DATASET_SELECT
RANDOM_REGION_TEMPLATES_PATH = BASE_PROJECT_PATH + 'pascal'

# General configuration (not relevant to change)
if TO_RESIZE:
    IMAGE_SIZE = RESIZE_DIM
else:
    IMAGE_SIZE = 256
FIXED_RANDOM = True
RANDOM_SEED = 42

# Infrastructure configuration
NUM_OF_WORKERS_DATALOADER = 0#
USE_GPU = True     

# Debug flags
SHOW_IMAGE = False#True
SHOW_EXAMPLES_RESULTS_ON_VALID_SET = True
ENABLE_TENSORBOARD = True
UNNORM_DISPLAY = True
NUM_EPOCHS_PER_DISPLAY = 10
