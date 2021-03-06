from dataset import MaskingMethod

# Dataset selection
DATASET_SELECT = 'photo' # 'photo' / 'monet'
BASE_PROJECT_PATH = 'C:/Users/keller/ran/ContextEncoder4Inpainting/'
#'/home/ranz/projects/github_projects/dl4vision_course/context_encoders/'

# Train/validation set size
TRAIN_SET_RATIO = 0.8
VALID_SET_RATIO = 1 - TRAIN_SET_RATIO

# Transforms
TO_RESIZE = True
RESIZE_DIM = 128
TO_ADD_NOISE_TO_TRAIN_SET = False #True #True
TO_NORMALIZE = True


# Models Load / Save
ENABLE_MODEL_SAVE = True
if DATASET_SELECT == 'photo':
    MODEL_SAVE_PATH = BASE_PROJECT_PATH + 'models/photo'
    ENABLE_PRETRAINED_MODEL_LOAD = False#True
    PRETRAINED_MODEL_PATH = BASE_PROJECT_PATH + 'models/photo/good_model_random_region'
    PRETRAINED_MODEL_PATH_FOR_EVAL = PRETRAINED_MODEL_PATH
else:
    MODEL_SAVE_PATH = BASE_PROJECT_PATH + 'models/monet'
    ENABLE_PRETRAINED_MODEL_LOAD = True
    # for Monet dataset, we use transfer learning - we load the big dataset (photo) pretrained model
    PRETRAINED_MODEL_PATH = BASE_PROJECT_PATH + 'models/photo/good_model_random_region'
    PRETRAINED_MODEL_PATH_FOR_EVAL = BASE_PROJECT_PATH + 'models/monet/good_model_random_region'

# Inpainting configuration
MASKING_METHOD = "RandomRegion" #"CentralRegion" "RandomBlock" "RandomRegion"
MASK_SIZE = 64
MASK_OVERLAP = 7
RANDOM_REGION_MASK_MAX_PIXELS = 2000 # Used only for the fully random mode
MAX_BLOCKS = 10 # Used for the RandomBlock masking method

# Training hyper-parameters
DISC_BETA1 = 0.5
DISC_BETA2 = 0.999

if DATASET_SELECT == 'photo':
    LAMBDA_STYLE = 0
    LAMBDA_REC = 0.999
    LAMBDA_ADV = 0.001#0.001
    if MASKING_METHOD == "RandomRegion":
        GEN_LR = 0.002
        DISC_LR = 0.00002
        NUM_EPOCHS = 370
        BATCH_SIZE = 128
        ENABLE_AUGMENTATIONS = False
        AUGMENTATIONS_AMOUNT = 4
    elif MASKING_METHOD == "CentralRegion":
        GEN_LR = 0.002
        DISC_LR = 0.0002
        BATCH_SIZE = 128
        if MASK_SIZE == 64:
            NUM_EPOCHS = 120
        else:
            NUM_EPOCHS = 160
        ENABLE_AUGMENTATIONS = False
        AUGMENTATIONS_AMOUNT = 4
    else: # RandomBlock
        GEN_LR = 0.002
        DISC_LR = 0.00002
        NUM_EPOCHS = 370
        BATCH_SIZE = 128
        ENABLE_AUGMENTATIONS = False
        AUGMENTATIONS_AMOUNT = 4
    WEIGHT_DECAY = False
    WEIGHT_DECAY_VAL = 0.001
    NET_CROSS_STYLE_LOSS = False
    EXTERNAL_REF_NET_CROSS_STYLE_LOSS = False
else: # 'monet'
    LAMBDA_STYLE = 2000 
    LAMBDA_REC = 0.999#0.099
    LAMBDA_ADV = 0.001#0.001
    if MASKING_METHOD == "RandomRegion":
        GEN_LR = 0.0002
        DISC_LR = 0.00002
        NUM_EPOCHS = 370
        BATCH_SIZE = 32
        ENABLE_AUGMENTATIONS = True #True
        AUGMENTATIONS_AMOUNT = 8
    elif MASKING_METHOD == "CentralRegion":
        GEN_LR = 0.0002
        DISC_LR = 0.00002
        BATCH_SIZE = 32
        if MASK_SIZE == 64:
            NUM_EPOCHS = 100
        else:
            NUM_EPOCHS = 160
        ENABLE_AUGMENTATIONS = True #True
        AUGMENTATIONS_AMOUNT = 8
    else: # RandomBlock
        GEN_LR = 0.0002
        DISC_LR = 0.00002
        NUM_EPOCHS = 100
        BATCH_SIZE = 32
        ENABLE_AUGMENTATIONS = True #True
        AUGMENTATIONS_AMOUNT = 8
    WEIGHT_DECAY = False
    WEIGHT_DECAY_VAL = 0.000001
    NET_CROSS_STYLE_LOSS = False
    EXTERNAL_REF_NET_CROSS_STYLE_LOSS = True



DOWNSCALE_GEN_TRAIN = False
DOWNSCALE_GEN_TRAIN_RATIO = 5

APPLY_GAUSSIAN_WEIGHT_INIT = True # only relevant in case there's no pretraining

# Derived constants on dataset (do not change)
DATASET_PATH = BASE_PROJECT_PATH + 'data/' + DATASET_SELECT
RANDOM_REGION_TEMPLATES_PATH = BASE_PROJECT_PATH + 'data/masks_for_random_region'
RANDOM_BLOCK_TEMPLATES_PATH = BASE_PROJECT_PATH + 'data/masks_for_random_block'

# General configuration (not relevant to change)
ORIG_IMAGE_SIZE = 256
if TO_RESIZE:
    IMAGE_SIZE = RESIZE_DIM
else:
    IMAGE_SIZE = ORIG_IMAGE_SIZE
FIXED_RANDOM = True
RANDOM_SEED = 42

# Infrastructure configuration
NUM_OF_WORKERS_DATALOADER = 0#
USE_GPU = True     

# Debug flags
SHOW_IMAGE = False#True
SHOW_EXAMPLES_RESULTS_ON_VALID_SET = True
ENABLE_TENSORBOARD = True
#UNNORM_DISPLAY = True
NUM_EPOCHS_PER_DISPLAY = 5
CANCEL_ADV_TRAIN = False #True
NUM_OF_IMAGES_DISPLAY_ON_TRAIN = 8