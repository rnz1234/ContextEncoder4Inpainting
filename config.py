


DATASET_PATH = '/home/ranz/projects/github_projects/dl4vision_course/context_encoders/data/photo'

TRAIN_SET_RATIO = 0.7
VALID_SET_RATIO = 1 - TRAIN_SET_RATIO

BATCH_SIZE = 32
NUM_OF_WORKERS_DATALOADER = 0 #1 #4

NUM_EPOCHS = 20

GEN_LR = 0.002

DISC_LR = 0.0002
DISC_BETA1 = 0.5
DISC_BETA2 = 0.999

LAMBDA_REC = 0.999
LAMBDA_ADV = 0.001

USE_GPU = False
