import torch

MODEL_NAME = "lvwerra/distilbert-imdb"
DATASET_NAME = "imdb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TOKENS = 128

LOG_FILE_PATH = "data/imbd_inference_logs.npz"


## HMM CONFIG

NUM_TRAIN_SAMPLES = 500 # NUMBER OF REVIEWS TO LOG FOR HMM TRAINING
NUM_TEST_SAMPLES = 20 # NUMBER OF REVIEWS TO LOG FOR HMM TESTING

NUM_HMM_STATES = 3
HMM_MODEL_PATH = "models/sentiment_hmm.pkl"

HMM_N_ITER = 100
#HMM_TOL = 1e-4
#HMM_COV_TYPE = "diag"



PROB_THRESHOLDS = {"NEG_OBS":0.4,
                   "NEU_OBS": 0.6,
                   "POS_OBS": 1}

TARGET_SENTIMENT = 1 # 1 for positive, 0 for negative




