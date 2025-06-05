import torch

MODEL_NAME = "lvwerra/distilbert-imdb"
DATASET_NAME = "imdb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TOKENS = 100 

LOG_FILE_PATH = "data/imbd_inference_logs25k.npz"
COLOR_LOG_FILE_PATH = "data/clustering/color_transitions_log.pkl"
PLOT_SAVE_PATH="notebooks/plots"

## HMM CONFIG

NUM_TRAIN_SAMPLES = 25000 # NUMBER OF REVIEWS TO LOG FOR HMM TRAINING
NUM_TEST_SAMPLES = 20 # NUMBER OF REVIEWS TO LOG FOR HMM TESTING

NUM_HMM_STATES = 4
HMM_MODEL_PATH = "models/sentiment_hmm25k_4states.pkl"

HMM_N_ITER = 500
HMM_TOL = 1e-4
HMM_COV_TYPE = "full"



PROB_THRESHOLDS = {"NEG_OBS":0.3,
                   "NEU_OBS": 0.7,
                   "POS_OBS": 1}

TARGET_SENTIMENT = 1 # 1 for positive, 0 for negative




