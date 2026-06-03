from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Raw data lives in the project root alongside the PGN file
RAW_DIR = Path(r"C:\Users\chifl\OneDrive\Desktop\IS527 CHESS PROJECT")

INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

PGN_FILE = RAW_DIR / "lichess_db_standard_rated_2013-02.pgn"

# Minimum games to include a player in the graph
MIN_GAMES = 3

# Minimum games to include a player in the feature matrix / model
MIN_FEATURES_GAMES = 5

# Node2Vec hyperparameters
N2V_DIMENSIONS = 32
N2V_WALK_LENGTH = 20
N2V_NUM_WALKS = 50
N2V_WORKERS = 4
N2V_P = 1.0
N2V_Q = 1.0

# Approximate betweenness: number of pivot nodes to sample
# (set to None to compute exactly -- slow on large graphs)
BETWEENNESS_K = 500

# XGBoost hyperparameters
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_RANDOM_STATE = 42
XGB_EARLY_STOPPING_ROUNDS = 30

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Output filenames
MODEL_FILENAME = "xgb_rating_model.joblib"
