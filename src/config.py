import os

# --- PATHS ---
# Update BASE_DIR to your exact Google Drive path where the project is located
BASE_DIR = '/content/drive/MyDrive/NLP_CSL_Recognition' 
DATA_DIR = os.path.join(BASE_DIR, 'data')
KEYPOINTS_DIR = os.path.join(DATA_DIR, 'processed_keypoints')
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Generated files
LABEL_MAP_FILE = os.path.join(METADATA_DIR, 'label_map.json')
SPLITS_FILE = os.path.join(METADATA_DIR, 'data_splits.json')

# --- HYPERPARAMETERS ---
# Max frames per video (shorter videos will be padded, longer will be truncated)
MAX_FRAMES = 60 

# IMPORTANT: Adjust FEATURE_DIM based on your MediaPipe extraction!
# e.g., Pose(33*4) + LH(21*3) + RH(21*3) = 258 
# If you included face landmarks, this will be much higher.
FEATURE_DIM = 225 

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 100