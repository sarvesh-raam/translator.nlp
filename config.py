import os

# Shared between preprocessing, training and serving; the saved weights only load
# if these match the values they were trained with.
DATA_DIR = os.environ.get("NMT_DATA_DIR", "data")
MODELS_DIR = os.environ.get("NMT_MODELS_DIR", "models")

MAX_SEQ_LEN = 30
EMBEDDING_DIM = 256
UNITS = 256
