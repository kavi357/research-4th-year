import tensorflow_hub as hub
import numpy as np
import librosa
from pathlib import Path

# ------------------------------------------------------
# Resolve project ROOT dynamically
# backend/ingest/extract_yamnet.py → parents[2] = project root
# ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

# Path to local YAMNet model
YAMNET_PATH = ROOT / "models" / "yamnet"

print("🔍 Loading YAMNet model from:", YAMNET_PATH)

# Load the TF Hub SavedModel
YAMNET_MODEL = hub.load(str(YAMNET_PATH))


def extract_yamnet_embedding(y, sr):
    """
    Extract a single YAMNet embedding (mean over frames).
    Input:
        y  - waveform (numpy array)
        sr - sample rate of y
    Output:
        1024-dim numpy vector
    """

    # Convert to mono if stereo
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Ensure type float32
    y = y.astype(np.float32)

    # Resample to 16 kHz because YAMNet requires it
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Run YAMNet model (returns frame-level embeddings)
    scores, embeddings, spectrogram = YAMNET_MODEL(y)

    # Convert EagerTensor → numpy
    embeddings = embeddings.numpy()

    # Mean pool across time frames → fixed 1024-d vector
    emb_vector = np.mean(embeddings, axis=0).astype(np.float32)

    return emb_vector
