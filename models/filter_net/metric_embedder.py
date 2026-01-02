import torch
import numpy as np
import joblib
from pathlib import Path

# ---------------- PATHS ----------------
BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "similarity_net_no_covers80.pth"
SCALER_PATH = BASE / "scaler_no_covers80.joblib"
PCA_PATH = BASE / "pca_384_no_covers80.joblib"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- MODEL DEF (MUST MATCH TRAINING) ----------------
class SimilarityNet(torch.nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128)
        )

    def forward(self, x):
        return torch.nn.functional.normalize(self.net(x), dim=1)

# ---------------- LOAD OBJECTS (ONCE) ----------------
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

model = SimilarityNet(input_dim=384)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- EMBEDDING FUNCTION ----------------
@torch.no_grad()
def embed_feature_vector(x: np.ndarray) -> np.ndarray:
    """
    x: (1591,) raw feature vector
    returns: (128,) learned embedding
    """
    x_scaled = scaler.transform(x.reshape(1, -1))
    x_pca = pca.transform(x_scaled)

    x_tensor = torch.tensor(x_pca, dtype=torch.float32, device=DEVICE)
    emb = model(x_tensor)

    return emb.cpu().numpy()[0]
