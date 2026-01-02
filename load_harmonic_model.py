import torch
import torch.nn as nn
import torch.nn.functional as F

model_path = "models/harmonic_encoder/harmonic_encoder.pth"


# Define the model class exactly as in training
class HarmonicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(240, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load the model state
model = HarmonicNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ Model loaded successfully!")
