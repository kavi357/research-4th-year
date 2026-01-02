import torch
from siamese_encoder import SimilarityNetwork

MODEL_PATH = "music_siamese_encoder.pth"

ckpt = torch.load(MODEL_PATH, map_location="cpu")

model = SimilarityNetwork(ckpt["input_dim"])
model.load_state_dict(ckpt["model_state"])
model.eval()
