import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityNetwork(nn.Module):
    def __init__(self, input_dim=1570):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64)
        )

    def forward(self, x):
        """
        Returns L2-normalized embedding (shape: [batch, 64])
        """
        x = self.encoder(x)
        return F.normalize(x, dim=1)

    @staticmethod
    def cosine_similarity(emb1, emb2):
        """
        Returns raw cosine similarity in range [-1, 1]
        """
        return F.cosine_similarity(emb1, emb2)
