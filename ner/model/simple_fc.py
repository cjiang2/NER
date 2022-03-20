import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        hidden_dim: int = 128,
        embed_dim: int = 300,
        ):
        super(SimpleFC, self).__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lens):
        x = self.embed(x)
        
        x = self.fc1(x)
        x = self.fc(x)

        return x.permute(0, 2, 1)