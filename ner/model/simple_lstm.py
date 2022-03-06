import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SimpleLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        hidden_dim: int = 128,
        embed_dim: int = 300,
        dropout: float = 0.5,
        num_layers: int = 1,
        bidirectional: bool = True,
        ):
        super(SimpleLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim if not bidirectional else int(hidden_dim // 2)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(embed_dim, self.hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lens):
        x = self.embed(x)
        
        # Pack variable-length seqs
        packed = pack_padded_sequence(x, lens, batch_first=True)

        # Forward LSTM
        out, _ = self.lstm(packed)

        # Recover
        out, _ = pad_packed_sequence(out, batch_first=True)

        out = self.dropout(out)
        out = self.fc(out)

        return out