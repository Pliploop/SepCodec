import torch
import torch.nn as nn
import math
from torch.nn import Module


class PositionalEncoding(Module):

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 1024, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if self.batch_first:
            x = x + self.pe.permute(1,0,2)[:,:x.size(1),:]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        


# class LearnedPositionalEncoding(Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         # Define a learnable parameter for positional encodings
#         self.positional_encodings = nn.Parameter(
#             torch.randn(max_len, 1, d_model))

#     def forward(self, x):
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         # Get the positional encodings
#         pe = self.positional_encodings[:x.size(0)]

#         # Add positional encodings to the input
#         x = x + pe

#         return self.dropout(x)

