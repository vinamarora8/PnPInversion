import torch
from torch import nn, Tensor

class TextEncoderAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = max(input_dim, output_dim)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.mask_emb = nn.Parameter(torch.randn(output_dim))

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        out = self.adapter(embeddings)
        mask = mask.to(out.dtype)
        out = out * mask.unsqueeze(-1) + self.mask_emb * (1 - mask.unsqueeze(-1))
        return out
