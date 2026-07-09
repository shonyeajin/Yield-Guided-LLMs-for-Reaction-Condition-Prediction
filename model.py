import torch
import torch.nn as nn
from config import PipelineConfig

class ChemicalGeneratorLLM(nn.Module):
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.V, config.D)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.D, nhead=4, batch_first=True
        )
        self.lm_head = nn.Linear(config.D, config.V)

    def forward(self, input_ids, causal_mask=None):
        hidden_states = self.embedding(input_ids)
        if causal_mask is not None:
            mask_squeezed = causal_mask.squeeze(1)
            attn_mask = torch.zeros_like(mask_squeezed, dtype=torch.float32)
            attn_mask = attn_mask.masked_fill(mask_squeezed == 0, float('-inf'))
            hidden_states = self.transformer_layer(hidden_states, mask=attn_mask)
        else:
            hidden_states = self.transformer_layer(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, hidden_states

class FrozenYieldPredictor(nn.Module):
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.V, config.D)
        self.encoder = nn.TransformerEncoderLayer(d_model=config.D, nhead=4, batch_first=True)
        self.regression_head = nn.Sequential(
            nn.Linear(config.D, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, virtual_sequences):
        x = self.embedding(virtual_sequences)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        yield_scores = self.regression_head(pooled)
        return yield_scores
