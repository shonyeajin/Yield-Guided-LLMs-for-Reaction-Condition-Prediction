import torch
from config import PipelineConfig

def generate_a3_causal_mask(group_ids, config: PipelineConfig):
    B, N = group_ids.shape
    tgt_group = group_ids.unsqueeze(2).expand(B, N, N)
    src_group = group_ids.unsqueeze(1).expand(B, N, N)
    mask = (tgt_group >= src_group) | (src_group == 0)
    return mask.unsqueeze(1).long()
