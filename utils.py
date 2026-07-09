import torch
from config import PipelineConfig

def generate_a3_causal_mask(group_ids, config: PipelineConfig):
    B, N = group_ids.shape
    device = group_ids.device
    
    # tgt_group: 행 (정보를 받는 토큰의 그룹), src_group: 열 (정보를 주는 토큰의 그룹)
    tgt_group = group_ids.unsqueeze(2).expand(B, N, N)
    src_group = group_ids.unsqueeze(1).expand(B, N, N)
    
    # 기본 A3 메커니즘 규칙 적용: 
    # 자기 자신 그룹 이상이거나, 소스 그룹이 Fixed(0)인 경우만 True (참조 가능)
    # 그 외의 미래/순서상 뒤섞인 그룹 정보는 False (참조 차단)
    accessible_mask = (tgt_group >= src_group) | (src_group == 0)
    
    # Hugging Face Forwrad 주입용 4차원 텐서 구조 변환: (B, 1, N, N)
    # Softmax 직전에 더해지므로 참조 가능한 곳은 0.0, 차단할 곳은 큰 음수(-10000.0)로 세팅
    hf_attention_mask = torch.zeros((B, 1, N, N), dtype=torch.float32, device=device)
    hf_attention_mask = hf_attention_mask.masked_fill(~accessible_mask.unsqueeze(1), -10000.0)
    
    return hf_attention_mask
