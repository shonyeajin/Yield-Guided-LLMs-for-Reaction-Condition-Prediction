# train.py
import torch
import torch.nn as nn
from config import PipelineConfig
from dataset import get_chemical_dataloader
from model import ChemicalGeneratorLLM
from utils import generate_a3_causal_mask

def train_pipeline():
    config = PipelineConfig()
    
    # 1. DataLoader 로드
    dataloader = get_chemical_dataloader(config)
    
    # 2. 오픈소스 LLM 빌드 (Gemma/Llama 가중치 인스턴스화)
    generator = ChemicalGeneratorLLM(config)
    generator.train()
    
    optimizer = torch.optim.AdamW(generator.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[훈련 단계] 실제 DataLoader 기반 루프 시작 (Device: {device})")
    
    for epoch in range(config.epochs):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            group_ids = batch["group_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            # A3 특화형 Causal Mask 생성 -> (B, 1, N, N)
            causal_mask = generate_a3_causal_mask(group_ids, config).to(device)
            
            # Forward 통과
            logits, hidden_states = generator(input_ids, causal_mask)
            
            # Cross-Entropy Loss (Next-Token Prediction 목적)
            loss = criterion(logits.view(-1, config.V), target_ids.view(-1))
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 2 == 0:
                print(f"Epoch [{epoch+1}/{config.epochs}] | Batch {batch_idx} | Loss: {loss.item():.4f}")
                print(f" -> 텐서 차원 추적 확인: Logits {logits.shape}, Hidden {hidden_states.shape}")
                break # 시뮬레이션을 위해 루프 탈출 설정
                
    print("[훈련 단계 완료]\n")

if __name__ == "__main__":
    train_pipeline()
