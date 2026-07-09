import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from config import PipelineConfig

class ChemicalGeneratorLLM(nn.Module):
    """
    명세 3단계: Hugging Face Pre-trained LLM 기반 생성기
    """
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        
        # 최신 오픈소스 가중치 로드 
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 명세 상 차원 동기화
        self.hf_config = AutoConfig.from_pretrained(config.model_name_or_path)
        config.D = self.hf_config.hidden_size
        config.V = self.hf_config.vocab_size

    def forward(self, input_ids, causal_mask=None):
        # input_ids: (B, N)
        
        # Hugging Face 모델의 인풋에 맞춰 attention_mask 가공
        # causal_mask가 주어지면 custom 4D attention mask 형식을 Hugging Face 모델에 주입
        # 최신 transformers 패키지는 4차원 텐서 마스크 (B, 1, N, N) 입력을 직접 지원
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=causal_mask, # A3 특화형 4D Mask 반영
            output_hidden_states=True
        )
        
        logits = outputs.logits # (B, N, V)
        hidden_states = outputs.hidden_states[-1] # 최종 히든 스테이트 (B, N, D)
        
        return logits, hidden_states

class FrozenYieldPredictor(nn.Module):
    """
    명세 4단계: 수율 예측 모델 (Frozen)
    일반적으로 이 파트도 별도의 Pre-trained BERT류나 전용 Encoder 가중치를 로드
    """
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        
        # 가상의 전용 인코더 백본 구조 생성 (실제 가중치 파일 있을 시 로드)
        self.embedding = nn.Embedding(config.V, config.D)
        self.encoder = nn.TransformerEncoderLayer(d_model=config.D, nhead=8, batch_first=True, dtype=torch.float16)
        self.regression_head = nn.Sequential(
            nn.Linear(config.D, 128, dtype=torch.float16),
            nn.ReLU(),
            nn.Linear(128, 1, dtype=torch.float16),
            nn.Sigmoid()
        )
        
        # 가중치 완전히 동결
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, virtual_sequences):
        # virtual_sequences: (B * |U|, N)
        x = self.embedding(virtual_sequences)
        x = self.encoder(x)
        pooled = x.mean(dim=1) # 양방향 어텐션 후 풀링
        yield_scores = self.regression_head(pooled) # (B * |U|, 1)
        return yield_scores
