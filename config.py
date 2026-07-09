from dataclasses import dataclass

@dataclass
class PipelineConfig:
    # 오픈소스 LLM 및 토크나이저 설정 (예: Llama-3.1-8B, Gemma-2-9b 등
    model_name_or_path: str = "google/gemma-2-2b-it" 
    
    # 기본 배치 및 길이 정의 (명세 기반)
    B: int = 2            # DataLoader 배치 크기
    N: int = 256          # 최대 토큰 길이 (최신 LLM에 맞게 확장 가능)
    D: int = 2048         # Gemma-2-2B 기준 Hidden Dimension (모델 로드 시 자동 매핑 가능)
    V: int = 256000       # Gemma-2 기준 Vocab Size (모델 설정에 맞춤)
    K: int = 5            # 예측해야 할 화학 조건 그룹의 개수
    
    lr: float = 2e-5
    epochs: int = 3
