from dataclasses import dataclass

@dataclass
class PipelineConfig:
    B: int = 4            # 배치 크기 (Batch Size)
    V: int = 1000         # 생성 LLM의 어휘 사전 크기 (Vocabulary Size)
    N: int = 128          # 전체 텍스트 시퀀스의 최대 토큰 길이 (Max Sequence Length)
    D: int = 256          # 생성 LLM의 은닉층 차원 (Hidden Dimension)
    K: int = 5            # 화학 조건 그룹의 개수
    lr: float = 5e-5
    epochs: int = 3
    M: int = 128          # 수율 모델 전용 차원
