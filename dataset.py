# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import PipelineConfig

class ChemicalDataset(Dataset):
    def __init__(self, raw_data, config: PipelineConfig):
        self.config = config
        # 최신 오픈소스 LLM 패키징 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        text = item["text"]
        
        # 2-1 단계: 원시 문자열 토큰화 (Tokenization)
        encoding = self.tokenizer(
            text,
            max_length=self.config.N,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0) # (N,)
        
        # 2-2 단계: 그룹 마스크(Group Mask) 매핑 테이블 생성
        # 실제 환경에서는 정규식이나 SMILES 위치 기반으로 매핑하지만, 
        # 구조 검증을 위해 규칙 기반 토큰별 그룹 정수(0~5) 레이아웃을 생성
        group_ids = torch.zeros(self.config.N, dtype=torch.long)
        
        # 샘플용 가상 그룹 인덱싱 기입 (0: fixed, 1~5: 각 조건 그룹)
        # 예시로 토큰 구간별로 임의 할당
        group_ids[20:40] = 1   # G_cat
        group_ids[40:60] = 2   # G_sol1
        group_ids[60:80] = 3   # G_sol2
        group_ids[80:100] = 4  # G_rea1
        group_ids[100:120] = 5 # G_rea2
        
        # Next-Token Prediction 정답지 생성
        target_ids = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "group_ids": group_ids,
            "target_ids": target_ids
        }

def get_chemical_dataloader(config: PipelineConfig):
    # 가상의 원시 화학 반응 문장 데이터셋 샘플
    mock_raw_data = [
        {"text": "The reactant is CC(=O)O and the product is CC(=O)OCC. Catalyst is [Pd], Solvent1 is Water, Solvent2 is EtOH."},
        {"text": "The reactant is C1CCCCC1 and the product is C1=CC=CC=C1. Catalyst is [Pt], Solvent1 is THF, Solvent2 is DCM."},
    ] * 10 # 20개 샘플 시뮬레이션
    
    dataset = ChemicalDataset(mock_raw_data, config)
    dataloader = DataLoader(dataset, batch_size=config.B, shuffle=True)
    return dataloader
