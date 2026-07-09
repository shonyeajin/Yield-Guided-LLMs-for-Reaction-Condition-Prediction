import torch
from config import PipelineConfig
from model import ChemicalGeneratorLLM, FrozenYieldPredictor

def run_inference_loop():
    config = PipelineConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    generator = ChemicalGeneratorLLM(config)
    yield_predictor = FrozenYieldPredictor(config).to(device)
    
    generator.eval()
    yield_predictor.eval()
    
    print("[추론 단계] 최신 오픈소스 LLM + 수율 Predictor 제어 루틴")
    
    # 초기 상태 (B, N)
    main_sequences = torch.randint(10, 500, (config.B, config.N)).to(device)
    U = [1, 2, 3, 4, 5] # 남은 화학 그룹 검색 공간
    
    while len(U) > 0:
        current_U_size = len(U)
        all_candidate_sequences = []
        
        # Step 1. 후보별 가상 디코딩 생성 연산
        for candidate_i in U:
            # 실제 서빙 시에는 여기서 generator.model.generate() 메서드를 호출하여 
            # 해당 화학 조건 영역의 텍스트 토큰을 끝까지 자동 완성(Autoregressive)하게 만듬
            with torch.no_grad():
                # 차원 정합성을 검증하기 위한 가상 가공 시퀀스 빌드
                virtual_seq_i = main_sequences.clone()
                virtual_seq_i[:, 50:100] = virtual_seq_i[:, 50:100] + candidate_i # 디코딩 효과 모방
                
            all_candidate_sequences.append(virtual_seq_i)
            
        # 텐서 취합 -> 차원: (B * |U|, N)
        virtual_sequences = torch.cat(all_candidate_sequences, dim=0)
        
        # Step 2. Frozen 수율 예측 모델 평가 (Yield Predictor Pass)
        with torch.no_grad():
            yield_scores_raw = yield_predictor(virtual_sequences) # (B * |U|, 1)
            
        # 차원 복원 정렬 -> (B, |U|)
        yield_scores = yield_scores_raw.view(config.B, current_U_size)
        
        # Step 3. Argmax 방향 정렬 및 결합 (Commit)
        next_group_indices = torch.argmax(yield_scores, dim=-1)
        
        print(f" -> 남은 후보군 수: {current_U_size} | 취합된 후보 텐서 구조: {virtual_sequences.shape} | 스칼라 점수 행렬: {yield_scores.shape}")
        
        # 가장 높은 확률의 다음 그룹 타겟 확정 후 U에서 소거하는 배치 프로세싱 진행
        chosen_idx = next_group_indices[0].item()
        confirmed_group = U[chosen_idx]
        U.remove(confirmed_group)
        
    print("\n[추론 루프 완료] 최적 수율 제어 오더 스트림 디코딩 정상 작동 완료.")

if __name__ == "__main__":
    run_inference_loop()
