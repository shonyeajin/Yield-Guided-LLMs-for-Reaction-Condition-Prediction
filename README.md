# Yield-Guided Chemical LLM Pipeline

최적의 화학 합성 조건을 탐색하는 End-to-End 파이프라인

---

## 🏗️ 시스템 아키텍처 및 파일 구조


- `config.py`: LLM 모델 ID 사양 및 핵심 하이퍼파라미터 정의
- `dataset.py`: 화학 데이터셋 토큰화 및 PyTorch DataLoader 구축
- `utils.py`: Hugging Face LLM 호환용 A3 특화형 4D Causal Mask 생성 장치
- `model.py`: Pre-trained LLM 백본 탑재 및 Frozen 수율 모델 정의
- `train.py`: DataLoader 기반 무작위 순열 및 NTP 학습 루프
- `inference.py`: 수율 가이드 기반 동적 검색 및 조건 확정 루틴

---

## 📐 텐서 요약 맵 (Tensor Dimension Mapping)

데이터가 입력되어 변환 및 연산되는 모든 과정을 단계별로 추적한 명세입니다.

| 프로세스 단계 | 텐서 변수명 | 데이터 타입 | 데이터 차원 (Shape) | 설명 |
| :--- | :--- | :--- | :--- | :--- |
| **공통 입력** | `input_ids` | `Int64` | `(B, N)` | 토큰화된 화학 반응식 문장 및 SMILES ID |
| **공통 입력** | `group_ids` | `Int64` | `(B, N)` | 토큰별 소속 그룹 번호 ($G_{\text{fixed}}$:0, $G_{\text{cat}}$:1, ...) |
| **LLM 내부** | `hidden_states` | `Float32` | `(B, N, D)` | 생성 LLM의 최종 은닉 레이어 텐서 |
| **LLM 출력** | `logits` | `Float32` | `(B, N, V)` | 단어 사전($V$)에 대한 다음 토큰 확률 점수 |
| **추론 루프** | `virtual_sequences` | `Int64` | `(B * \|U\|, N)` | 현재 루프 단계에서 가능한 후보군 조합을 병렬화한 가상 시퀀스 |
| **수율 출력** | `yield_scores` | `Float32` | `(B, \|U\|)` | 수율 모델 통과 후 배치 단위별로 재정렬된 예상 수율값 |
| **결정 단계** | `next_group` | `Int64` | `(B,)` | `dim=-1` 방향 축의 수율 최적화 조건 Index |

---

## 🛠️ 핵심 메커니즘 설명

### 1) A3 특화형 4D Causal Masking (`utils.py`)
학습 단계에서 화학 조건 그룹의 순서를 무작위로 뒤섞는(Permutation) 기법을 지원하기 위해, Hugging Face `transformers` 엔진에 직접 주입할 수 있는 `(B, 1, N, N)` 형태의 소프트맥스 가산형 마스크를 생성합니다. 
- **참조 가능 영역 (자기 자신 및 이전 위계 그룹)**: `0.0`
- **참조 차단 영역 (미래 토큰 혹은 순서상 배제된 그룹)**: `-10000.0`

### 2) 동적 가치 탐색 추적 (`inference.py`)
추론 단계에서는 가치 평가 모델인 `FrozenYieldPredictor`가 개입하여 다음에 생성할 최적의 화학 조건 그룹을 동적으로 지시합니다. 탐색 공간인 미완성 후보 집합 $\mathcal{U}$가 전부 소거될 때까지 **[가상 디코딩 $\rightarrow$ 병렬 수율 평가 $\rightarrow$ Argmax 최적 후보 Commit]** 과정을 반복 수행합니다.

---


## 🚀 시작 가이드

### 1) 환경 구성 및 필수 패키지 설치
거대 오픈소스 가중치를 안정적으로 핸들링하기 위해 아래 라이브러리 구성을 권장합니다.
```
pip install torch transformers accelerate
```


### 2) Hugging Face 인증 (Gated Model 사용 시)
meta-llama/Meta-Llama-3.1-8B-Instruct 등 접근 권한이 필요한 가중치를 베이스로 삼을 경우, 터미널에서 Hugging Face 허브 토큰을 인증해 주어야 합니다.

```bash
huggingface-cli login --token YOUR_HF_TOKEN
```

### 3) 파이프라인 실행 및 시뮬레이션
훈련 단계 검증 (DataLoader 및 A3 마스크 연동)


```bash
python train.py
```

추론 단계 검증 (수율 기반 동적 검색 루프 작동)


```bash
python inference.py
```
