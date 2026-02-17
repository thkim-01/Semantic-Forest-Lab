# Semantic-Forest-lab

SMILES 기반 분자 구조를 **Drug Target Ontology (DTO)** 기반 온톨로지로 변환한 뒤,
설명가능한 결정트리를 **배깅(bootstrap aggregating)** 으로 학습해 분류 성능을 높이는 실험 레포입니다.

이 레포는 알고리즘별 버전을 **하나의 프로젝트 안에서** 관리합니다:

- **ID3**: Information Gain (`information_gain`)
- **C4.5**: Gain Ratio (`gain_ratio`)
- **CART**: Gini impurity (`gini`)

## 주요 특징

- **단일 레포 다중 알고리즘 버전 관리**: ID3/C4.5/CART 공존
- **알고리즘 프로파일 기반 실행**: `configs/algorithms/*.json`
- **온톨로지 기반**: Drug Target Ontology를 활용한 의미론적 결정트리
- **Random Forest**: 배깅을 통한 앙상블 학습으로 성능 향상
- **설명 가능성**: 의사결정 과정을 명확하게 추적 가능

단일 트리로 학습하는 버전은 별도 레포로 분리했습니다:

- https://github.com/thkim-01/Semantic-Decision-Tree

## 개발 환경

- Python: 3.9+
- 주요 의존성: `owlready2`, `rdkit`, `scikit-learn`, `pandas`, `numpy`
- 선택 의존성(가속): `torch` (CUDA 가능 환경에서 GPU 사용)

## 설치

```bash
pip install -r requirements.txt

# (선택) PyTorch 가속 사용 시
# pip install torch
```

## 빠른 시작 (Quick Start)

### 0. 알고리즘 버전 실행 (권장)

```bash
# ID3 버전 실행
python experiments/run_semantic_forest_lab.py --algorithm id3

# C4.5 버전 실행
python experiments/run_semantic_forest_lab.py --algorithm c45

# CART 버전 실행
python experiments/run_semantic_forest_lab.py --algorithm cart

# (선택) torch 백엔드 사용
python experiments/run_semantic_forest_lab.py --algorithm id3 --compute-backend torch --torch-device auto
```

> 참고: 현재 코드의 주요 병목 중 일부는 온톨로지 객체 순회/정제(refinement) 판정 로직입니다.
> PyTorch 가속은 impurity/entropy 계산을 우선 가속하며, 전체 파이프라인을 완전 GPU-only로 바꾸지는 않습니다.

### 1. 단일 데이터셋 테스트 (BBBP)

```bash
# BBBP 데이터셋으로 Semantic Forest 학습 및 평가
python experiments/verify_semantic_forest.py --split-criterion information_gain
```

### 2. 전체 벤치마크 실행

```bash
# 모든 데이터셋에 대해 성능 평가
python experiments/verify_semantic_forest_multi.py

# 결과는 output/semantic_forest_benchmark.csv에 저장됩니다
```

### 3. 커스텀 설정으로 실행

```bash
# 특정 데이터셋만 실행
python experiments/verify_semantic_forest_multi.py --datasets bbbp,clintox

# 트리 개수 및 깊이 조정
python experiments/verify_semantic_forest_multi.py --n-estimators 50 --max-depth 8

# 분할 기준 변경
python experiments/verify_semantic_forest_multi.py --split-criterion information_gain
python experiments/verify_semantic_forest_multi.py --split-criterion gain_ratio
python experiments/verify_semantic_forest_multi.py --split-criterion gini
```

## 버전 구조

```text
configs/algorithms/
	id3.json
	c45.json
	cart.json
src/algorithms/
	profiles.py
experiments/
	run_semantic_forest_lab.py
```

## 알고리즘 설명

### ID3 (Iterative Dichotomiser 3)

- **분할 기준**: Information Gain을 사용하여 최적의 분할 지점 선택
- **Entropy**: $H(S) = -\sum_{i=1}^{n} p_i \log_2 p_i$
- **Information Gain**: $IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$
- **특징**: 해석이 직관적이며 엔트로피 기반으로 불확실성 감소를 최대화

### C4.5

- **분할 기준**: Gain Ratio
- **핵심 아이디어**: Information Gain을 split info로 정규화

### CART

- **분할 기준**: Gini impurity
- **Gini impurity**: $Gini = 1 - \sum_{i=1}^{n} p_i^2$

### 처리 과정

1. **입력**: `data/` 디렉토리의 CSV 파일 (예: `data/bbbp/BBBP.csv`)
2. **전처리**: SMILES → RDKit을 통한 분자 피처 추출
3. **온톨로지 변환**: DTO 기반 온톨로지 인스턴스 생성
4. **학습**: ID3 정보이득 기준으로 여러 트리 학습 (배깅)
5. **평가**: AUC-ROC, Accuracy 등 성능 지표 계산

## 데이터셋

지원하는 데이터셋:
- BBBP (Blood-Brain Barrier Penetration)
- BACE (β-secretase inhibitors)
- ClinTox (Clinical toxicity)
- HIV (HIV replication inhibition)
- Tox21 (Toxicity prediction)
- SIDER (Side effects)
- 기타 분자 특성 예측 데이터셋

## 출력 결과

- **콘솔 로그**: 학습 진행 상황 및 성능 지표
- **CSV 파일**: `output/semantic_forest_benchmark.csv` - 전체 벤치마크 결과
- **요약 파일**: 개별 실험 결과 요약

## 라이센스

본 프로젝트는 연구 및 교육 목적으로 사용 가능합니다.
