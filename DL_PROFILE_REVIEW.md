# Description Logic Profile 검토 보고서

## 현재 설정

| Dataset | Ontology | Current DL | 평가 | 추천 |
|---------|----------|-----------|------|------|
| BBBP | ChEBI | ALC | ✓ | ALC (유지) |
| BACE | DTO | EL | ⚠️ | **ALC로 변경** |
| ClinTox | ChEBI + DTO | EL | ⚠️ | **ALC로 변경** |
| HIV | Thesaurus + BAO | ALC | ✓ | ALC (유지) |
| Tox21 | PATO + GO + BAO | EL | ⚠️ | **ALC로 변경** |
| SIDER | MeSH | EL | ✓ | EL (유지) |

## 상세 분석

### ✓ 적절한 설정

#### 1. BBBP (Blood-Brain Barrier Penetration)
- **Ontology**: ChEBI (화학실체 분류)
- **DL**: ALC ✓
- **평가 이유**:
  - 화학 구조의 성질을 표현: 친수성/소수성 보수
  - `Not(Lipophilic)` 같은 보수 연산 가능
  - 분자 구조 특성의 정확한 표현에 ALC 필요
  - **권장**: 유지

#### 2. HIV (HIV Replication Inhibition)
- **Ontology**: Thesaurus (NCIT) + BAO (Bioassay)
- **DL**: ALC ✓
- **평가 이유**:
  - 질병 개념(Thesaurus)과 실험 방법(BAO) 결합
  - `Not(ViralWildType)`, `Complement(InhibitoryMechanism)` 필요
  - HIV like molecules ⟷ 미확인 메커니즘 구분
  - **권장**: 유지

#### 3. SIDER (Side Effects)
- **Ontology**: MeSH (의학 주제어 계층)
- **DL**: EL ✓
- **평가 이유**:
  - MeSH는 well-structured 계층 구조
  - 부작용 분류: 일반적으로 포함/배제 관계
  - 복잡한 보수 연산 불필요
  - **권장**: 유지

---

### ⚠️ 개선 필요

#### 4. BACE (β-secretase Inhibitors) - **ALC로 변경 권장**
- **Current**: DTO + EL
- **Issue**:
  - DTO는 Drug-Target 상호작용의 **구조적 관계** 표현
  - 약물이 타깃에 **NOT bind**하는 경우도 구분 필요
  - EL은 `∃ bindsTarget.BetaSecretase` 만 표현
  - ALC로 `¬(bindsTarget.WrongTarget)` 표현 가능
- **개선 효과**: 
  - ✓ 음성 사례(Non-inhibitor) 더 명확하게 구분
  - ✓ 단백질 제외 조건: `∃ bindsTarget.BetaSecretase ⊓ ¬(bindsTarget.Protease)`
- **권장 DL**: **ALC**

#### 5. ClinTox (Clinical Toxicity) - **ALC로 변경 권장**
- **Current**: ChEBI + DTO + EL
- **Issue**:
  - **2개 ontology 결합**: 화학 구조(ChEBI) + 약물 특성(DTO)
  - 동일 분자가 어떤 맥락에서는 독성, 다른 맥락에서는 안전
  - EL: `∃ hasClinicProperty.ClinicallyApproved`
  - ALC: `∃ hasClinicProperty.ClinicallyApproved ⊓ ¬(hasStructure.Mutagenic)`
- **개선 효과**:
  - ✓ 구조적 위험 인자 명시적 제외 가능
  - ✓ FDA_APPROVED vs CT_TOX 구분 강화
- **권장 DL**: **ALC**

#### 6. Tox21 (Toxicity Prediction) - **ALC로 변경 권장**
- **Current**: PATO + GO + BAO + EL
- **Issue**:
  - **3개 ontology 결합**: 생물학적 과정(GO) + 세포 표현형(PATO) + 실험(BAO)
  - 12개의 다양한 독성 메커니즘 예측
  - 예: `SR-MMP` (매트릭스 메탈로프로테이제):
    - EL 만: `∃ induces.MatrixMetalloproteinase`
    - ALC로: `∃ induces.MatrixMetalloproteinase ⊓ ¬(protectedBy.TIMP)`
  - Assay 특이성(Specificity) 표현에 보수 필요
- **개선 효과**:
  - ✓ NR(Nuclear Receptor) vs SR(Stress Response) 경계 명확
  - ✓ False positive 필터링 강화
  - ✓ LBD(Ligand Binding Domain) vs full domain 구분
- **권장 DL**: **ALC**

---

## 권장 개선안

### 옵션 1: 보수적 접근 (현재 설정 유지 + 성능 비교)
```json
{
  "bbbp": "ALC",    // ✓ 유지
  "bace": "EL",     // 유지 후 성능 벤치마크
  "clintox": "EL",  // 유지 후 성능 벤치마크
  "hiv": "ALC",     // ✓ 유지
  "sider": "EL",    // ✓ 유지
  "tox21": "EL"     // 유지 후 성능 벤치마크
}
```

### 옵션 2: 적극적 개선 (권장) ⭐
```json
{
  "bbbp": "ALC",    // ✓ 유지 (화학 구조 특성)
  "bace": "ALC",    // ★ 변경 (약물-표적 결합/비결합 구분)
  "clintox": "ALC", // ★ 변경 (임상 안전성 + 화학 위험도)
  "hiv": "ALC",     // ✓ 유지 (질병 + 실험 결합)
  "sider": "EL",    // ✓ 유지 (MeSH 계층 충분)
  "tox21": "ALC"    // ★ 변경 (3개 ontology 복합 예측)
}
```

---

## 성능 평가 방안

### 1단계: 베이스라인 (현재 설정)
```bash
python experiments/verify_semantic_forest_multi.py \
  --n-estimators 20 \
  --all-tasks \
  --out output/benchmark_baseline_EL.csv
```

### 2단계: ALC 확대 설정
```bash
# dl_profile_config.json을 옵션 2로 변경 후
python experiments/verify_semantic_forest_multi.py \
  --n-estimators 20 \
  --all-tasks \
  --out output/benchmark_proposed_ALC.csv
```

### 3단계: 비교 분석
```bash
# 각 데이터셋별 AUC/ACC 개선도 확인
# BACE: EL vs ALC 비교 (Target specificity 개선도)
# ClinTox: EL vs ALC 비교 (Safety/Toxicity 구분도)
# Tox21: EL vs ALC 비교 (Assay 특이성)
```

---

## 결론

### 현재 설정의 문제점

1. **BACE, ClinTox, Tox21**: 복합 ontology 또는 구조적 복잡도에 비해 EL이 과도하게 단순
2. **온톨로지 풍부도와 DL 불일치**: 
   - BBBP (1개 ontology) + ALC ✓
   - Tox21 (3개 ontology) + EL ✗ (역전)

### 추천 액션

**옵션 2 (적극적 개선)를 우선 시도하되, 베이스라인과 비교**

```python
# 수정 위치: experiments/dl_profile_config.json
{
  "bace": "ALC",      // EL → ALC
  "bbbp": "ALC",      // 유지
  "clintox": "ALC",   // EL → ALC
  "hiv": "ALC",       // 유지
  "sider": "EL",      // 유지
  "tox21": "ALC"      // EL → ALC
}
```

**성능이 향상되지 않으면**: 원래 설정으로 롤백하고 다른 하이퍼파라미터 조정 검토

