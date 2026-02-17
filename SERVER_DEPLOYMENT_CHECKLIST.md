# 서버 배포 체크리스트

## 1. 필수 변경 파일 (코드 변경)

### A. `experiments/verify_semantic_forest_multi.py` ⭐ 필수
**변경 사항**:
- `import json` 추가
- `_dataset_dl_profile()` 함수 추가 (DL 프로필 로드)
- `evaluate_task()` 함수에 `dl_profile` 파라미터 추가
- `learner_kwargs`에 `dl_profile` 전달
- main()에서 각 데이터셋별 DL 프로필 로드 로직 추가

**적용 방법**:
```bash
git pull origin feature/dataset-ontology-mapping
# 또는 특정 파일만
git checkout origin/feature/dataset-ontology-mapping -- experiments/verify_semantic_forest_multi.py
```

**확인 명령**:
```bash
python -m py_compile experiments/verify_semantic_forest_multi.py
```

---

### B. `experiments/dl_profile_config.json` ⭐ 필수
**현재 서버의 내용** (변경 전):
```json
{
  "bace": "EL",
  "bbbp": "ALC",
  "clintox": "EL",
  "hiv": "ALC",
  "sider": "EL",
  "tox21": "EL",
  ...
}
```

**변경할 내용** (옵션 2 적용):
```json
{
  "bace": "ALC",        // EL → ALC
  "bbbp": "ALC",        // 유지
  "clintox": "ALC",     // EL → ALC
  "hiv": "ALC",         // 유지
  "sider": "EL",        // 유지
  "tox21": "ALC",       // EL → ALC
  ...
}
```

**적용 방법**:
```bash
git pull origin feature/dataset-ontology-mapping
# 또는
git checkout origin/feature/dataset-ontology-mapping -- experiments/dl_profile_config.json
```

**확인**:
```bash
cat experiments/dl_profile_config.json | grep -E '"(bace|clintox|tox21)"'
# 출력:
#   "bace": "ALC",
#   "clintox": "ALC",
#   "tox21": "ALC",
```

---

## 2. 의존성 파일 (이미 적용됨)

### C. `requirements.txt` (이미 main에 있음)
- 이미 Python 3.8 호환 버전으로 고정됨
- 서버의 환경과 일치하는지 확인만 필요

```bash
pip show numpy pandas
# numpy==1.24.4, pandas==2.0.3 확인
```

---

### D. `src/ontology/molecule_ontology.py` (이미 main에 있음)
- `base_ontology_paths` 파라미터 지원 추가
- legacy fallback 포함
- 이미 서버에 배포됨

---

### E. `src/utils/compute_backend.py` (이미 main에 있음)
- GPU/CPU 자동 선택 로직
- 이미 서버에 배포됨

---

## 3. 온톨로지 파일 (존재 확인만)

### F. `ontology/` 디렉토리 구조
서버에 다음 파일들이 있는지 확인:
```
ontology/
├── DTO.owl (또는 DTO.xrdf, DTO.xml)
├── chebi.owl              # BBBP, ClinTox
├── Thesaurus.owl          # HIV
├── bao_complete.owl       # HIV, Tox21
├── pato.owl               # Tox21
├── go.owl                 # Tox21
└── mesh.owl               # SIDER
```

없으면 다운로드 필요 ([README의 온톨로지 다운로드 가이드](README.md) 참조)

---

## 4. 서버 적용 순서

### Step 1: 코드 업데이트
```bash
cd /path/to/Semantic-Forest-Lab
git fetch origin
git checkout feature/dataset-ontology-mapping
# 또는
git pull origin feature/dataset-ontology-mapping
```

### Step 2: 파일 검증
```bash
# Python 문법 검사
python -m py_compile experiments/verify_semantic_forest_multi.py

# DL 프로필 설정 확인
python -c "import json; print(json.load(open('experiments/dl_profile_config.json')))"
```

### Step 3: 온톨로지 파일 확인
```bash
ls -lh ontology/*.owl
# 또는 (용량 확인)
du -sh ontology/
```

### Step 4: 테스트 실행 (1-2개 데이터셋만)
```bash
python experiments/verify_semantic_forest_multi.py \
  --datasets bbbp,bace \
  --n-estimators 5 \
  --max-depth 5 \
  --out output/test_dl_enhanced.csv
```

**예상 출력**:
```
Loaded DL profile for BBBP: ALC
Loaded DL profile for BACE: ALC
...
```

### Step 5: 전체 벤치마크 실행
```bash
python experiments/verify_semantic_forest_multi.py \
  --n-estimators 20 \
  --all-tasks \
  --out output/benchmark_dl_enhanced.csv
```

---

## 5. 빠른 체크리스트

서버에서 실행하기 전 확인:

- [ ] `git pull origin feature/dataset-ontology-mapping` 완료
- [ ] `experiments/verify_semantic_forest_multi.py` 업데이트 확인
- [ ] `experiments/dl_profile_config.json`에서:
  - `"bace": "ALC"` ✓
  - `"clintox": "ALC"` ✓
  - `"tox21": "ALC"` ✓
- [ ] `python -m py_compile experiments/verify_semantic_forest_multi.py` 성공
- [ ] `ontology/*.owl` 파일 존재 확인
- [ ] 소규모 테스트 실행 성공

---

## 6. 트러블슈팅

### "No module named 'src.ontology'"
→ `verify_semantic_forest_multi.py`가 업데이트되지 않음. git pull 재실행

### "KeyError: 'bbbp' in dl_profile_config.json"
→ `dl_profile_config.json`이 옛날 버전. 파일 다시 확인

### "FileNotFoundError: ontology/chebi.owl"
→ 온톨로지 파일 없음. README의 다운로드 가이드 따라 설치

### "CUDA device requested but not available"
→ `--torch-device cpu`로 명시적 지정

