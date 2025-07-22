# Audio Embedding Search with CLAP & WavLM (FAISS 기반)

이 프로젝트는 오디오 임베딩 모델 **CLAP** 또는 **WavLM**을 사용해 오디오를 벡터로 임베딩하고,
벡터 검색 라이브러리 **FAISS**를 통해 유사한 오디오 조각을 검색하는 로컬 기반 파이프라인입니다.

---

## 프로젝트 구조

```
audio-embedding-search/
├── main.py                  # 실행 엔트리포인트
├── requirements.txt         # 필요한 패키지 목록
├── audio_samples/           # 임베딩할 원본 오디오 파일
├── segments/                # 검색용 오디오 세그먼트 (.wav)
├── queries/                 # 저장된 쿼리 waveform (.pt)
├── faiss_index/             # FAISS 인덱스 및 메타데이터
└── src/
    ├── core/
    │   └── faiss_ops.py     # 임베딩 및 검색 처리
    └── utils/
        ├── audio_utils.py   # 오디오 분할 및 리샘플링
        ├── model_clap.py    # CLAP 모델 로딩 및 임베딩
        └── model_wavlm.py   # WavLM 모델 로딩 및 임베딩
```

---

## 설치

```bash
pip install -r requirements.txt
```
> `ffmpeg` 설치 필요 (mp3 지원 시)

---

## 사용법

### 1️. 오디오 임베딩 → FAISS 인덱스 저장
```bash
python main.py --model clap --task embed --audio_dir ./audio_samples
```

### 2️. 검색용 오디오 쪼개기 (3초 단위)
```bash
python main.py --model clap --task split --audio_path ./audio_samples/example.wav
```

### 3️. 검색 수행 (직접 검색 + 옵션으로 쿼리 저장도 가능)
```bash
# 검색만
python main.py --model clap --task search --query_path ./segments/example_003_00m03s.wav

# 검색 + 쿼리 waveform 저장 (.pt)
python main.py --model clap --task search --query_path ./segments/example.wav --save_query
```

---

## 검색 결과 예시
```
검색 결과
   1. 🎵 song1 @ 00:15  (sim=0.8913)
   2. 🎵 song2 @ 00:45  (sim=0.8741)
   3. 🎵 song3 @ 01:00  (sim=0.8502)
```

---

## 모델 정보

| 모델 | 경로 | 특징 |
|------|------|------|
| CLAP | `laion/clap-htsat-unfused` | 음악/오디오 이해에 특화된 모델 |
| WavLM | `microsoft/wavlm-base-plus` | 음성 임베딩에 강한 음성 모델 |

---

## 기타 정보

- 벡터 검색은 `FAISS.IndexFlatIP` + cosine 유사도 기반
- 인덱스와 메타데이터는 `faiss_index/index_<model>.pkl`로 저장됨
- 검색 쿼리는 `.pt` 파일로 저장 시 재사용 가능 (`--save_query`)

---

## 테스트 준비 팁

- `audio_samples/`에 mp3 또는 wav 오디오 파일을 넣고 테스트하세요
- `segments/` 폴더는 자동 생성됩니다 (없으면 만들어짐)
- 검색용 쿼리는 `.wav` 또는 `.pt` 형태 모두 지원됩니다

---

## 기여 및 확장

- Gradio UI 버전 확장 가능
- CLAP/WavLM 외 다른 임베딩 모델 추가도 용이

---

> Maintained by Anyfive. PR & Issues welcome!
