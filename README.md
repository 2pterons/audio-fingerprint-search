# Audio Search with fingerprint
**AI Champion 프로그램 기술 데모**

이 프로젝트는 판소리, 뮤지컬 등 공연 중 실시간 오디오를 분석하여  
해당 곡이나 장면을 식별하고 관람자에게 **다국어 자막 및 해설**을 제공하기 위한  
기술의 핵심 구성 요소를 시연합니다.

---

## 주요 기능

- FingerPrint
  - 오디오를 초 단위로 세분화하여 분석
  - 각 세그먼트에서 **고유한 fingerprint**를 생성
  - 미리 구축한 DB와 비교하여 **실시간으로 곡 또는 위치를 식별**
  - 향후 자막, 번역, 해설 연동의 기반이 되는 검색 엔진 제공

---

## 설치 방법
```bash
git clone https://github.com/2pterons/audio-fingerprint-search.git
cd audio-fingerprint-search
pip install -r requirements.txt
```
※ ffmpeg가 설치되어 있어야 pydub이 mp3 파일을 읽을 수 있습니다.  
Ubuntu 예: sudo apt install ffmpeg
  
## 데모 사용 방법
1. DB 생성
```bash
python run_demo.py build-db
```

2. 쿼리 오디오 자르기 (예: 80~90초 잘라서 query_segment.wav 저장)
```bash
python run_demo.py cut-multi
```
or
```bash
python run_demo.py cut-query \
  --file_name I_fall_in_love_too_easily.mp3 \
  --start_time 80 \
  --duration 10
```

3. 쿼리 오디오로 검색
```bash
python run_demo.py query \
   --query [파일명].wav
```

