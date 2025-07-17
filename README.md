# 실시간 공연 자막/해설 시스템 데모
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

- AudioEmbedding
  - ...

---

## 데모 사용 방법

```bash
git clone https://github.com/your-username/ai-music-fingerprint-demo.git
cd ai-music-fingerprint-demo
pip install -r requirements.txt
