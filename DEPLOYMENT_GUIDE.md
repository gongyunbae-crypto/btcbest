# 🌐 온라인 배포 가이드

이 앱을 온라인 서버에 배포하는 방법을 안내합니다.

## 🎯 추천 방법: Streamlit Community Cloud (무료)

### 1단계: GitHub 저장소 생성

1. [GitHub](https://github.com)에 로그인
2. 새 저장소(Repository) 생성
3. 저장소 이름 입력 (예: `btc-strategy-miner`)
4. Public으로 설정 (무료 배포를 위해)

### 2단계: 코드 업로드

```bash
# Git 초기화
cd "c:\Users\hashmusic\Antigravity\btc best"
git init

# 원격 저장소 연결 (YOUR_USERNAME을 본인 GitHub 아이디로 변경)
git remote add origin https://github.com/YOUR_USERNAME/btc-strategy-miner.git

# 파일 추가 및 커밋
git add .
git commit -m "Initial commit: BTC Strategy Miner V3"

# GitHub에 푸시
git branch -M main
git push -u origin main
```

### 3단계: Streamlit Community Cloud 배포

1. [Streamlit Community Cloud](https://streamlit.io/cloud) 접속
2. "New app" 클릭
3. GitHub 저장소 연결
4. 다음 정보 입력:
   - **Repository**: `YOUR_USERNAME/btc-strategy-miner`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. "Deploy!" 클릭

### 4단계: 배포 완료! 🎉

몇 분 후 앱이 온라인에 배포됩니다.
URL 형식: `https://YOUR_USERNAME-btc-strategy-miner.streamlit.app`

---

## 🔐 API 키 보안 설정 (선택사항)

실전 매매를 위한 Binance API 키를 안전하게 저장하려면:

1. Streamlit Cloud 대시보드에서 앱 선택
2. "Settings" → "Secrets" 클릭
3. 다음 형식으로 입력:

```toml
BINANCE_API_KEY = "your_api_key_here"
BINANCE_API_SECRET = "your_api_secret_here"
```

4. `app.py`에서 다음과 같이 사용:

```python
import streamlit as st

api_key = st.secrets.get("BINANCE_API_KEY", "")
api_secret = st.secrets.get("BINANCE_API_SECRET", "")
```

---

## 🚀 대안 배포 옵션

### Option 2: Heroku (유료)

- 더 많은 리소스 필요 시
- 월 $7부터 시작
- [Heroku 가이드](https://devcenter.heroku.com/articles/getting-started-with-python)

### Option 3: AWS EC2 (고급)

- 완전한 제어 필요 시
- 프리 티어 1년 무료
- 설정이 복잡함

### Option 4: Google Cloud Run (중급)

- 사용량 기반 과금
- 자동 스케일링
- Docker 지식 필요

---

## ⚠️ 중요 참고사항

### 데이터 파일 처리

- `btc_futures_data_5m.csv` (6.7MB)는 Git에 포함되지 않음
- 앱 첫 실행 시 자동으로 Binance에서 다운로드됨
- 초기 로딩에 1-2분 소요 가능

### 성능 제한

- Streamlit Community Cloud 무료 플랜:
  - 1GB RAM
  - 1 CPU 코어
  - 동시 사용자 제한 있음
- 대량 트래픽 예상 시 유료 플랜 고려

### 지속적 실행

- Streamlit Cloud는 비활성 시 슬립 모드 진입
- 24/7 실전 매매봇으로는 부적합
- 실전 매매는 VPS나 전용 서버 권장

---

## 🆘 문제 해결

### 배포 실패 시

1. `requirements.txt` 확인
2. Python 버전 호환성 확인 (3.8-3.11)
3. Streamlit Cloud 로그 확인

### 앱이 느릴 때

1. 데이터 캐싱 확인 (`@st.cache_data`)
2. 불필요한 재계산 제거
3. 유료 플랜으로 업그레이드 고려

---

## 📞 도움이 필요하신가요?

- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [Streamlit 커뮤니티 포럼](https://discuss.streamlit.io/)
- [GitHub Issues](https://github.com/YOUR_USERNAME/btc-strategy-miner/issues)
