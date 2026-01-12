@echo off
echo ========================================
echo  BTC Strategy Miner - GitHub 배포 도우미
echo ========================================
echo.

REM Git 설치 확인
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [오류] Git이 설치되어 있지 않습니다.
    echo Git 다운로드: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [1/5] Git 저장소 초기화...
git init
if %ERRORLEVEL% NEQ 0 (
    echo Git이 이미 초기화되어 있습니다.
)

echo.
echo [2/5] GitHub 저장소 URL을 입력하세요.
echo 예시: https://github.com/YOUR_USERNAME/btc-strategy-miner.git
echo.
set /p REPO_URL="GitHub 저장소 URL: "

if "%REPO_URL%"=="" (
    echo [오류] URL이 입력되지 않았습니다.
    pause
    exit /b 1
)

echo.
echo [3/5] 원격 저장소 연결...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo.
echo [4/5] 파일 추가 및 커밋...
git add .
git commit -m "Deploy: BTC Strategy Miner V3"

echo.
echo [5/5] GitHub에 업로드...
git branch -M main
git push -u origin main

echo.
echo ========================================
echo  업로드 완료!
echo ========================================
echo.
echo 다음 단계:
echo 1. https://streamlit.io/cloud 접속
echo 2. "New app" 클릭
echo 3. GitHub 저장소 선택
echo 4. Main file: app.py
echo 5. Deploy 클릭!
echo.
pause
