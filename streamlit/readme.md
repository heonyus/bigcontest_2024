0. 디렉토리 이동
```bash
cd streamlit
```

1. 가상환경 venv 생성
```bash
python -m venv venv
```

2. 가상환경 활성화
```bash
source venv/bin/activate # 윈도우는 venv\Scripts\activate
```

3. 필요 패키지 설치
```bash
# PyTorch 먼저 설치 (CUDA 지원 버전)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 나머지 패키지 설치
pip install -r requirements.txt
```

4. .streamlit/secrets.toml 파일 생성
```bash
touch .streamlit/secrets.toml
```
`secrets.toml` 파일은 재헌에게 문의.

5. data/JEJU_with_embeddings_final.csv 파일 생성
`JEJU_with_embeddings_final.csv` 파일은 재헌에게 문의.

6. streamlit 실행
```bash
streamlit run app.py
```
