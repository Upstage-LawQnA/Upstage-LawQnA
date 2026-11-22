# 실행 (시연용)

```bash
cd demo-web
pip install -r requirements.txt
cp .env.example .env   # UPSTAGE_API_KEY, LAW_OC 입력
python -X utf8 -m uvicorn app:app --host 127.0.0.1 --port 8001
```

http://127.0.0.1:8001
