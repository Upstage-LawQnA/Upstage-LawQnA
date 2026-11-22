# 실행 (채점용 API)

```bash
pip install -r requirements.txt
cp .env.example .env   # UPSTAGE_API_KEY, LAW_OC 입력
python -X utf8 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

`POST http://localhost:8000/ask`
