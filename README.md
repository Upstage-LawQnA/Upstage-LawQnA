# 서울 조례 길잡이 (서울특별시 조례 Q&A)

서울특별시 조례 6종(주차장 설치 및 관리, 동물보호, 자전거이용 활성화, 도시공원, 청년 기본, 금연환경 조성 및 간접흡연 피해방지)에서
근거 조항을 찾아 쉬운 말로 답하는 RAG 서버입니다. 2026년 10월 1일 시행본 기준이며, 자치구 조례·규칙은 다루지 않습니다.

## 실행

```bash
pip install -r requirements.txt
cp .env.example .env        # .env에 UPSTAGE_API_KEY를 넣습니다 (.env는 올리지 마세요)
python -X utf8 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

- `POST /ask` — 제출용. `{"question": "..."}` → `{"answer", "citation", "source_url", "effective_date"}`
- `POST /api/ask` — 웹 화면용. 제출용과 같고, "10분당 요금"처럼 요금표 단위의 배수를 물으면 환산한 금액으로 답합니다.
- `GET /` — 시민용 화면, `GET /api/article?label=...` — 인용 조항 원문, `GET /health`

근거를 찾지 못하면 `{"answer": "확인 불가", "citation": [], ...}`를 돌려줍니다.

## 구성

| 파일 | 내용 |
|---|---|
| `rag.py` | 검색(임베딩 0.9 + BM25 0.1, 질문 재작성) → solar-pro4 답변·근거 선택 → 검증 → 후처리 |
| `upstage.py` | Upstage API (solar-pro4, solar-embedding-2, 답변 검증) |
| `app.py` | FastAPI 서버 |
| `config.py` | 설정 (모델, 대상 조례, 검색 개수) |
| `viewer.py`, `parse_annex.py` | 인용 조항·별표 원문 보기 |
| `static/index.html` | 시민용 화면 |
| `data/` | 조례 원문, 검색 조각, 임베딩 벡터, 별표, 범위 밖 용어 목록 |
