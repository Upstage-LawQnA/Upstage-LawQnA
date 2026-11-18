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
- `POST /api/ask` — 웹 화면용. 지금은 제출용과 똑같이 답합니다.
- `GET /` — 시민용 화면, `GET /test` — 제출용 응답 확인 페이지, `GET /api/article?label=...` — 인용 조항 원문, `GET /health`

근거를 찾지 못하면 `{"answer": "확인 불가", "citation": [], ...}`를 돌려줍니다. 서버는 75초 안에 답하지 못하면 확인 불가로 응답합니다.

## 처리 순서

1. 범위 확인: 6개 조례 밖 다른 조례 용어, 서울 25개 자치구의 요금·조례 질문 → 확인 불가
2. 검색: 질문 재작성(solar-pro4) + 임베딩 0.9 / 글자 2개 단위 BM25 0.1, 같은 조의 다른 항 함께. 호가 많은 항에는 질문과 관련된 호 표시
3. 답변(solar-pro4, JSON): 답·근거 라벨·대상별 주장(근거 조항·호·면제/감면 등)
4. 주장 검증(코드): 면제·감면 대상을 목록에서 잘못 읽었으면 확인된 사실을 주고 한 번 다시 씀
5. 가드: 규정 없음·금액 없음·근거 없는 중복 제한 규칙 등은 확인 불가로
6. 답변 검증(solar-pro4): 근거에 기반하는지, 질문에 직접 답하는지
7. 편집(solar-pro4): 질문이 묻지 않은 문장과 그 인용 빼기
8. 후처리: 네/아니요 정리, 조항 번호·권고 문장 빼기, 다른 조례 인용 정리

보통 LLM 4번(재작성·답변·검증·편집) + 임베딩 1번입니다. 1차가 확인 불가면 추론 모드로 한 번 더 시도합니다.

## 구성

| 파일 | 내용 |
|---|---|
| `rag.py` | 검색 → 답변 → 주장 검증 → 가드 → 검증 → 편집 → 후처리 |
| `upstage.py` | Upstage API (solar-pro4 답변·검증·편집, solar-embedding-2) |
| `app.py` | FastAPI 서버 |
| `config.py` | 설정 (모델, 대상 조례, 검색 개수, 시간 제한) |
| `viewer.py`, `parse_annex.py` | 인용 조항·별표 원문 보기 |
| `static/index.html` | 시민용 화면 |
| `static/submit_test.html` | 제출용 `/ask` 응답 확인 페이지 |
| `data/` | 조례 원문, 검색 조각, 임베딩 벡터, 별표, 범위 밖 용어 목록 |

## 비교용 스위치 (환경변수, 기본값이면 위 순서대로 동작)

| 변수 | 기본 | 내용 |
|---|---|---|
| `RAG_CLAIMS` | `1` | 주장 단위 검증 (`0`이면 끔) |
| `RAG_QTYPE` | `regex` | 질문 유형 판정 방식 (`llm` / `mixed`) |
| `RAG_TOPIC` | `0` | 특정 조항·주제에 묶인 예전 필터 (`1`이면 켬) |
| `RAG_LEAN` | (없음) | `guard,post,prompt,trim,tail,items,strict` 중 일부를 끔 |
