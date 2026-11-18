"""5단계: 대회 제출용 API 서버.

실행:  uvicorn app:app --host 0.0.0.0 --port 8000
호출:  POST /ask  {"question": "장애인 주차요금 감면되나요?"}

채점은 운영진이 마감 직후 1회 실행합니다. 문항당 90초를 넘기거나 오류가 나면 0점입니다.
그래서 어떤 경우에도 정해진 JSON을 돌려주도록 감쌌습니다.
"""
import asyncio
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
import rag
import viewer

app = FastAPI(title="서울시 조례 Q&A")
log = logging.getLogger("uvicorn.error")
INDEX_HTML = config.ROOT / "static" / "index.html"


@app.get("/", include_in_schema=False)
def home():
    """시민용 화면. 채점용 POST /ask와는 별개라 채점에 영향을 주지 않습니다."""
    return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")


@app.get("/test", include_in_schema=False)
def submit_test():
    """제출용 POST /ask 테스트 페이지 (운영진이 받는 JSON 그대로 + 형식·인용 점수 점검). 시연 화면과 별개."""
    return FileResponse(config.ROOT / "static" / "submit_test.html", media_type="text/html; charset=utf-8")


@app.get("/api/article")
def article(label: str):
    """화면에서 인용 조항을 눌렀을 때 보여 줄 원문 (조 전체 + 인용된 항 표시, 또는 별표 표)."""
    found = viewer.lookup(label)
    if not found:
        raise HTTPException(404, "조항을 찾지 못했습니다")
    return found


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True, "chunk_unit": config.CHUNK_UNIT, "verify": config.USE_VERIFY}


@app.post("/ask")
async def ask(req: Ask):
    try:
        return await asyncio.wait_for(asyncio.to_thread(rag.answer, req.question),
                                      timeout=config.ANSWER_TIMEOUT_SEC)
    except Exception as e:  # noqa: BLE001  시간 초과·API 오류 모두 '확인 불가'로 응답
        log.warning("ask failed: %r", e)
        return dict(rag.UNKNOWN)


@app.post("/api/ask", include_in_schema=False)
async def ask_web(req: Ask):
    """웹 화면용. 지금은 제출용 /ask와 똑같이 답합니다 (시연용 웹은 따로 구현 예정).
    시간 환산 답(예: "10분 1,000원")이 필요하면 rag.answer(req.question, None, True)로 부르면 됩니다."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(rag.answer, req.question),
                                      timeout=config.ANSWER_TIMEOUT_SEC)
    except Exception as e:  # noqa: BLE001
        log.warning("ask_web failed: %r", e)
        return dict(rag.UNKNOWN)
