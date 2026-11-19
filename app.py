"""5단계: 대회 제출용 API 서버.

실행:  uvicorn app:app --host 0.0.0.0 --port 8000
호출:  POST /ask  {"question": "장애인 주차요금 감면되나요?"}

채점은 운영진이 마감 직후 1회 실행합니다. 문항당 90초를 넘기거나 오류가 나면 0점입니다.
그래서 어떤 경우에도 정해진 JSON을 돌려주도록 감쌌습니다.

요청 로그: /ask·/api/ask 요청마다 logs/requests.jsonl에 한 줄(JSON)씩 남깁니다.
  받은 시각, 보낸 쪽 IP, 경로, 질문 원문, 답·인용·처리 경로(trace), 걸린 시간, 오류 내용.
  로그 쓰기가 실패해도 응답에는 영향을 주지 않습니다.
"""
import asyncio
import json
import logging
import threading
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

import config
import rag
import viewer

app = FastAPI(title="서울시 조례 Q&A")
log = logging.getLogger("uvicorn.error")
INDEX_HTML = config.ROOT / "static" / "index.html"
REQUEST_LOG = config.ROOT / "logs" / "requests.jsonl"
_log_lock = threading.Lock()


def _client_ip(request: Request):
    """보낸 쪽 IP. 프록시·포트포워딩 장비가 X-Forwarded-For를 붙였으면 그 값도 함께 남깁니다."""
    ip = request.client.host if request.client else None
    fwd = request.headers.get("x-forwarded-for")
    return f"{ip} (forwarded: {fwd})" if fwd else ip


def _write_log(row):
    try:
        REQUEST_LOG.parent.mkdir(exist_ok=True)
        line = json.dumps(row, ensure_ascii=False)
        with _log_lock, open(REQUEST_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:  # noqa: BLE001  로그 실패는 응답에 영향을 주지 않음
        log.warning("request log failed: %r", e)


async def _answer(question, request: Request, path):
    """답변 + 요청 로그. 시간 초과·오류여도 정해진 JSON('확인 불가')을 돌려줍니다."""
    received = datetime.now().isoformat(timespec="seconds")
    t0, trace, error = time.time(), [], None
    try:
        res = await asyncio.wait_for(asyncio.to_thread(rag.answer, question, trace),
                                     timeout=config.ANSWER_TIMEOUT_SEC)
    except Exception as e:  # noqa: BLE001  시간 초과·API 오류 모두 '확인 불가'로 응답
        error = repr(e)
        log.warning("%s failed: %r", path, e)
        res = dict(rag.UNKNOWN)
    _write_log({"received": received, "ip": _client_ip(request), "path": path, "question": question,
                "answer": res.get("answer"), "citation": res.get("citation"), "trace": "/".join(trace),
                "sec": round(time.time() - t0, 2), "error": error})
    return res


@app.exception_handler(RequestValidationError)
async def _bad_request(request: Request, exc):
    """질문이 빠진 요청 등 형식 오류도 로그에 남기고, 응답은 FastAPI 기본(422) 그대로 돌려줍니다."""
    if request.url.path in ("/ask", "/api/ask"):
        try:
            body = (await request.body()).decode("utf-8", "replace")[:500]
        except Exception:  # noqa: BLE001
            body = None
        _write_log({"received": datetime.now().isoformat(timespec="seconds"), "ip": _client_ip(request),
                    "path": request.url.path, "question": None, "body": body, "answer": None, "citation": None,
                    "trace": "", "sec": 0, "error": f"요청 형식 오류: {exc.errors()}"})
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc):
    """본문을 해석하지 못한 요청(400) 등 HTTP 오류도 /ask·/api/ask면 로그에 남깁니다. 응답은 FastAPI 기본 그대로."""
    if request.url.path in ("/ask", "/api/ask"):
        _write_log({"received": datetime.now().isoformat(timespec="seconds"), "ip": _client_ip(request),
                    "path": request.url.path, "question": None, "answer": None, "citation": None, "trace": "",
                    "sec": 0, "error": f"HTTP {exc.status_code}: {exc.detail}"})
    return await http_exception_handler(request, exc)


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
async def ask(req: Ask, request: Request):
    return await _answer(req.question, request, "/ask")


@app.post("/api/ask", include_in_schema=False)
async def ask_web(req: Ask, request: Request):
    """웹 화면용. 지금은 제출용 /ask와 똑같이 답합니다 (시연용 웹은 따로 구현 예정).
    시간 환산 답(예: "10분 1,000원")이 필요하면 rag.answer(req.question, None, True)로 부르면 됩니다."""
    return await _answer(req.question, request, "/api/ask")
