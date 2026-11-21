"""시연용 웹 서버 (민원을 잇다). 제출용 서버(starter-kit/app.py, 8000번)와 별개입니다.

실행:  uvicorn app:app --app-dir demo-web --port 8001
화면:  http://localhost:8001/

API
  GET  /api/regions                 지역 목록 (시·도)
  GET  /api/recent?region=…         최근 시행된 자치법규
  POST /api/ask/stream              질문 처리 (단계별 NDJSON: triage → laws → answer)
  POST /api/compare                 같은 질문을 여러 지역에서 비교
  GET/POST /api/tickets, PATCH /api/tickets/{no}   담당자 문의(민원) 접수·조회·처리
요청 로그: logs/demo_requests.jsonl (개인정보는 가린 질문만 남김)
"""
import json
import logging
import threading
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import config
import corpus
import service

app = FastAPI(title="민원을 잇다 (시연용)")
log = logging.getLogger("uvicorn.error")
STATIC = config.ROOT / "static"
LOG = config.ROOT / "logs" / "demo_requests.jsonl"
_log_lock = threading.Lock()


def _write_log(row):
    try:
        LOG.parent.mkdir(exist_ok=True)
        with _log_lock, open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        log.warning("demo log failed: %r", e)


@app.on_event("startup")
def _startup():
    corpus.init()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html", media_type="text/html; charset=utf-8")


@app.get("/region_map.json", include_in_schema=False)
def region_map():
    """지역 비교의 '지도로 고르기'용 시·도 경계 (통계청 SGIS 행정동 경계 기반, CC BY 4.0)"""
    return FileResponse(STATIC / "region_map.json", media_type="application/json")


@app.get("/health")
def health():
    return {"ok": True, "laws_loaded": len(corpus.LAWS)}


@app.get("/api/regions")
def regions():
    return corpus.region_list()


@app.get("/api/recent")
def recent(region: str = "서울특별시", n: int = 6):
    try:
        rows = corpus.recent(region, n)
    except Exception as e:  # noqa: BLE001  법제처 API 장애여도 화면은 뜨게
        log.warning("recent failed: %r", e)
        rows = []
    return [{**r, "tag": corpus.tag_of(r["name"])} for r in rows]


@app.get("/api/law")
def law(mst: str):
    """조례 상세 화면: 기본 정보 · 조항 목록 · 제·개정 이유"""
    if not mst.isdigit():
        raise HTTPException(400, "잘못된 조례 번호예요")
    try:
        return corpus.law_detail(mst)
    except Exception as e:  # noqa: BLE001  법제처 API 장애
        log.warning("law detail failed: %r", e)
        raise HTTPException(502, "조례 정보를 불러오지 못했어요")


class AskReq(BaseModel):
    question: str
    region: str = "서울특별시"
    law: str | None = None


@app.post("/api/ask/stream")
def ask_stream(req: AskReq):
    if not req.question.strip():
        raise HTTPException(400, "질문을 입력해 주세요")

    def gen():
        final = None
        try:
            for ev in service.ask_stream(req.question, req.region, req.law):
                if ev["stage"] == "answer":
                    final = ev
                    if ev["type"] == "review":
                        # '담당자 확인 필요' 답변은 공무원 민원함에 근거와 함께 자동 등록 (시민이 버튼을 누르지 않아도)
                        try:
                            t = service.create_ticket({"question": ev["question"], "region": req.region,
                                                       "reason": "자동: 담당자 확인 필요", "result": ev})
                            ev["auto_ticket"] = t["no"]
                        except Exception:  # noqa: BLE001  등록 실패해도 답변은 보여 줌
                            log.exception("auto ticket failed")
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:  # noqa: BLE001
            log.exception("ask failed")
            yield json.dumps({"stage": "error", "message": f"처리 중 오류가 났어요 ({type(e).__name__})"},
                             ensure_ascii=False) + "\n"
        if final:
            _write_log({"at": datetime.now().isoformat(timespec="seconds"), "region": req.region, "law": req.law,
                        "question": final["question"], "type": final["type"], "answer": final["answer"],
                        "citation": final["citation"], "sec": final["sec"], "trace": final["trace"]})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


class CompareReq(BaseModel):
    question: str
    regions: list[str]


@app.post("/api/compare")
def compare(req: CompareReq):
    regions = [r for r in dict.fromkeys(req.regions) if r][:4]
    if not req.question.strip() or len(regions) < 2:
        raise HTTPException(400, "질문과 2개 이상의 지역을 골라 주세요")
    out = service.compare(req.question, regions)
    _write_log({"at": datetime.now().isoformat(timespec="seconds"), "compare": regions, "question": out["question"],
                "found": [r["region"] for r in out["rows"] if r["found"]]})
    return out


class TicketReq(BaseModel):
    question: str
    region: str = "서울특별시"
    title: str | None = None
    message: str = ""
    name: str = ""
    reason: str = ""
    result: dict | None = None


@app.post("/api/tickets")
def create_ticket(req: TicketReq):
    if not req.question.strip():
        raise HTTPException(400, "문의 내용을 입력해 주세요")
    return service.create_ticket(req.model_dump())


@app.get("/api/tickets")
def tickets(nos: str = ""):
    return service.list_tickets([n for n in nos.split(",") if n] or None)


class TicketUpdate(BaseModel):
    status: str | None = None
    reply: str | None = None


@app.patch("/api/tickets/{no}")
def update_ticket(no: str, req: TicketUpdate):
    t = service.update_ticket(no, {k: v for k, v in req.model_dump().items() if v is not None})
    if not t:
        raise HTTPException(404, "민원을 찾지 못했습니다")
    return t
