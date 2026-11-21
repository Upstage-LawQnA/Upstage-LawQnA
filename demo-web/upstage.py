"""Upstage API 호출을 한곳에 모았습니다. (Chat · Embedding · Document Parse · Groundedness Check)

Solar는 OpenAI SDK와 호환됩니다. base_url만 https://api.upstage.ai/v1 로 바꾸면 됩니다.
"""
import json
import random
import re
import threading
import time

import requests

import config

_client = None
# 토큰 사용량 누적 (측정용). 질문 하나가 Solar를 여러 번 부르므로 얼마나 쓰는지 바로 보이게 합니다.
USAGE = {"chat_calls": 0, "in": 0, "out": 0, "reason": 0, "embed_calls": 0, "embed_in": 0}
_usage_lock = threading.Lock()


def _count(resp, embed=False):
    u = getattr(resp, "usage", None)
    if u is None:
        return
    with _usage_lock:
        if embed:
            USAGE["embed_calls"] += 1
            USAGE["embed_in"] += u.prompt_tokens or 0
            return
        USAGE["chat_calls"] += 1
        USAGE["in"] += u.prompt_tokens or 0
        USAGE["out"] += getattr(u, "completion_tokens", 0) or 0
        USAGE["reason"] += getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0


def client():
    global _client
    if _client is None:
        if not config.UPSTAGE_API_KEY:
            # SystemExit이 아니라 일반 예외로: 서버(app.py)가 잡아서 '확인 불가'로 응답할 수 있게
            raise RuntimeError("UPSTAGE_API_KEY가 없습니다. .env.example을 .env로 복사하고 키를 넣어 주세요.")
        from openai import OpenAI
        _client = OpenAI(api_key=config.UPSTAGE_API_KEY, base_url=config.UPSTAGE_BASE_URL)
    return _client


def _retry(fn, tries=6):
    """429(too_many_requests)는 잠시 기다렸다 다시 부릅니다. 한도는 Organization 단위로 공유됩니다.

    동시 요청이 몰리면(채점 18건 동시 실행에서 확인) 4번·7초로는 모자라서 6번·최대 약 23초로 늘렸습니다.
    여러 요청이 같은 순간에 다시 몰리지 않도록 기다리는 시간에 무작위 값을 섞습니다.
    """
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) and i < tries - 1:
                time.sleep(min(2 ** i, 8) * (0.5 + random.random()))
                continue
            raise


def chat(messages, response_format=None, temperature=0.0, reasoning_effort=None, timeout=None):
    """timeout(초)을 주면 그 시간 안에 끝나지 않을 때 SDK 자동 재시도 없이 바로 예외를 냅니다."""
    kw = dict(model=config.CHAT_MODEL, messages=messages, temperature=temperature)
    if response_format:
        kw["response_format"] = response_format
    if reasoning_effort:                      # solar-pro4: 생략하면 추론 OFF, "low" 이상이면 ON
        kw["reasoning_effort"] = reasoning_effort
    c = client().with_options(timeout=timeout, max_retries=0) if timeout else client()
    resp = _retry(lambda: c.chat.completions.create(**kw))
    _count(resp)
    return resp.choices[0].message.content


def embed(texts, model):
    out = []
    for i in range(0, len(texts), 50):        # 한 번에 너무 많이 보내지 않습니다
        batch = texts[i:i + 50]
        resp = _retry(lambda: client().embeddings.create(model=model, input=batch))
        _count(resp, embed=True)
        out.extend(d.embedding for d in resp.data)
    return out


def document_parse(path):
    """Document Parse: 문서를 구조가 살아 있는 HTML/Markdown으로 바꿉니다. 표가 표로 남습니다."""
    url = f"{config.UPSTAGE_BASE_URL}/document-digitization"
    headers = {"Authorization": f"Bearer {config.UPSTAGE_API_KEY}"}
    with open(path, "rb") as f:
        resp = requests.post(url, headers=headers, timeout=300,
                             files={"document": f},
                             data={"model": "document-parse",
                                   "output_formats": json.dumps(["html", "markdown"])})
    resp.raise_for_status()
    content = resp.json().get("content", {})
    return content.get("markdown") or content.get("html") or content.get("text") or ""


_GROUND_SYSTEM = """너는 자치법규(조례·규칙) 답변 검증기다. [질문]·[근거]·[답변]을 보고 세 가지를 판정한다.

1) verdict: 답변이 근거에 기반하는가
- grounded: 답변의 모든 사실(금액·비율·나이·조건·예외)이 근거에 그대로 있거나 근거에서 바로 도출된다.
- notGrounded: 답변에 근거에 없는 사실이 있거나, 근거와 어긋나는 내용이 있다.
  근거에 없는 권고·안내(예: "구청에 신고하세요", "~을 이용하는 것이 맞습니다")를 사실처럼 덧붙인 경우도 notGrounded.
  질문에 붙은 구체적 조건(예: "○○ 없이 △△만 이용한 경우")을 근거가 다루지 않는데 답변이 "그 경우에도 된다/포함된다"고 단정한 경우도 notGrounded.
- notSure: 근거만으로는 판단하기 어렵다.

2) answers_question: 근거가 질문이 묻는 바로 그것에 직접 답하는가 (true/false)
- false인 경우:
  · 근거가 질문 대상과 다른 대상(다른 시설·다른 요금·다른 위반행위)에 관한 것인데 답변이 이를 끌어다 썼다.
  · 질문한 행위와 근거가 규제하는 행위가 다르다. (판매자·영업자에 대한 규정을 이용자·구매자의 행위에 그대로 적용했다면 false)
  · 질문이 근거에 없는 이름의 시설·제도(예: "○○ 전용 △△")를 물었는데, 답변이 다른 이름의 제도로 바꿔 답했다.
    이용대상이 겹쳐도 다른 제도이므로 false.
  · 근거는 시장·구청장 등 행정기관의 권한·의무만 정하는데, 답변이 시민 개인의 행위를
    "허용되지 않는다/금지된다/해도 된다"고 단정했다.
  · 질문이 묻는 금액·시간·기간·벌칙이 근거에 없어서, 답변이 "규정이 없다/명시되어 있지 않다/따로 정한다"로 끝난다.
  · 답변이 질문한 요금(예: 공원 입장료)에 다른 조례·다른 요금(예: 주차요금)의 중복 제한·감면 규정을 적용했다.
- true인 경우: 근거에 질문의 답이 되는 내용(예·아니요 판단의 근거, 금액, 대상, 절차 등)이 직접 있다.
  · 질문이 "~하면 어떻게 해야 해?"처럼 대처 방법을 묻고, 답변이 신고를 받아 행정기관이 하는 구조·보호 절차를
    근거대로 답했다면 true. (행위 허용 "~해도 되나요"를 행정기관 권한으로 단정한 경우와는 다르다)
  · 질문이 "~하려면?"처럼 절차를 묻고, 답변이 그 일을 하는 주체(예: 시장)와 절차(지정·의견수렴·고시·표지판 설치 등)를
    근거대로 답했다면 true. 질문한 사람이 직접 하는 절차가 아니라 행정기관이 하는 절차여도 질문에 답한 것이다.
  · 질문이 언제부터 적용되는지·기준일·소급 여부를 묻고, 답변이 근거의 "개정 이력"(신설·개정 날짜)이나 부칙의 시행일·적용례로
    답했다면 true. 이때 "그 전으로 거슬러 적용한다는 규정은 없다"는 덧붙임은 결론의 일부이므로 위의 '규정이 없다' false에 해당하지 않는다.
  · 질문이 요금 단위·기준을 잘못 전제했더라도(예: "○분당 얼마"), 바로 그 대상의 실제 요금·기준을 근거대로 바로잡아 답했다면 true.
    (단, 다른 대상의 요금으로 대신 답했다면 위의 false)

3) unneeded: [답변]의 문장 번호 중 질문에 답하는 데 필요 없는 문장 (정수 배열, 없으면 빈 배열)
- 필요 없는 문장: 질문이 묻지 않은 다른 상황·다른 대상·다른 항의 내용.
  예) 장소 출입을 물었는데 사고가 나면 보상·격리하는 내용, 이용할 수 있는지를 물었는데 설치 비율·위치 기준,
      해도 되는지를 물었는데 담당 기관·사무 위임, 질문과 관계없는 다른 혜택·다른 시설.
- 필요한 문장(넣지 않는다): 질문의 결론, 그 결론이 성립하는 조건·예외·단서, 질문이 물은 금액·대상·기간·절차.
- 1번 문장은 넣지 않는다. 판단이 애매하면 넣지 않는다.
결과는 JSON으로 낸다."""

_GROUND_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "groundedness",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"verdict": {"type": "string", "enum": ["grounded", "notGrounded", "notSure"]},
                           "answers_question": {"type": "boolean"},
                           "unneeded": {"type": "array", "items": {"type": "integer"}}},
            "required": ["verdict", "answers_question", "unneeded"],
            "additionalProperties": False,
        },
    },
}


def split_sentences(answer):
    """답변을 문장으로 나눕니다 (검증기 번호와 rag.py의 문장 빼기가 같은 기준을 쓰도록 한곳에 둠)."""
    return [x for x in re.split(r"(?<=[.!?])\s+", answer.strip()) if x]


def groundedness(context, answer, question="", detail=False):
    """답변 검증: grounded / notGrounded / notSure, 그리고 질문에 직접 답하는지(offTopic이면 'offTopic').

    원래 Upstage 'groundedness-check' 모델을 썼으나 지원 종료(400)되어 solar-pro4로 같은 판정을 합니다.
    질문을 함께 주면 '근거는 맞지만 질문과 다른 대상을 답한 경우'도 잡아 'offTopic'을 돌려줍니다.
    """
    numbered = "\n".join(f"[{i}] {x}" for i, x in enumerate(split_sentences(answer), 1))
    raw = chat([{"role": "system", "content": _GROUND_SYSTEM},
                {"role": "user", "content": f"[질문]\n{question}\n\n[근거]\n{context}\n\n[답변]\n{numbered}"}],
               response_format=_GROUND_SCHEMA)
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        out = {}
    verdict = out.get("verdict", "notSure")
    if question and out.get("answers_question") is False:
        verdict = "offTopic"
    if not detail:
        return verdict
    return verdict, [i for i in out.get("unneeded", []) if isinstance(i, int)]


_EDIT_SYSTEM = """너는 자치법규(조례·규칙) 답변의 편집자다. [질문]과 번호를 붙인 [답변] 문장을 보고,
질문이 묻지 않은 내용을 담은 문장의 번호를 고른다. 내용이 맞는지는 판단하지 않는다(이미 검증됨).
- 지울 문장: 질문과 다른 상황·다른 대상·다른 절차를 설명하는 문장.
  예) 이용할 수 있는지 물었는데 설치 기준·설치 비율·설치 위치를 설명함
      이용할 수 있는지만 물었는데(요금은 묻지 않음) 요금·입장료 면제·감면을 설명함
      장소 출입을 물었는데 사고가 났을 때의 보상·조치를 설명함
      대처 방법을 물었는데 그 뒤의 공고·반환·입양 같은 세부 절차까지 이어서 설명함
      질문에 나오지 않은 다른 시설·다른 동물·다른 사람·다른 요금에 관한 규정
      과태료 금액을 물었는데 가중·감경 기준, 이의제기 절차, 사무 위임을 설명함 ("다만"으로 시작해도 지운다)
      질문한 요금과 다른 요금(다른 시설·다른 조례)의 감면·중복 규정을 설명함
- 남길 문장: 질문에 대한 결론, 그 결론이 적용되는 조건·예외·단서(예: "원칙적으로 무료이고 특별한 경우에만 요금을 받는다"),
  질문이 물은 금액·대상·기간·절차의 핵심. 결론의 근거를 설명하는 문장.
- 1번 문장은 고르지 않는다. 1번이 "네, 이용할 수 있습니다."처럼 짧은 결론뿐이면 그 이유를 설명하는 2번 문장도 남긴다.
- 지울 문장이 없으면 빈 배열."""

_EDIT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "edit",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"remove": {"type": "array", "items": {"type": "integer"}}},
            "required": ["remove"],
            "additionalProperties": False,
        },
    },
}


def unasked_sentences(answer, question):
    """질문이 묻지 않은 문장 번호 (1부터). 근거 없이 질문과 답 문장만 보고 고릅니다."""
    sents = split_sentences(answer)
    if len(sents) < 2:
        return []
    numbered = "\n".join(f"[{i}] {x}" for i, x in enumerate(sents, 1))
    raw = chat([{"role": "system", "content": _EDIT_SYSTEM},
                {"role": "user", "content": f"[질문]\n{question}\n\n[답변]\n{numbered}"}],
               response_format=_EDIT_SCHEMA)
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [i for i in out.get("remove", []) if isinstance(i, int) and 1 < i <= len(sents)]
