"""Upstage API 호출을 한곳에 모았습니다. (Chat · Embedding · Document Parse · Groundedness Check)

Solar는 OpenAI SDK와 호환됩니다. base_url만 https://api.upstage.ai/v1 로 바꾸면 됩니다.
"""
import json
import random
import time

import requests

import config

_client = None


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
    return resp.choices[0].message.content


def embed(texts, model):
    out = []
    for i in range(0, len(texts), 50):        # 한 번에 너무 많이 보내지 않습니다
        batch = texts[i:i + 50]
        resp = _retry(lambda: client().embeddings.create(model=model, input=batch))
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


_GROUND_SYSTEM = """너는 서울특별시 조례 답변 검증기다. [질문]·[근거]·[답변]을 보고 두 가지를 판정한다.

1) verdict: 답변이 근거에 기반하는가
- grounded: 답변의 모든 사실(금액·비율·나이·조건·예외)이 근거에 그대로 있거나 근거에서 바로 도출된다.
- notGrounded: 답변에 근거에 없는 사실이 있거나, 근거와 어긋나는 내용이 있다.
  근거에 없는 권고·안내(예: "구청에 신고하세요", "~을 이용하는 것이 맞습니다")를 사실처럼 덧붙인 경우도 notGrounded.
- notSure: 근거만으로는 판단하기 어렵다.

2) answers_question: 근거가 질문이 묻는 바로 그것에 직접 답하는가 (true/false)
- false인 경우:
  · 근거가 질문 대상과 다른 대상(다른 시설·다른 요금·다른 위반행위)에 관한 것인데 답변이 이를 끌어다 썼다.
  · 질문한 행위와 근거가 규제하는 행위가 다르다. 예) 질문은 시민이 음식을 사 먹거나 배달시키는 것인데 근거는 물건을 파는 상행위·영업을 규제한다.
    (판매자·영업자에 대한 규정을 이용자·구매자의 행위에 그대로 적용했다면 false)
  · 질문이 근거에 없는 이름의 시설·제도(예: 임산부 전용주차구역, 여성전용주차장)를 물었는데, 답변이 다른 이름의 제도
    (예: 가족배려주차장 주차구획)로 바꿔 답했다. 이용대상이 겹쳐도 다른 제도이므로 false.
  · 근거는 시장·구청장 등 행정기관의 권한·의무(포획·방사, 급식소 설치 등)만 정하는데, 답변이 시민 개인의 행위를
    "허용되지 않는다/금지된다/해도 된다"고 단정했다.
  · 질문이 묻는 금액·시간·기간·벌칙이 근거에 없어서, 답변이 "규정이 없다/명시되어 있지 않다/따로 정한다"로 끝난다.
  · 질문이 특정 자치구(OO구)의 조례·규칙·요금·시설을 묻는데 근거는 서울특별시 조례다.
- true인 경우: 근거에 질문의 답이 되는 내용(예·아니요 판단의 근거, 금액, 대상, 절차 등)이 직접 있다.
  · 질문이 "~하려면?"처럼 절차를 묻고, 답변이 그 일을 하는 주체(예: 시장)와 절차(지정·의견수렴·고시·표지판 설치 등)를
    근거대로 답했다면 true. 질문한 사람이 직접 하는 절차가 아니라 행정기관이 하는 절차여도 질문에 답한 것이다.
  · 질문이 요금 단위·기준을 잘못 전제했더라도(예: "5분당 얼마"), 바로 그 대상의 실제 요금·기준을 근거대로 바로잡아 답했다면 true.
    (단, 다른 대상의 요금으로 대신 답했다면 위의 false)
결과는 JSON으로 낸다."""

_GROUND_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "groundedness",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"verdict": {"type": "string", "enum": ["grounded", "notGrounded", "notSure"]},
                           "answers_question": {"type": "boolean"}},
            "required": ["verdict", "answers_question"],
            "additionalProperties": False,
        },
    },
}


def groundedness(context, answer, question=""):
    """답변 검증: grounded / notGrounded / notSure, 그리고 질문에 직접 답하는지(offTopic이면 'offTopic').

    원래 Upstage 'groundedness-check' 모델을 썼으나 지원 종료(400)되어 solar-pro4로 같은 판정을 합니다.
    질문을 함께 주면 '근거는 맞지만 질문과 다른 대상을 답한 경우'도 잡아 'offTopic'을 돌려줍니다.
    """
    raw = chat([{"role": "system", "content": _GROUND_SYSTEM},
                {"role": "user", "content": f"[질문]\n{question}\n\n[근거]\n{context}\n\n[답변]\n{answer}"}],
               response_format=_GROUND_SCHEMA)
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return "notSure"
    if question and out.get("answers_question") is False:
        return "offTopic"
    return out.get("verdict", "notSure")
