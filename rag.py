"""4단계: 검색 → Solar 답변 → 검증 → 대회 응답 형식.

설계 원칙 세 가지
  1. LLM은 근거 조항의 '라벨'(키값)만 고릅니다. 고를 수 있는 값은 이번에 넘긴 근거 라벨로 제한(enum)하고,
     citation·링크·시행일은 코드가 그 라벨의 조각에서 채웁니다.
  2. source_url · effective_date도 코드가 채웁니다. LLM이 링크나 날짜를 쓰게 두지 않습니다.
  3. 의심스러우면 '확인 불가'. 근거 없는 답은 맞아도 점수가 없고, 함정 문항에서는 감점입니다.
"""
import json
import math
import re
import threading
import time
from collections import Counter

import numpy as np

import config

# 일반화 점검용 스위치 (환경변수 RAG_LEAN="guard,post,prompt" 중 일부). 비워 두면 지금 동작 그대로 (제출·시연 영향 없음)
#   guard : 답을 확인 불가로 돌리는 정규식 가드(no_rule·no_amount·annex_only·name_mismatch 등)를 끔. 검증기는 그대로
#   post  : 답을 고쳐 쓰는 후처리(네/아니요 교정, 문장·인용 빼기 등)와 공원↔주차장 근거 분리를 끔
#   prompt: 질문별 예시가 쌓인 SYSTEM 대신 일반 규칙만 있는 SYSTEM_LEAN 사용
import os as _os
LEAN = {x.strip() for x in _os.environ.get("RAG_LEAN", "").split(",") if x.strip()}
_G = "guard" not in LEAN
# 하드코딩(특정 조항 번호·주제·문구에 묶인 규칙)을 한꺼번에 켜고 끄는 스위치. 기본은 끔.
#   대상: 조항별 문장 빼기 필터, 공원↔주차장 근거 분리, 공영주차장·금주구역·65세·중복할인 문구 교정,
#         특정 조항 예외(GENERIC_BASIS), 중복감면·금주구역 인용 보충, '전용주차 없다' 문장 빼기
#   일반 장치(프롬프트 일반 규칙, 가드, 검증기 문장 빼기, 관련 호 표시 등)로 대신합니다. 비교용으로 RAG_TOPIC=1 이면 켬.
USE_TOPIC_FILTERS = _os.environ.get("RAG_TOPIC", "0") == "1"
TAIL_STRIP = "tail" not in LEAN       # 덧붙인 '정해져 있지 않다' 문장을 떼고 검증기에 맡김 (RAG_LEAN=tail 이면 끔)
RELATED_ITEMS = "items" not in LEAN   # 긴 목록 항에 '질문과 관련된 호' 표시 (RAG_LEAN=items 이면 끔)
CITE_SOFT_MAX = 1                     # 인용이 이보다 많으면 새 내용을 보탠 근거만 남김
# 인용 상한 (2026-11-21 운영 공지: 인용 수가 정답 근거 수보다 2개 넘게 많으면 넘친 비율만큼 감점).
# 정답 근거는 거의 1~2개라 3개까지는 어떤 문항에서도 감점되지 않습니다. 넘치면 첫 근거 + 답과 가장 많이 겹치는 근거만 남깁니다.
CITE_HARD_MAX = int(_os.environ.get("RAG_CITE_MAX", "3"))
RELATED_MIN_ITEMS = 5                 # 호가 이만큼 이상인 항에만 표시
RELATED_KEEP = 3                      # 표시할 호 최대 개수
RELATED_COMMON = 0.25                 # 그 항의 호 중 이 비율 이상에 나오는 글자쌍은 목록 공통어로 보고 뺌
RELATED_MIN_LEX = 6.0                 # 글자 겹침으로 고를 최소 점수 (idf 합)
RELATED_SEM_GAP = 0.02                # 의미 점수 1위가 2위보다 이만큼 높으면 그 호도 더함
STRICT_RETRY = "strict" not in LEAN   # 1차가 '근거 없음'이면 2차 답을 엄격하게 검증 (RAG_LEAN=strict 이면 끔)
CLAIM_CHECK = _os.environ.get("RAG_CLAIMS", "1") == "1"   # 주장 단위 검증 (기본 켬, RAG_CLAIMS=0 이면 끔) (대상→조항·호·분류를 조문과 대조)
CLAIM_ROUNDS = 1                      # 주장이 조문과 어긋날 때 사실을 알려 주고 다시 쓰게 하는 최대 횟수
FIRST_TRIES = int(_os.environ.get("RAG_FIRST_TRIES", "3"))   # 1차 답을 동시에 몇 번 돌려 먼저 통과한 것을 쓸지 (1이면 예전처럼 한 번)
TRIM_UNNEEDED = "trim" not in LEAN     # 검증기가 고른 '필요 없는 문장' 빼기 (RAG_LEAN=trim 이면 끔)

UNKNOWN = {"answer": "확인 불가", "citation": [], "source_url": "", "effective_date": ""}
DISTRICT = re.compile(r"서울특별시\s*\S+구\s")      # 'OO구' 자치구 조례 인용 방지
# 질문이 특정 자치구의 요금·조례·규칙을 물으면 6개 서울특별시 조례 범위 밖입니다 (대회 범위). 25개 자치구 이름은 공개 행정구역 목록.
GU_NAMES = re.compile(r"(?<![가-힣])(?:종로구|중구|용산구|성동구|광진구|동대문구|중랑구|성북구|강북구|도봉구|노원구|은평구|서대문구|마포구|양천구|"
                      r"강서구|구로구|금천구|영등포구|동작구|관악구|서초구|강남구|송파구|강동구)(?=[\s,?.]|의|에|은|는|도|$)")
# 할인·감면·면제도 그 자치구 시설 요금에 대한 질문입니다 ("강남구 공영주차장에서 장애인·다둥이 할인 같이?"가 서울시 조례로 답했었음)
GU_RULE_Q = re.compile(r"요금|조례|규칙|과태료|얼마|정기권|입장료|사용료|이용료|수수료|할인|감면|면제")
AMOUNT_Q = re.compile(r"얼마(?!\s*동안|나)|얼마나\s*(?:내|드|들|받|할인|감면|깎|비싸|싸|나와|나오|줘|주|돼|되)")            # 금액·비율을 묻는 질문 ('얼마 동안/얼마나 자주'는 제외)
AMOUNT_A = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:만\s*원|천\s*원|원)|\d+분의\s*\d+|\d+(?:\.\d+)?\s*(?:%|퍼센트)|\d+\s*할(?!인)|무료|전액")
NO_AMOUNT = re.compile(r"(금액|요금|비용|액수)[^.]{0,25}(정해져\s*있지\s*않|규정되어\s*있지\s*않|나와\s*있지\s*않|명시되어\s*있지\s*않|정하고\s*있지\s*않|알\s*수\s*없)"
                       # "충전요금이라는 별도의 요금이 없습니다 / 충전요금 자체를 따로 정하지 않고" → 다른 요금(주차요금)으로 대신 답한 것
                       r"|따로\s*정하지\s*않|별도의?\s*(요금|금액|비용|이용료|사용료)\S*\s*(은|는|이|가)?\s*없"
                       r"|(요금|금액|비용|이용료|사용료)(은|는|이|가)\s*(따로\s*|별도로\s*)?없")
COST_Q = re.compile(r"비용|요금|돈|이용료|입장료|사용료|수수료|과태료|벌금|얼마\s*(드|들|내)")   # 금액을 묻는 질문
RATE_Q = re.compile(r"할인|감면|면제|퍼센트|%|비율|몇\s*배|돌려|반환|환불")   # 환불액은 "1할"처럼 비율로 정해짐                                   # 비율을 묻는 질문
WON_A = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:만\s*원|천\s*원|원)|무료")                         # 답에 실제 금액
PREMISE_FIX = re.compile(r"(분|시간|일|월|회)\s*(단위|당|기준)(가|이)?\s*아니라")   # "5분 단위가 아니라 1시간당 …"
TARGET_Q = re.compile(r"([가-힣]{2,})\s*(?:전용|우선)")                               # "여성전용/임산부 전용/장애인 전용"
NAME_ACK = re.compile(r"(이라는|라는)\s*(명칭|이름|용어)|(명칭|이름)(은|의)?[^.]{0,20}(없|아니)"
                      r"|별도(의|로)?\s*(제도|구획|주차장|주차구역)[^.]{0,15}(없|아니)"
                      r"|(은|는)\s*조례에\s*(는\s*)?따로\s*(없|있는 것이 아니)[^.]{0,60}(대신|역할|말하|정하고|정한 것)")  # 답이 '그런 이름은 없다'고 인정
# "임시 보호하는 제도는/것은/것에 관해서는 …" — 문장이 가리키는 행위·제도 이름을 꺼냅니다
METHOD_ACK = re.compile(r"([가-힣]{2,}(?:\s[가-힣]{2,})?)\s*(?:하|가)?는\s*(?:제도|것|방식)(?:은|는|이|가|에\s*관해서는|에\s*대해서는)?\s")
# U23 "네, 급식소 설치라는 방식으로 가능합니다. 다만 개인이 밥을 주는 행위를 직접 허용하는 규정은 아니고…"
STRONG_NO_RULE = re.compile(r"(허용|금지|정하)(하)?는\s*규정(은|이|을|도)?\s*(따로\s*)?(없|아니|두고\s*있지\s*않)")
ZONE_FINE = re.compile(r"([가-힣]{1,6}구역)에서")   # "○○구역에서 … 과태료" → 제목에 "○○구역 … 지정"이 있는 조를 함께 인용                                     # 과태료 항이 가리키는 구역
UNIT_Q = re.compile(r"\d+\s*(분|시간)\s*(당|에|마다|단위)|시간\s*(당|단위)")              # "5분당/10분에/1시간 단위/시간당"
MUST_Q = re.compile(r"야\s*(하나요|해요|되나요|돼요|합니까|됩니까|해(?![가-힣])|하나(?![가-힣])|돼(?![가-힣])|되(?![가-힣])|하는지|하는\s*거)")   # "~해야 하나요/해야해?"
WH_QUESTION = re.compile(r"얼마|몇|누구|누가|어디|언제|무엇|뭐|뭘|무슨|어떻게|어떤|며칠|[가-힣](?:은|는)\s*\?*\s*$|면\s*\?*\s*$")   # "…지원은?"도 내용 질문   # 예·아니요로 답하지 않는 질문
NEG_END = re.compile(r"(없습니다|않습니다|안 됩니다|안됩니다|불가합니다|아닙니다|못합니다|금지됩니다|금지되어 있습니다)[.]?$")
NOT_NEEDED = re.compile(r"(않아도|안 해도|안 내도|없어도)\s*됩니다|필요(가|는)?\s*없|(해야|하여야)\s*하는\s*것(은|이)\s*아니")
MUST_NOT = re.compile(r"^(?:반드시\s*|꼭\s*)?[^.,]{0,30}(해야|하여야)\s*하는\s*것(은|이)\s*아니")   # 첫 문장이 "반드시 ~해야 하는 것은 아니…"
# 근거에 없는 추측·유보("법령과 현장 여건에 따라 달라질 수 있습니다", "해당 법령에 따라 판단해야 합니다")
RESTATE = re.compile(r"^\s*즉\s*,[^.]*(?:제도|뜻|의미|말|것)입니다")   # "즉, … 지원받을 수 있는 제도입니다" 같은 되풀이
BARE_YES = re.compile(r"^\s*(네|아니요)\s*,?\s*[^.]{0,20}니다\.?\s*$")
HEDGE = re.compile(r"(?:시가|시장이)\s*마련하는\s*(?:사업|시책)[^.]{0,10}?(?:에서|에\s*따라)\s*정|예산\s*(?:상황|사정|여건)에\s*따라|(?:마련하는|정하는)\s*(?:지원\s*)?(?:방안|사업|시책)에서\s*정|달라질\s*수\s*있습니다|따라\s*판단(해야|하여야|됩니다|할\s*수)")
# "네, (원칙적으로) 내야 하지만 / 내야 합니다만," 같은 머리말 (감면·면제 답에서 떼어 냄)
PAY_BUT = re.compile(r"^네\s*,\s*(?:원칙적으로\s*|기본적으로\s*|일반적으로\s*)?(?:[가-힣]+(?:은|는|을|를)\s*)?"
                     r"내야\s*(?:하지만|하나|합니다만|하는데)\s*,?\s*")
# 첫 문장이 '규정 없음'이면 확인 불가. 예) D10 "야간 정기권 이용 시간이 정해져 있지 않습니다", D2 "운영 시간은 조례에 정해져 있지 않습니다"
# '않은'(꾸밈말: "기간이 따로 정해져 있지 않은 경우")은 결론이 아니라서 뺍니다 (F12 오탐 방지)
NO_RULE = re.compile(r"(규정|조항|내용)(은|이|도)?\s*((이\s*)?(근거|조례)들?(에는|에)\s*)?(따로\s*)?(없|존재하지 않)"
                     r"|명시(되어|돼)?\s*있지 않(?!은)|정하고 있지 않(?!은)|규정하고 있지 않(?!은)"
                     # 킥보드 "두 명이 함께 타는 것을 허용하거나 금지하는 내용은 정하지 않습니다"
                     r"|(내용|규정|사항)(은|을|도)?\s*(따로\s*)?정하지\s*않(?!은|으면)"
                     r"|(정해져|나와|적혀|규정되어|기재되어)\s*있지\s*않(?!은)|정해지지\s*않(?!은)|따로\s*정하는\s*바에\s*따르"
                     r"|(근거|조례)들?(에는|에|만으로는)\s*(따로\s*)?없|(근거|조례)들?만으로는|(답할|확인할|판단할|말할)\s*수\s*없"
                     # Z2 "음주 자전거를 벌금으로 직접 부과하지 않습니다. 대신 (금주구역 과태료 10만원)…"
                     r"|(직접|따로|별도로)\s*(부과|처벌|정|규정)하?(지|고\s*있지)\s*않(?!은)"
                     # U24 침묵 논증: "조례는 밥 주는 행위를 금지하지 않습니다 / 허용·금지하는 규정을 두고 있지 않습니다"
                     r"|(규정|조항)(을|은|는|도)?\s*(따로\s*)?두고\s*있지\s*않(?!은)|금지하지\s*않(?!은|아도)")

_cache = {}


# ------------------------------------------------------------ 검색
def _bigrams(text):
    t = re.sub(r"\s+", "", text)
    return [t[i:i + 2] for i in range(len(t) - 1)]


def _load():
    unit = config.CHUNK_UNIT
    if unit in _cache:
        return _cache[unit]
    chunks = json.loads((config.DATA / f"chunks_{unit}.json").read_text(encoding="utf-8"))
    vec_path = config.DATA / f"vectors_{unit}.npy"
    vecs = np.load(vec_path) if vec_path.exists() else None
    if vecs is not None:
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    # 키워드 검색용 (글자 2개 단위 BM25)
    docs = [Counter(_bigrams(c["label"] + " " + c["text"])) for c in chunks]
    df = Counter(g for d in docs for g in d)
    avg = sum(sum(d.values()) for d in docs) / len(docs)
    _cache[unit] = (chunks, vecs, docs, df, avg)
    return _cache[unit]


def _bm25(query, docs, df, avg, k1=1.2, b=0.75):
    n = len(docs)
    q = set(_bigrams(query))
    scores = np.zeros(n)
    for i, d in enumerate(docs):
        ln = sum(d.values())
        s = 0.0
        for g in q:
            if g in d:
                idf = math.log(1 + (n - df[g] + 0.5) / (df[g] + 0.5))
                s += idf * d[g] * (k1 + 1) / (d[g] + k1 * (1 - b + b * ln / avg))
        scores[i] = s
    return scores


REWRITE_SYSTEM = """너는 서울특별시 조례 검색 도우미다.
시민의 질문을 조례 조문에 실제로 쓰일 법률 용어로 바꾼 검색어를 서로 다른 관점으로 3개 만든다.
- 일상어를 조문 용어로 바꾼다. 예) 돈을 내야 → 요금·사용료 징수, 할인 → 감면, 벌금 → 과태료, 몇 살 → 정의·연령
- 같은 뜻의 다른 법률 표현을 각각 써 본다. 예) 팔다 → 판매·영업·상행위, 맡기다·맡아 주다·돌봐 주다 → 위탁·인수·인수신청·보호조치, 월세 → 차임·임차료, 전세 → 임차보증금
- 질문에 답하려면 함께 봐야 할 조항의 용어도 넣는다. 예) 금지·지정 질문 → 과태료 부과, 감면 질문 → 중복 감면·감면율
- 이용할 수 있는지 묻는 질문은 이용·개방·이용대상 용어를 넣는다.
- "~하면 어떻게 해야 하나요"처럼 대처 방법을 묻는 질문은 발견·신고·구조·조치 절차 용어를 넣는다.
- 질문에 없는 조례 이름이나 숫자는 지어내지 않는다.
- 검색어 하나는 15단어 이내로 짧게 쓴다.
질문 유형도 함께 판정한다.
- yes_no: "~되나요/~할 수 있나요/~해야 하나요/~인가요?"처럼 네·아니요로 답할 질문이면 true.
  "얼마/누구/언제/어디/무엇/어떻게/~은?/~다면?"처럼 내용을 묻는 질문이면 false.
- asks: 질문이 알고 싶은 것 하나. 금액·요금(얼마·요금·할인율), 대상·자격(누가 받나), 기간·시기(언제·며칠·몇 년),
  절차·방법(어떻게 하나·하려면), 허용·의무 여부(해도 되나·해야 하나), 결과·제재(하면 어떻게 되나·과태료), 사실·정의, 기타.
  "얼마나 크게/자주"처럼 돈이 아닌 정도를 물으면 금액·요금이 아니다.
- multi: 둘 이상의 혜택·감면·사유를 함께(둘 다·중복·같이·동시에·두 번) 받거나 적용하는지 묻는 질문이면 true.
결과는 JSON으로 낸다."""

REWRITE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "queries", "strict": True,
        "schema": {"type": "object",
                   "properties": {"queries": {"type": "array", "items": {"type": "string"}},
                                  "yes_no": {"type": "boolean"},
                                  "asks": {"type": "string", "enum": ["금액·요금", "대상·자격", "기간·시기", "절차·방법", "허용·의무 여부", "결과·제재", "사실·정의", "기타"]},
                                  "multi": {"type": "boolean"}},
                   "required": ["queries", "yes_no", "asks", "multi"], "additionalProperties": False},
    },
}


QTYPE_MODE = _os.environ.get("RAG_QTYPE", "regex")   # regex | llm(전부 LLM 판정) | mixed(둘 다·중복 판정만 LLM)
QTYPE_LLM = QTYPE_MODE == "llm"
_qtype_cache = {}
BOTH_RE = re.compile(r"둘\s*다|모두|함께|같이|동시에|중복|두\s*번|겹")


def _qt(question):
    return _qtype_cache.get(question) if QTYPE_LLM else None


def is_content_q(question):
    """내용 질문(얼마·누구·언제·~은?) = 네·아니요로 답하지 않는 질문."""
    t = _qt(question)
    return (not t["yes_no"]) if t else bool(WH_QUESTION.search(question))


def is_amount_q(question):
    t = _qt(question)
    return t["asks"] == "금액·요금" if t else bool(AMOUNT_Q.search(question))


def is_when_q(question):
    t = _qt(question)
    return t["asks"] == "기간·시기" if t else bool(WHEN_Q.search(question))


def is_proc_q(question):
    t = _qt(question)
    return t["asks"] == "절차·방법" if t else bool(PROC_Q.search(question))


def is_multi_q(question):
    t = _qtype_cache.get(question) if QTYPE_MODE in ("llm", "mixed") else None
    return t["multi"] if t else bool(BOTH_RE.search(question))


def rewrite(question):
    """질문을 조문 용어로 바꾼 검색어 목록(최대 3개). 실패하면 빈 목록(원래 질문만으로 검색)."""
    if not (config.USE_REWRITE and config.UPSTAGE_API_KEY):
        return []
    import upstage
    try:
        out = json.loads(upstage.chat([{"role": "system", "content": REWRITE_SYSTEM},
                                       {"role": "user", "content": question}], response_format=REWRITE_SCHEMA))
        if question not in _qtype_cache and isinstance(out.get("yes_no"), bool):
            _qtype_cache[question] = {"yes_no": out["yes_no"], "asks": out.get("asks", "기타"),
                                      "multi": bool(out.get("multi"))}
        return [q.strip()[:120] for q in out.get("queries", []) if q.strip()][:3]
    except Exception:  # noqa: BLE001  재작성은 보조 수단이라 실패해도 검색은 계속합니다
        return []


HO_LINE = re.compile(r"^\s*\d+(?:의\d+)?\.\s")
_subs = {}


def _sub_ids(chunks, parent):
    if not _subs:
        for c in chunks:
            if c.get("sub"):
                _subs.setdefault(c["parent"], []).append(c["id"])
    return _subs.get(parent, [])


def _trim_ho(hit, chunks, score, best_i):
    """긴 항에서 검색에 걸린 호(와 그 아래 목)만 남깁니다.

    예) 주차장 조례 제7조 제1항은 감면 대상 15개 호(약 3,000자)라, 경차 질문이면 제2호만 필요합니다.
    항 머리 문장은 그대로 두고, 생략한 자리에는 '(다른 호 생략)'을 남겨 목록이 일부임을 알립니다.
    호 조각이 아니라 항 전체가 가장 잘 맞았으면 손대지 않습니다.
    """
    if not chunks[best_i].get("sub") or len(hit["text"]) < config.TRIM_MIN_CHARS:
        return hit
    subs = sorted(_sub_ids(chunks, hit["id"]), key=lambda i: -score[i])
    top = score[subs[0]]
    keep = {chunks[i]["text"].splitlines()[-1].strip()
            for i in subs[:config.TRIM_KEEP] if score[i] >= top - config.TRIM_MARGIN}
    head, segs = [], []
    for ln in hit["text"].splitlines():
        if HO_LINE.match(ln):
            segs.append([ln])
        elif segs:
            segs[-1].append(ln)                 # 목(가. 나. …)·이어지는 줄은 앞 호에 붙임
        else:
            head.append(ln)
    out, skipped = list(head), False
    for seg in segs:
        if seg[0].strip() in keep:
            out += seg
            skipped = False
        elif not skipped:
            out.append("(다른 호 생략)")
            skipped = True
    return {**hit, "text": "\n".join(out), "trimmed": True}


def retrieve(question, top_k=None):
    """임베딩(의미) + 키워드(용어) 혼합 검색.

    1) 원래 질문 + 조문 용어로 재작성한 질문, 둘 다로 검색 (임베딩은 더 가까운 쪽 점수)
    2) 호 단위 보조 조각에 걸리면 원래 조·항 조각으로 묶어서 돌려줌
    3) 상위 조의 다른 항(예외·중복 적용·감면율 등)을 함께 붙임
    """
    top_k = top_k or config.TOP_K
    chunks, vecs, docs, df, avg = _load()
    rqs = rewrite(question)
    kw = _bm25(" ".join([question] + rqs), docs, df, avg)
    kw_norm = kw / kw.max() if kw.max() > 0 else kw
    if vecs is not None and config.UPSTAGE_API_KEY:
        import upstage
        qv = np.array(upstage.embed([question] + rqs, config.EMBED_QUERY), dtype=np.float32)
        qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
        cos = (vecs @ qv.T).max(axis=1)
        # 의미 점수 비중: 연습+새 문항 36개로 0.5~1.0을 재 본 결과 0.8~0.95가 가장 좋아 0.9로 둡니다
        score = config.EMBED_WEIGHT * cos + (1 - config.EMBED_WEIGHT) * kw_norm
    else:
        cos, score = None, kw_norm

    # 보조 조각(호) → 원래 조·항(parent)으로 묶고, 묶음 점수는 가장 높은 조각 점수
    best = {}
    for i in np.argsort(score)[::-1]:
        p = chunks[i].get("parent", i)
        if p not in best:
            best[p] = i
        if len(best) >= top_k:
            break
    hits = [{**chunks[p], "score": float(score[i]),
             "cos": float(cos[i]) if cos is not None else None} for p, i in best.items()]
    if config.TRIM_HO:
        hits = [_trim_ho(h, chunks, score, best[h["id"]]) for h in hits]
    sem = cos if cos is not None else score
    for h in hits:
        h["related"] = _related_items(h, chunks, sem, question)

    # 같은 조의 다른 항 붙이기: 원칙 조항만 찾고 예외·감면율 조항을 놓치는 일을 막습니다
    if config.EXPAND_SIBLINGS:
        have = {h["id"] for h in hits}
        arts = list(dict.fromkeys(h.get("article") for h in hits[:config.EXPAND_TOP] if h.get("article")))
        for c in chunks:
            if len(hits) >= config.MAX_CONTEXT:
                break
            if c.get("article") in arts and not c.get("sub") and c["id"] not in have \
                    and c["label"] != c.get("article"):          # 항이 있는 조만 (항 없는 조·별표 제외)
                hits.append({**c, "score": 0.0, "cos": None, "expanded": True})
                have.add(c["id"])
    return hits


# ------------------------------------------------------------ 답변
# 일반 규칙만 둡니다. 특정 질문을 그대로 옮긴 예시는 넣지 않습니다 (처음 보는 질문에 효과가 없고 답을 길게 만들었음, 2026-11-18 점검).
# 이전 버전(질문별 예시 40개)은 backup_code_v4_before_cleanup/rag.py
SYSTEM = """너는 서울특별시 조례 질의응답 도우미다.
답하는 방식:
- 아래 [근거]에 적힌 내용만으로 답한다. 근거에 없는 사실·권고("~하세요/~하는 것이 좋습니다")·해석은 덧붙이지 않는다.
- 질문이 묻는 것에만 답한다. 같은 조의 다른 항, 과태료 가중·감경, 이의제기 절차, 사무 위임, 용어 정의처럼 묻지 않은 내용은 쓰지 않는다.
  같은 뜻을 "즉, …/이는 …입니다"로 되풀이하지 않는다. 일반 시민이 알기 쉬운 말로 1~3문장으로 쓴다.
- 질문에 답할 수 있는 내용을 답했으면, 근거에 없는 세부사항에 대해 "~은 정해져 있지 않습니다/직접 정하지 않습니다"를 덧붙이지 않는다.
- 근거 끝에 "(질문과 관련된 호: …)"가 붙어 있으면 그 호부터 확인한다. 판단은 근거 본문으로 한다.
- 답 본문에 조례 이름·조항 번호를 쓰지 않는다 (인용은 따로 붙는다). 금액·비율·나이·기간 같은 숫자는 근거 그대로 쓴다.
- 예·아니요로 답할 질문이면 "네," 또는 "아니요,"로 시작한다. 네·아니요는 질문의 동사에 대한 답이고, 뒤 내용과 어긋나면 안 된다.
  예) "~내야 하나요?"에 안 내도 되면 "아니요, 내지 않아도 됩니다." / "얼마·누구·언제·무엇·어떻게"를 묻는 질문은 네·아니요 없이 답한다.
  · "반드시 ~해야 하나요?"인데 근거가 "~하도록 노력하여야 한다"(노력 의무)뿐이면 "아니요"이고, 노력 의무를 이어서 쓴다.
  · "~내야 하나요?"인데 근거가 그 사람의 감면·면제를 정하면 "네, 내야 하지만"으로 시작하지 말고 감면·면제부터 쓴다.
  · 질문의 구체적인 이름(물건·행위·사람)이 근거에 없고 더 넓은 말만 있으면 네·아니요로 단정하지 않는다.
    그 이름이 따로 정해져 있지 않다는 것과 근거가 정한 넓은 기준을 그대로 쓰고, "○○이 이에 해당합니다"처럼 해석하지 않는다.
  · 근거가 질문보다 좁거나 간접적이면 "네"로 단정하지 말고, 근거가 실제로 정한 방식(누가·누구에게·어떤 조건으로)을 첫 문장에 쓴다.
- 근거가 "시장은 ~할 수 있다"(재량)나 "~하도록 노력하여야 한다 / 대책을 수립하여야 한다"(정책 의무)면,
  "시장은 ~할 수 있습니다"처럼 근거의 주어·말투대로 쓴다. 시민이 반드시 받는다고 단정하지 않고, 근거에 없는 세부 결정 방식을 지어내지 않는다.
- 근거에 대상·요건 목록이 있으면 그 목록 기준으로 답한다. 목록에 질문한 사람의 이름이 없으면 그 이름으로는 대상이 아님을 쓰고, 해당할 수 있는 항목을 쓴다.
  한 항목에 여러 사람이 함께 적혀 있어도 질문한 사람만 쓴다. 여러 사람이 함께 쓰는 제도를 한 사람의 "전용/을 위한" 시설로 쓰지 않는다.
- 질문한 장소·대상의 일부만 근거에 있으면 그 일부와 조건만 정확히 쓰고, 전체를 "된다/안 된다"로 단정하지 않는다.
- 이용할 수 있는지(이용 가능·있나요)만 묻고 요금을 묻지 않았으면, 이용·개방을 정한 조항으로 답하고 요금 면제·감면은 쓰지 않는다.
- 절차·방법을 묻는 질문은 절차와 요건만 쓴다. 결과(피해·위반하면?)를 묻는 질문은 결과(조치·보상·과태료)만 쓴다.
- 금지·지정 구역을 쓸 때는 지정 조항이 정한 장소를 그대로 쓰고 더 넓은 장소로 바꾸지 않는다. 금지 여부를 물으면 위반 시 과태료 항이 있는지 본다.
- 원칙 조항과 함께 예외·단서·중복 적용 제한·감면율을 확인한다. 단서는 그 단서가 걸리는 경우에만 쓰고, 다른 경우의 반대 결론을 추론하지 않는다.
- 감면·면제를 둘 이상 함께 받을 수 있는지(둘 다·중복·같이) 물으면 다음 순서로 답한다. found=false로 하지 않는다.
  1) 각 대상이 질문한 요금의 어느 목록에 해당하는지 먼저 쓴다 (면제 목록인지, 몇 % 감면 목록인지).
  2) 그 요금을 정한 같은 조례에 중복 적용을 정한 항(예: "둘 이상의 감면사유에 해당하면 높은 하나만")이 있으면 그 항대로 답한다.
     다른 조례나 다른 요금의 중복 제한 규정은 질문한 요금에 가져다 쓰지 않는다.
  3) 같은 조례에 중복 적용 규정이 없으면:
     · 하나라도 면제 대상이면, 면제만으로 그 요금을 내지 않으므로 더 받을 감면이 없다고 쓴다.
       둘 다 면제 대상이면 "네/아니요" 없이 "두 경우 모두 면제 대상이라 하나만 해당해도 내지 않는다"로 쓴다.
     · 둘 다 감면율 대상이면 각 감면율을 쓰고, 겹쳐 적용하는지는 조례에 정해져 있지 않다고 쓴다.
  4) 서로 다른 요금(예: 입장료와 시설 이용료)에 대한 혜택이면 각각 적용된다고 쓴다.
- 질문이 묻는 시설·대상과 다른 시설의 혜택은 같은 사람에 관한 것이라도 쓰지 않는다.

근거 고르기 (evidence):
- 답의 내용이 실제로 적힌 조항만 넣는다. 답에 쓴 조항은 빠짐없이, 답에 쓰지 않은 정의·목적 조항이나 낱말만 같은 옆 조항은 넣지 않는다.
- 금액을 별표에서 가져왔으면 그 별표도 넣는다.
주장 적기 (claims, 있을 때만):
- 답에서 대상별로 내린 판단을 하나씩 적는다. subject는 질문 속 대상·행위 이름 그대로(예: 국가유공자, 30명 이상 단체, 맹견 출입),
  label은 그 판단의 근거 라벨, ho는 그 대상이 적힌 목록의 호 번호(목록이 아니면 빈 문자열), kind는 판단 종류.
  목록에 없어 대상이 아니라고 답한 대상은 kind를 "대상 아님"으로 적는다.
- found=false면 빈 배열.

found=false (확인 불가) 로 할 때:
- 질문이 묻는 내용이 근거에 없으면 found=false. 비슷한 다른 내용으로 대신 답하지 않는다.
  · 질문의 대상(시설·요금 종류·위반행위)과 근거의 대상이 다르면. 이름이 비슷한 다른 제도로 바꿔 답하지 않는다.
  · 질문의 행위와 근거가 규제하는 행위가 다르면 (예: 판매·영업 규정을 구매·이용 행위에 적용).
  · 질문은 시민이 해도 되는지 묻는데 근거는 행정기관의 권한·의무만 정하고 그 행위를 허용·금지하는 문장이 없으면.
    단, "어떻게 해야 하나요/어디에 신고하나요"처럼 대처 방법을 물었고 근거에 신고·구조 등 행정 절차가 있으면 그 절차로 답한다.
    "금지됩니다"는 근거에 그 행위를 금지하는 문장이 있을 때만 쓴다.
  · 근거가 다른 조례를 가리키기만 하고 질문이 묻는 기준·대상·금액을 직접 정하지 않으면.
  · 질문이 묻는 금액·시간·기간·벌칙이 근거에 없어서 답이 "정해져 있지 않다"가 될 상황이면.
  · 질문이 특정 자치구(OO구)의 조례·규칙·요금·시설을 묻는 것이면. 자치구 조례·규칙은 근거로 쓰지 않는다.
- 예외(전제 바로잡기): 질문이 같은 대상의 요금을 잘못된 단위로 물었을 뿐이고(예: 1시간 단위 요금을 "○분당 얼마"로 물음) 근거에 그 대상의 실제 기준이 있으면
  found=true로 하고 "○분 단위가 아니라 △△ 기준입니다"처럼 바로잡는다. 근거가 원칙(예: 무료)과 예외(예: 특별한 경우에만 요금)를 함께 정하면
  첫 문장에 원칙→예외 순서로 쓴다 (예외 금액만 쓰면 늘 그 요금을 내는 것처럼 읽힌다). 근거 단위보다 짧은 시간으로 나누어 근거에 없는 금액을 만들지 않는다.
  [계산 안내]가 붙어 있으면 그 안내를 따른다.
결과는 JSON으로 낸다."""

SYSTEM_LEAN = """너는 서울특별시 조례 질의응답 도우미다.
규칙:
- 아래 [근거]에 적힌 내용만으로 답한다. 근거에 없는 사실·권고·해석은 덧붙이지 않는다.
- 예·아니요로 답할 질문이면 "네," 또는 "아니요,"로 시작하고, 뒤 내용이 그 결론과 어긋나지 않게 쓴다.
  "얼마/누구/언제/무엇/어떻게"처럼 내용을 묻는 질문은 네·아니요 없이 바로 답한다.
- 질문이 묻는 것에만 답한다. 같은 조의 다른 항이라도 질문과 관계없는 내용은 쓰지 않는다. 같은 말을 되풀이하지 않는다.
- 근거가 질문을 직접 답하지 못하면(다른 대상·다른 행위·근거에 없는 금액이나 기간) found=false로 한다.
- 근거가 "~할 수 있다"(재량)면 "~할 수 있습니다"로 쓰고, 반드시 받는다고 단정하지 않는다.
- 조항 번호는 본문에 쓰지 않는다. evidence에는 답에 실제로 쓴 근거 라벨만 넣는다.
- 질문이 특정 자치구(OO구)의 조례·규칙을 묻는 것이면 found=false.
결과는 JSON으로 낸다."""

def _schema(hits):
    """근거 고르기 칸. label 방식이면 이번에 넘긴 근거 라벨만 고를 수 있게 enum으로 묶습니다."""
    if config.EVIDENCE_KEY == "index":
        item = {"type": "integer"}
    else:
        item = {"type": "string", "enum": list(dict.fromkeys(h["label"] for h in hits))}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "law_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "found": {"type": "boolean"},
                    "answer": {"type": "string"},
                    "evidence": {"type": "array", "items": item},
                    **({"claims": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"subject": {"type": "string"}, "label": item, "ho": {"type": "string"},
                                       "kind": {"type": "string", "enum": ["면제", "감면", "대상 아님", "허용", "금지", "의무", "금액", "절차", "기타"]}},
                        "required": ["subject", "label", "ho", "kind"], "additionalProperties": False}}}
                       if CLAIM_CHECK else {}),
                },
                "required": ["found", "answer", "evidence"] + (["claims"] if CLAIM_CHECK else []),
                "additionalProperties": False,
            },
        },
    }


AMEND_TAG = re.compile(r"\s*<(?:개정|신설|전문개정)[^>]*>|\s*\[(?:본조신설|전문개정|제목개정|종전|제\d+조에서)[^\]]*\]")


def _related_items(hit, chunks, sem, question):
    """호가 많은 항에서 질문과 관련된 호 번호를 고릅니다. 조항·주제를 지정하지 않는 계산입니다.

    1) 글자 2개 단위 겹침: 그 항의 여러 호에 두루 나오는 글자쌍(할인·주차처럼 목록 공통어)과 항 머리말의 글자쌍은 빼고,
       드문 글자쌍만 idf로 셉니다. 질문에 대상이 여럿이면(다둥이 + 국가유공자) 가장 잘 맞는 호를 고른 뒤
       그 호가 설명한 글자쌍을 지우고 다음 호를 고르는 식으로 대상마다 하나씩 짚습니다.
    2) 의미(임베딩): 호 단위 보조 조각 점수 1위가 2위보다 분명히 높으면 그 호도 더합니다 ('경차' ↔ '경형자동차').
    """
    subs = _sub_ids(chunks, hit["id"])
    if len(subs) < RELATED_MIN_ITEMS:
        return []
    items = []                                    # (번호, 그 호 줄, 보조 조각 번호)
    for i in subs:
        ln = chunks[i]["text"].splitlines()[-1].strip()
        if HO_LINE.match(ln):
            items.append((ln.split(".", 1)[0], ln, i))
    if len(items) < RELATED_MIN_ITEMS:
        return []
    _, _, _, df, _ = _load()
    n = len(chunks)
    head = set(_bigrams(hit["text"].split("\n1.", 1)[0]))
    local = Counter(g for _, ln, _ in items for g in set(_bigrams(ln)))
    common = {g for g, c in local.items() if c >= max(2, len(items) * RELATED_COMMON)}
    q = set(_bigrams(question)) - head - common
    picked = []
    for _ in range(RELATED_KEEP):
        best, best_sc, best_g = None, 0.0, set()
        for num, ln, _i in items:
            if num in picked:
                continue
            g = q & set(_bigrams(ln))
            sc = sum(math.log(1 + (n - df[x] + 0.5) / (df[x] + 0.5)) for x in g)
            if sc > best_sc:
                best, best_sc, best_g = num, sc, g
        if best is None or best_sc < RELATED_MIN_LEX:
            break
        picked.append(best)
        q -= best_g
    ranked = sorted(items, key=lambda t: -sem[t[2]])
    if len(ranked) > 1 and sem[ranked[0][2]] - sem[ranked[1][2]] >= RELATED_SEM_GAP and ranked[0][0] not in picked:
        picked.append(ranked[0][0])
    return picked[:RELATED_KEEP]


def _context(hits, question=""):
    def body(t):          # 개정 이력 꼬리표는 답에 쓸 일이 없어 빼고 넣을 수 있게 합니다 (CLEAN_AMEND)
        t = AMEND_TAG.sub("", t) if config.CLEAN_AMEND else t
        return t
    # 면제·감면 목록인데 질문한 대상(다른 목록에는 실제로 있는 이름)이 이 목록 호에는 없으면 그렇다고 적어 줍니다.
    # ("다둥이 행복카드 → 운동시설 이용료 감면?"에 운동시설 감면 목록에 다둥이를 넣어 답하던 일)
    missing, all_in = {}, set()
    if question and CLAIM_CHECK:
        try:
            subs = _question_subjects(question, hits)
            lst = _list_lists(hits)
            for lb, core, _, _ in lst:
                # 같은 요금의 다른 목록(입장료 감면 ↔ 입장료 면제)에 있는 대상은 그 요금 혜택이 있으므로 '없음' 표시를 하지 않음
                same_fee = {l2 for l2, c2, _, _ in lst if c2 == core}
                gone = [w for w in subs if not any(l[0] in same_fee for l in _locate(w, hits))]
                if gone:
                    missing[lb] = gone
                elif len(subs) >= 2:
                    all_in.add(lb)            # 질문한 대상이 모두 같은 목록에 있음 (같은 요금에 대한 혜택)
        except Exception:  # noqa: BLE001  표시는 보조 정보라 실패해도 근거는 그대로
            missing, all_in = {}, set()
    def rel(h):
        r = h.get("related") if RELATED_ITEMS else None
        note = f"\n(질문과 관련된 호: {', '.join(n + '호' for n in r)})" if r else ""
        if h["label"] in missing:
            note += f"\n(질문한 대상 {', '.join(repr(w) for w in missing[h['label']])}은(는) 이 목록의 호에 없음)"
        # ('모두 이 목록에 있음' 표시는 "겹쳐 적용은 정해져 있지 않다" 꼬리를 늘려 장애인+다둥이가 7/10으로 떨어져 넣지 않음)
        return note
    if config.EVIDENCE_KEY == "index":
        return "\n\n".join(f"[근거 {i}] {h['label']}\n{body(h['text'])}{rel(h)}" for i, h in enumerate(hits, 1))
    return "\n\n".join(f"[근거: {h['label']}]\n{body(h['text'])}{rel(h)}" for h in hits)


Q_TIME = re.compile(r"(\d+)\s*(분|시간)")                      # 질문의 "10분당 / 30분 세우면 / 1시간"
BASE_TIME = re.compile(r"(\d+)?\s*(분|시간)\s*당")                # 근거 요금표의 "5분당 / 1시간당 / 시간당"


def _unit_ratios(question, hits):
    """질문 시간(분)과 근거 요금표 단위(분)들. 금액·단위를 묻는 질문이 아니면 (None, [])."""
    m = Q_TIME.search(question)
    if not m or not (AMOUNT_Q.search(question) or UNIT_Q.search(question) or COST_Q.search(question)):
        return None, []
    q = int(m.group(1)) * (60 if m.group(2) == "시간" else 1)
    bases = {int(n or 1) * (60 if u == "시간" else 1)
             for h in hits for n, u in BASE_TIME.findall(h["text"])}
    return q, sorted(bases)


def _tlabel(mins):
    return f"{mins // 60}시간" if mins % 60 == 0 else f"{mins}분"


def _unit_hint(question, hits, convert=False):
    """질문 시간과 근거 요금 단위를 비교한 계산 안내 (답변 생성용).
    "공영주차장 10분당 요금은?" (근거 5분당)
        - 웹 화면(convert=True): 2배로 환산해 "10분 1,000원"으로 답하라.
        - 제출용(convert=False): 환산하지 말고 "10분 단위가 아니라 5분당 500원"으로 조례 금액 그대로 답하라.
    "자전거 주차장 10분에 얼마?" (근거 1시간당) → 둘 다 나누지 말고 단위를 바로잡아라.
    프롬프트 규칙만으로는 서로의 예시에 끌려가 흔들려서, 이 질문에 맞는 쪽만 알려 줍니다."""
    q, bases = _unit_ratios(question, hits)
    lines = []
    for b in bases:
        if b < q and q % b == 0 and convert:
            lines.append(f"- 근거의 '{_tlabel(b)}당' 요금: 질문 시간 {_tlabel(q)}은 {q // b}배이다. 그 요금마다 {q // b}를 곱한 "
                         f"{_tlabel(q)} 금액을 첫 문장에 쓰고, '{_tlabel(b)}당 얼마'는 뒤 문장에 덧붙인다. 바로잡기만 하지 않는다.")
        elif b < q and q % b == 0:
            lines.append(f"- 근거의 '{_tlabel(b)}당' 요금: 환산하지 말고 근거 금액을 그대로 '{_tlabel(b)}당 ○원'으로 쓴다. "
                         f"첫 문장은 '{_tlabel(q)} 단위가 아니라 {_tlabel(b)}당 기준입니다'처럼 바로잡는다.")
        elif b > q:
            lines.append(f"- 근거의 '{_tlabel(b)}당' 요금: 질문 시간 {_tlabel(q)}보다 길다. 나누어 계산하지 말고 "
                         f"'{_tlabel(q)} 단위가 아니라 {_tlabel(b)}당 ○원'으로 바로잡는다.")
    return "\n\n[계산 안내]\n" + "\n".join(lines) if lines else ""


def _unit_note(question, hits):
    """검증기용 계산 근거. 5분당 500원 × 12 = 6,000원처럼 환산한 금액을 '근거에 없는 숫자'로 떨어뜨리지 않게 합니다."""
    q, bases = _unit_ratios(question, hits)
    lines = [f"- 근거의 '{_tlabel(b)}당' 요금 × {q // b} = {_tlabel(q)} 요금. 이렇게 환산한 금액은 근거에서 계산한 값이다."
             for b in bases if b < q and q % b == 0]
    return "\n\n[계산 근거]\n" + "\n".join(lines) if lines else ""


# 질문에 없는 시설을 다룬 문장 빼기: (답 문장 속 시설, 질문에 이 말이 있으면 물은 것, 그 시설의 근거 라벨)
#   "임산부 혜택은?" → "공공자전거 이용요금도 감면받을 수 있습니다"(감면 대상에 임산부 없음, 틀림) 또는
#                      "공공자전거 감면 대상에는 임산부가 없습니다"(묻지 않은 내용)
#   "임산부가 이용할 수 있는 주차장?" → "장애인전용주차구획은 「장애인·노인·임산부 등의 …법률」에 따라 설치"
#                      (법률 이름의 '임산부' 때문에 임산부도 쓸 수 있는 것처럼 읽힘)
#   "장애인 혜택?" → "자전거주차장·보관소 이용 시 요금 할인"(등록 자전거 혜택, 자전거 조례 제15조 제2항, 장애인 아님)
#   "임산부 주차장?" → "확장형 주차구획 50% 이상"(설치 기준, 묻지 않은 내용)
UNASKED_FACILITY = [
    (re.compile(r"자전거|따릉이"), re.compile(r"자전거|따릉이"), re.compile(r"자전거이용 활성화에 관한 조례")),
    # "장애인용 경사로에 가장 가까운 위치"는 장애인전용주차구획 위치 기준(제25조 제2항)인데 가족배려주차장 설명에 섞인 일이 있음
    (re.compile(r"장애인\s*전용\s*주차|장애인용\s*경사로"), re.compile(r"장애인"),
     re.compile(r"주차장 설치 및 관리 조례 제25조(?:\s|$)")),
    (re.compile(r"확장형"), re.compile(r"확장형"), re.compile(r"주차장 설치 및 관리 조례 제25조의2 제3항")),
    # "자전거도로가 있으면 반드시 써야 해?"에 도로교통 법령 준수·안전장비(자전거 제4조 제2항),
    # 자동차운전자의 배려 의무(제4조 제4항)를 덧붙였던 일
    (re.compile(r"도로교통에\s*관한\s*법령을\s*(준수|지)|보행자에(게)?\s*(위해|주의)|안전장비"),
     re.compile(r"법령|법규|교통법|보행자|안전|사고|헬멧|장비|주의|지켜야"),
     re.compile(r"자전거이용 활성화에 관한 조례 제4조 제2항"), "cited_whole"),
    (re.compile(r"자전거이용시설이\s*설치된\s*경우|해당\s*시설을\s*이용하도록"),
     re.compile(r"도로|전용|이용시설|차도|인도|길로|노선|다녀|다니|통행|달려"),
     re.compile(r"자전거이용 활성화에 관한 조례 제4조 제6항"), "cited_whole"),
    (re.compile(r"자동차\s*운전자|자동차를\s*운전"), re.compile(r"자동차|운전자|차량|배려|양보|거리"),
     re.compile(r"자전거이용 활성화에 관한 조례 제4조 제4항"), "cited_whole"),
    # "공원에 동물이 있다면 어떻게 해야 해?" → 맹견을 묻지 않았는데 맹견 출입 금지, 외출 의무(목줄·인식표·배설물),
    # 공원 반려동물 공간 설치 노력까지 덧붙였던 일. 질문에 그 대상·상황이 있을 때만 남깁니다.
    (re.compile(r"맹견"), re.compile(r"맹견|사나운|무서운|공격|물(?:면|었|어)|위험한\s*개"),
     re.compile(r"동물보호 조례 제7조의[23]")),
    (re.compile(r"목줄|인식표|배설물|외출할\s*때|동반하고\s*외출"),
     re.compile(r"산책|외출|데리고|데려|목줄|인식표|배설물|동반|키우|반려견과|우리\s*(?:개|강아지)|내\s*(?:개|강아지)"),
     re.compile(r"동물보호 조례 제7조 제3항")),
    (re.compile(r"머무를\s*수\s*있는\s*공간|목줄\s*등을\s*걸어"), re.compile(r"화장실|매점|공간|머무|걸어"),
     re.compile(r"도시공원 조례 제22조의2 제2항")),
    # "공원에 맹견이 있다면?"에 "유기되거나 학대받는 동물을 발견했다면 구청장에게 신고…"(동물보호 제10조)를 덧붙였던 일.
    # 유기·학대·발견·신고를 묻거나 동물 일반이 '있다면/발견하면'을 물었을 때만 남깁니다.
    #   (제10조가 인용됐을 때만, 주어를 공유하는 뒤 절까지 문장째 뺍니다 — F25 위탁 보호의 '동물보호센터' 문장은 건드리지 않게)
    (re.compile(r"유기되거나|학대받는|유기동물|피학대"),
     re.compile(r"유기|버려|버린|학대|주인\s*없|떠돌|길\s*잃|구조|신고|발견|동물이\s*있|동물을\s*보"),
     re.compile(r"동물보호 조례 제10조"), "cited_whole"),
]


# 그 항에만 나오는 낱말. 답이 이 낱말을 안 쓰면 그 항은 답과 무관한 덤 인용입니다.
#   "임산부 주차장?" → 제25조의2 제3항(확장형 주차구획 50%), "임산부 혜택?" → 도시공원 제20조 제2항(기후동행카드·단체)·제3항(운동시설)
OWN_TERMS = ["확장형", "운동시설", "기후동행카드", "30명 이상의 단체"]


def _label_text(label):
    """같은 라벨의 조각을 모두 이은 글 (항이 호별 조각으로 나뉘어 있어도 항 전체를 봅니다)."""
    return " ".join(c["text"] for c in _load()[0] if c["label"] == label)


def _drop_offtopic_paragraphs(ans, used):
    def own(h):
        if "별표" in h["label"]:            # 별표는 금액표라 빼지 않습니다 (U10 운동시설 이용료 = 도시공원 별표 2)
            return True
        full = _label_text(h["label"]) or h["text"]
        terms = [t for t in OWN_TERMS if t in full]
        return not terms or any(t in ans for t in terms)
    rest = [h for h in used if own(h)]
    return rest or used


# "임산부 혜택은?"처럼 한 사람의 혜택을 나열하는 질문: 그 사람이 조문에 없는 조항은 그 사람의 혜택이 아닙니다.
#   웹 "운동시설 이용료는 50% 범위에서 감면" (도시공원 제20조 제3항: 6세 미만·65세 이상·장애인 등, 임산부 없음 → 틀림)
#   제출 "청년 주거안정 대책은 임산부 개인 혜택으로 정해져 있지 않습니다" (청년 조례, 묻지 않은 내용)
LIST_Q = re.compile(r"혜택|받을\s*수\s*있는\s*(것|게|건)|뭐가\s*있|무엇이\s*있|어떤\s*(게|것이)\s*있")
PERSON_GROUPS = [("임산부",), ("장애인",), ("다자녀", "다둥이"), ("국가유공자", "유공자"), ("청소년",),
                 ("65세", "어르신", "노인"), ("한부모",), ("수급자", "기초생활")]
NEG_SENT = re.compile(r"(않습니다|없습니다|아닙니다)[.]?\s*$")
SETUP_SENT = re.compile(r"(기준|규정)에\s*(맞게|따라|적합하게)\s*설치|설치(됩니다|해야\s*합니다|하여야\s*합니다)")
SETUP_LAW = re.compile(r"주차장 설치 및 관리 조례 제25조(?:\s|$)")


def _without_law_names(text):
    """「장애인ㆍ노인ㆍ임산부 등의 편의증진 보장에 관한 법률」처럼 법률 이름 속 낱말은 대상이 아니라서 빼고 봅니다."""
    return re.sub(r"「[^」]*(?:법률|법|조례|시행령|시행규칙|규칙|령)」", "", text)   # 「다둥이 행복카드」는 남깁니다


def _person_terms(question):
    return next((g for g in PERSON_GROUPS if any(t in question for t in g)), None)


# 감면·이용 대상을 나열한 항. 질문한 사람이 이 목록에 없으면 그 사람의 혜택이 아닙니다.
#   (대상 목록이 아닌 항은 사람 이름이 없어도 다른 항과 이어져 그 사람의 혜택일 수 있어 건드리지 않습니다.
#    예: F04 동물보호 제25조의2 제2항 '진료비 지원'은 수급자가 제5항에만 나옴)
ELIGIBILITY_LIST = re.compile(r"도시공원 조례 제20조 제[123]항|주차장 설치 및 관리 조례 제7조 제1항|"
                              r"자전거이용 활성화에 관한 조례 제12조의3 제2항|동물보호 조례 제28조")
#   (가족배려주차장 제25조의2 제2항은 넣지 않음: U35 "가족배려주차장에 장애인이 주차해도 되나요?"의 정답이 바로 이 항)
LAW_TOPIC = {"주차장": r"주차", "동물보호": r"동물|반려|개|고양이", "자전거": r"자전거|따릉이",
             "도시공원": r"공원", "청년": r"청년", "금연": r"담배|흡연|금연"}


def _hangul_words(s):
    return {re.sub(r"(은|는|이|가|을|를|에|의|도|로|과|와|에서)$", "", w) for w in re.findall(r"[가-힣]{2,}", s)}


def _drop_other_person_benefits(question, ans, used):
    terms = _person_terms(question)
    if not terms:
        return ans, used
    listing = LIST_Q.search(question)
    texts = {h["label"]: _label_text(h["label"]) or h["text"] for h in used}
    has = lambda t: any(w in _without_law_names(t) for w in terms)
    drop = {lb for lb, t in texts.items() if ELIGIBILITY_LIST.search(lb) and not has(t)}
    if listing:      # 혜택 나열 질문: 그 사람이 한 번도 안 나오는 조례(질문 주제도 아님)의 인용은 덤입니다
        for law, topic in LAW_TOPIC.items():
            labs = [lb for lb in texts if law in lb.split("조례")[0]]
            if labs and not any(has(texts[lb]) for lb in labs) and not re.search(topic, question):
                drop.update(labs)
    if len(drop) == len(texts):
        drop = set()
    if not drop and not listing:
        return ans, used
    kept_txt = " ".join(t for lb, t in texts.items() if lb not in drop) + " " + question
    own = {w for w in set().union(set(), *(_hangul_words(texts[lb]) for lb in drop)) if w not in kept_txt}
    parts = re.split(r"((?<=[.!?])\s+)", ans)
    sents, seps = parts[0::2], parts[1::2] + [""]
    # 남의 혜택 문장, 그리고 혜택 나열 질문의 "~는 해당하지 않습니다"·"~기준에 맞게 설치됩니다"(혜택이 아닌 설치 기준) 문장을 뺍니다
    keep = [(s, sep) for i, (s, sep) in enumerate(zip(sents, seps))
            if not (_hangul_words(s) & own)
            and not (listing and i > 0 and (NEG_SENT.search(s.strip()) or SETUP_SENT.search(s)))]
    if any(s.strip() for s, _ in keep):
        ans = "".join(s + sep for s, sep in keep).strip()
    if listing:      # 설치 기준 문장을 뺐으면 그 기준 조항(장애인전용주차구획 설치기준 제25조)도 뺍니다
        drop.update(lb for lb in texts if SETUP_LAW.search(lb) and not re.search(r"전용\s*주차", ans))
        if len(drop) == len(texts):
            return ans, used
    return ans, [h for h in used if h["label"] not in drop]


# "임산부가 이용 가능한 주차장은?" "임산부를 위한 주차구역이 있나요?"처럼 이용·존재를 물으면
# 이용대상(제25조의2 제2항)만 답합니다. 설치 의무 비율(제1항: 30대 이상 10%)·확장형(제3항)·설치 위치(제4항)는
# 설치 기준이고, 요금 할인(제7조)은 요금 질문이 아니면 묻지 않은 내용이라 문장(절)과 인용을 뺍니다.
# 설치·비율·기준·위치나 요금·할인을 함께 물으면 그쪽은 그대로 둡니다.
RULE_CLAIM = re.compile(r"(높은|큰|유리한)\s*(것|쪽)?\s*하나만|하나만\s*적용")          # 답이 말한 중복 제한 규칙
RULE_TEXT = re.compile(r"하나만|높은\s*것|중복\s*(하여|해서)?\s*(감면|적용)하지")      # 조문에 실제 중복 제한 문구
TIME_A = re.compile(r"\d+\s*(?:년|개월|달|월|주|일|시간|분)")   # 답에 실제 기간·시점
WHEN_Q = re.compile(r"언제|시기|기간|며칠|몇\s*(시|일|월|년)|마감")
NO_WHEN = re.compile(r"(시기|기간|일정|날짜|기한|시간대|언제)[^.]{0,25}(명시되어\s*있지\s*않|정해져\s*있지\s*않|나와\s*있지\s*않|규정되어\s*있지\s*않|적혀\s*있지\s*않|정하고\s*있지\s*않|없습니다|없으며|없고)")
NOT_LISTED = re.compile(r"^(?!\s*아니요)[^.]*?(근거|조례)(에서|가|에)\s*정한[^.]{0,40}(아닙니다|않습니다)")   # "아니요, …" 결론 문장은 제외
def _law_name_re():
    try:
        names = [o["name"] for o in json.loads((config.DATA / "ordinances.json").read_text(encoding="utf-8"))]
    except Exception:  # noqa: BLE001
        names = []
    short = sorted({re.sub(r"^서울특별시\s*", "", n) for n in names}, key=len, reverse=True)
    alts = [re.escape(n).replace("\\ ", "\\s*") for n in short]
    return "(?:" + "|".join(alts) + ")" if alts else r"[가-힣·ㆍ]+\s?조례"


_LAW_NAME_RE = _law_name_re()   # 본문의 "도시공원 조례 제20조에 따라"에서 조례 이름만 정확히 잡도록 (앞 문장까지 지우던 일)
ART_SUBJ = re.compile(r"(?:「[^」]*」\s*)?(?:서울특별시\s*)?" + _LAW_NAME_RE + r"\s*제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?\s*(은|는|에서는|에는|에서)")
GENERIC_BASIS = re.compile(r"도시공원 조례 제5조 제1항$")
ADVICE = re.compile(r"(확인|문의|참고|안내를\s*받)\s*하?(시는|는)\s*것이\s*(좋|바람직)|(확인|문의|참고)하세요|(확인|문의)해\s*보세요"
                    r"|(확인|문의)하시기\s*바랍니다|(확인|문의)하셔야\s*합니다|(확인|문의)해\s*보시는\s*것이|(확인|문의)이?\s*필요합니다")
# 답 본문의 "도시공원 조례 제20조제1항제2호에 따라 " 같은 조항 번호 인용 (인용은 citation에 따로 붙음)
LAW_REF = re.compile(r"(^|(?<=[.!?])\s+)(?:「[^」]*조례」|(?:서울특별시\s*)?" + _LAW_NAME_RE + r")에\s*(?:따르면|의하면)\s*,?\s*")   # 문장 첫머리 "○○ 조례에 따르면, "
INFER = re.compile(r"^\s*따라서[^.]*(?:위반한|위반하는|위반이\s*되는|위반에\s*해당하는)\s*것입니다")   # 근거에 없는 결론 추론
ART_REF = re.compile(r"(?:「[^」]*」\s*)?(?:(?:서울특별시\s*)?" + _LAW_NAME_RE + r"\s*)?제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?에\s*(?:따라|따르면|의하면)\s*,?\s*")
NO_SUCH_FACILITY = re.compile(r"전용\s*주차\S*\s*(?:은|는|이|가)?\s*(?:따로|별도로|별도)?\s*(?:없|정해져\s*있지\s*않|두고\s*있지\s*않|규정되어\s*있지\s*않)")
USE_Q = re.compile(r"이용\s*(가능|할\s*수\s*있|해도\s*되)|있나요|있어요|있습니까|있을까요|있는지|되나요|돼요")
FEE_Q = re.compile(r"요금|할인|감면|면제|돈|얼마|무료|비용|혜택|입장료|이용료|사용료|수수료|보상|내야|내도|안\s*내")
INSTALL_Q = re.compile(r"설치|만들|몇\s*(퍼센트|%|대|면|개)|비율|기준|위치|어디에|확장형|규모")
# "65세 도시공원 이용 가능한가요?" → 입장료 면제·운동시설 감면(도시공원 제20조)은 묻지 않은 내용
UNASKED_FACETS = [   # (질문에 이 말이 있으면 물은 것, 답 문장 속 표현, 그 내용의 인용 라벨)
    (FEE_Q, re.compile(r"할인|감면|면제|입장료|이용료|사용료|주차요금"),
     re.compile(r"주차장 설치 및 관리 조례 제7조|도시공원 조례 제(?:15|20)조|도시공원 조례 별표 [12]$|"
                r"자전거이용 활성화에 관한 조례 제12조의3|동물보호 조례 제28조")),
    (INSTALL_Q, re.compile(r"설치|사각|폐쇄회로|CCTV|승강기|계단|인접한|확장형|통행이\s*잦|총\s*주차대수"),
     re.compile(r"주차장 설치 및 관리 조례 제25조의2 제[134]항")),
]


# "무서운 개가 다른 사람에게 피해를 준다면?"처럼 피해·사고의 결과를 물으면 결과(보험·보상·격리조치·과태료)만 답합니다.
# 출입 금지 장소 같은 예방 의무(동물보호 제7조의3)는 출입·장소를 묻지 않았으면 문장(절)과 인용을 뺍니다.
# (H09 "맹견을 데리고 어린이공원에 들어가도 되나요?"는 출입을 물었으므로 그대로)
CONSEQ_Q = re.compile(r"(피해|다치|물(?:면|었|어서)|사고|위반|공격)")
CONSEQ_FACETS = [   # (질문에 이 말이 있으면 물은 것, 답 문장 속 표현, 인용 라벨)
    (re.compile(r"출입|들어가|데려가|데리고|장소|어디"), re.compile(r"출입하지\s*(?:않|아니)|출입\s*금지|출입을\s*금지"),
     re.compile(r"동물보호 조례 제7조의3")),
]


# "주차장에 전기자전거 충전소 설치 가능해?"
#   ① 자전거를 물었는데 전기자동차 충전시설 화재 예방(주차장 제9조의2)을 덧붙임 → 다른 대상
#   ② 설치를 물었는데 "충전 외 다른 목적으로 사용 금지"(자전거 제10조의2 제3항) 같은 사용 규칙을 덧붙임
OTHER_VEHICLE = re.compile(r"전기\s*자동차|전기차|자동차\s*충전")
USE_RULE = re.compile(r"충전을\s*위한\s*경우|충전\s*외의|다른\s*목적으로\s*사용")
MAINT_RULE = re.compile(r"상시\s*점검|점검\s*[·ㆍ]?\s*보수|유지\s*[·ㆍ]?\s*관리")


def _drop_other_object(question, ans, used):
    if "자전거" in question and not re.search(r"자동차|전기차|차량|승용", question):
        new = _remove_sentences(ans, OTHER_VEHICLE, clauses=False)   # 주어(관리자)를 공유하는 뒤 절도 같은 대상이라 문장째
        rest = [h for h in used if not re.search(r"주차장 설치 및 관리 조례 제9조의2", h["label"])]
        if new and rest:
            ans, used = new, rest
    if re.search(r"설치|만들|둘\s*수", question) and not re.search(r"사용|이용|세워|세우|주차해|주차할", question):
        new = _remove_sentences(ans, USE_RULE)
        rest = [h for h in used if not re.search(r"자전거이용 활성화에 관한 조례 제10조의2 제3항", h["label"])]
        if new and rest:
            ans, used = new, rest
    # 설치 가능 여부를 물었는데 설치 뒤의 관리 의무("상시 점검·보수하고 … 유지·관리해야", 제10조의2 제2항)를 덧붙인 경우
    if re.search(r"설치|만들|둘\s*수", question) and not re.search(r"관리|점검|보수|유지", question) \
            and any(re.search(r"자전거이용 활성화에 관한 조례 제10조의2 제2항", h["label"]) for h in used):
        new = _remove_sentences(ans, MAINT_RULE, clauses=False)
        rest = [h for h in used if not re.search(r"자전거이용 활성화에 관한 조례 제10조의2 제2항", h["label"])]
        if new and rest:
            ans, used = new, rest
    return ans, used


def _dedupe_clauses(ans):
    """앞에서 이미 말한 내용과 거의 같은 절은 뺍니다(낱말 75% 이상이 앞 문장에 이미 나옴, 낱말 3개 이상인 절만).
    예) "…구조한 동물은 동물보호센터에서 보호합니다. 구조할 때는 …사용하고, 구조한 동물은 동물보호센터에 보호조치합니다." """
    parts = re.split(r"((?<=[.!?])\s+)", ans)
    sents, seps = parts[0::2], parts[1::2] + [""]
    seen, out = set(), []
    for s, sep in zip(sents, seps):
        words_of = lambda c: {w for w in _hangul_words(c) if len(w) >= 2}
        clauses = re.split(r"(?<=[고며]),\s*", s)
        dup = [c for c in clauses if len(words_of(c)) >= 3 and len(words_of(c) & seen) / len(words_of(c)) >= 0.75]
        if seen and dup and len(dup) == len(clauses):
            continue                                       # 문장 전체가 반복
        for c in dup if seen else []:
            t = _trim_clause(s, re.compile(re.escape(c)))
            if t:
                s = t
        out.append((s, sep))
        seen |= words_of(s)
    return "".join(s + sep for s, sep in out).strip() or ans


def _drop_removed_citations(kept, removed, used):
    """편집 단계에서 뺀 문장에만 쓰인 인용을 뺍니다.

    첫 근거가 이미 가진 낱말을 빼고, 그 근거만 가진 낱말(3글자 이상)이 뺀 문장에는 있는데 남은 답에는 없으면
    그 근거는 뺀 문장의 근거입니다. 예) 맹견 출입 답에서 보상·격리 문장을 빼면 그 문장의 근거(격리·보험 항)도 뺌.
    """
    if len(used) < 2:
        return used
    def words(h):
        text = re.sub(r"\[[^\]]*\]", " ", _label_text(h["label"]) or h["text"])
        return {w for w in _hangul_words(_without_law_names(text)) if len(w) >= 3}
    base = words(used[0])
    kf, rf = re.sub(r"\s+", "", kept), re.sub(r"\s+", "", removed)
    out = [used[0]]
    for h in used[1:]:
        own = words(h) - base
        if "별표" not in h["label"] and any(w in rf for w in own) and not any(w in kf for w in own):
            continue
        out.append(h)
    return out


_LAW_KEYS = {}


def _law_keys(law):
    """조례 이름에서 분야어를 자동으로 뽑습니다. 예) 주차장 설치 및 관리 조례 → 주차, 도시공원 조례 → 공원.
    여러 조례 이름에 두루 나오거나 일반적인 말(이용·보호·환경·관리·설치 등)은 뺍니다."""
    if not _LAW_KEYS:
        laws = {c.get("law", "") for c in _load()[0] if c.get("law")}
        generic = {"서울", "특별", "조례", "관한", "설치", "관리", "이용", "보호", "환경", "조성", "기본", "방지", "활성", "피해"}
        cand = {}
        for lw in laws:
            keys = set()
            for w in re.findall(r"[가-힣]{2,}", lw.replace("서울특별시", "")):
                keys |= {w[:2]} | ({w[-2:]} if len(w) >= 4 else set())
            cand[lw] = {k for k in keys if k not in generic}
        for lw, keys in cand.items():
            _LAW_KEYS[lw] = {k for k in keys if sum(k in other for other in cand.values()) == 1}
    return _LAW_KEYS.get(law, set())


def _drop_other_law_citations(ans, used):
    """첫 근거와 다른 조례의 인용인데, 답 본문에 그 조례의 분야어가 하나도 없으면 뺍니다.
    예) 공원 입장료 답에 주차장 조례 중복감면 항이 인용만 붙어 있던 일 (본문엔 '주차' 얘기가 없음)."""
    if len(used) < 2:
        return used
    first_law = used[0].get("law")
    out = [used[0]]
    for h in used[1:]:
        keys = _law_keys(h.get("law", ""))
        if h.get("law") != first_law and keys and not any(k in ans for k in keys):
            continue
        out.append(h)
    return out


def _cap_citations(ans, used):
    """인용이 CITE_HARD_MAX개를 넘으면 첫 근거(모델이 고른 대표 근거)와, 답 본문에 그 근거만의 낱말이 가장 많이
    들어 있는 근거 순으로 남깁니다. 답에 금액·비율이 있으면 별표(요금표)를 먼저 남깁니다. 남은 것의 순서는 그대로."""
    uniq = list({h["label"]: h for h in used}.values())
    if CITE_HARD_MAX <= 0 or len(uniq) <= CITE_HARD_MAX:
        return used
    flat = re.sub(r"[^가-힣]", "", ans)          # 조문 쪽과 같게 한글만 남겨 비교
    amount = bool(re.search(r"\d\s*(원|%|퍼센트)|무료|면제", ans))

    def words(h):
        # 글자 3개 묶음으로 비교 (낱말 비교는 '고시하여야'↔'고시해야' 같은 활용형 차이에 약했음)
        text = re.sub(r"\[[^\]]*\]", " ", _label_text(h["label"]) or h["text"])
        t = re.sub(r"[^가-힣]", "", _without_law_names(text))
        return {t[i:i + 3] for i in range(len(t) - 2)}
    base = words(uniq[0])
    scored = []
    for k, h in enumerate(uniq[1:], 1):
        w = words(h)
        scored.append(((amount and "별표" in h["label"]), sum(x in flat for x in w - base), sum(x in flat for x in w), -k, h["label"]))
    keep = {t[-1] for t in sorted(scored, reverse=True)[:CITE_HARD_MAX - 1]}
    return [uniq[0]] + [h for h in uniq[1:] if h["label"] in keep]


def _drop_unused_citations(ans, used):
    """답과 겹치는 낱말(3글자 이상)이 하나도 없는 인용은 덤입니다. 첫 근거와 별표는 항상 남깁니다.
    (H09 맹견 출입: 답은 두 문장인데 도시공원 제5조 제2~4항·제22조 등 무관한 인용이 11~12개 붙었음)
    겹치는 낱말이 하나라도 있으면 남겨 정답 조항을 잃지 않게 합니다(H19 '도시공원에서' ↔ 제22조 제1항)."""
    if len(used) < 3:
        return used
    flat = re.sub(r"\s+", "", ans)

    def shares_word(h):
        text = _label_text(h["label"]) or h["text"]
        text = re.sub(r"\[[^\]]*\]", " ", text)            # 머리의 [조례 이름 제N조(제목)] 빼기
        words = {w for w in _hangul_words(_without_law_names(text)) if len(w) >= 3}
        return any(w in flat for w in words)               # 조사 차이에 흔들리지 않게 '답 안에 들어 있는지'로 봅니다

    keep = [h for k, h in enumerate(used) if k == 0 or "별표" in h["label"] or shares_word(h)]
    if len(keep) > CITE_SOFT_MAX:
        def words(h):
            text = re.sub(r"\[[^\]]*\]", " ", _label_text(h["label"]) or h["text"])
            return {w for w in _hangul_words(_without_law_names(text)) if len(w) >= 3}
        seen, slim = set(), []
        for k, h in enumerate(keep):
            own = words(h) - seen
            if k == 0 or "별표" in h["label"] or any(w in flat for w in own):
                slim.append(h)
                seen |= words(h)
        keep = slim
    return keep or used


# 반대 방향: "공원에 맹견이 있다면?"처럼 피해·사고를 묻지 않았으면 피해가 난 뒤의 조치
# (신체적 피해 시 격리조치 제7조의2 제1항, 피해 보상 보험 제7조의2 제2항)는 묻지 않은 내용입니다.
HARM_ASKED = re.compile(r"피해|다치|물(?:면|었|어서|리)|사고|공격|격리|보험|보상|죽")
HARM_SENT = re.compile(r"신체적\s*피해|다치게|죽게|격리\s*조치|격리조치|보험|보상")


APPLIES_SENT = re.compile(r"(이에|이\s*규정에|여기에|그\s*안전장비에|안전장비에)\s*(해당합니다|해당됩니다|포함됩니다)")


def _drop_absent_inference(question, ans, used):
    """"헬멧 등 안전장비를 갖추는 것이 이에 해당합니다"처럼, 질문의 낱말(헬멧)이 인용 근거에 없는데
    근거의 넓은 말(안전장비)에 들어간다고 해석한 문장은 뺍니다 (근거에 없는 해석)."""
    if not APPLIES_SENT.search(ans):
        return ans
    cited = " ".join(_label_text(h["label"]) or h["text"] for h in used)
    absent = [w for w in re.findall(r"[가-힣]{2,}", question)
              if w[:2] not in cited and not re.search(r"(요|까|야|해|니다|는지|면|고|도|나)$", w)]   # 동사·어미는 빼고 명사만
    if not absent:
        return ans
    pat = re.compile(r"^(?=.*(?:" + "|".join(re.escape(w[:2]) for w in absent) + r")).*?" + APPLIES_SENT.pattern)
    return _remove_sentences(ans, pat, clauses=False) or ans


def _drop_unasked_harm(question, ans, used):
    if HARM_ASKED.search(question) or not any(re.search(r"동물보호 조례 제7조의2", h["label"]) for h in used):
        return ans, used
    new = _remove_sentences(ans, HARM_SENT, clauses=False)
    if new is None and HARM_SENT.search(ans):
        return ans, used
    rest = [h for h in used if not re.search(r"동물보호 조례 제7조의2", h["label"])]
    if rest:
        return (new or ans), rest
    return ans, used


def _drop_unasked_prevention(question, ans, used):
    if not CONSEQ_Q.search(question):
        return ans, used
    for asked, sent_pat, label_pat in CONSEQ_FACETS:
        if asked.search(question) or not any(label_pat.search(h["label"]) for h in used):
            continue
        new = _remove_sentences(ans, sent_pat)
        if new is None:
            if sent_pat.search(ans):
                continue
            new = ans
        rest = [h for h in used if not label_pat.search(h["label"])]
        if rest:
            ans, used = new, rest
    return ans, used


def _drop_unasked_facets(question, ans, used):
    if not USE_Q.search(question):
        return ans, used
    for asked, sent_pat, label_pat in UNASKED_FACETS:
        if asked.search(question) or not any(label_pat.search(h["label"]) for h in used):
            continue
        new = _remove_sentences(ans, sent_pat)
        if new is None:
            if sent_pat.search(ans):          # 지울 수 없는 문장(답 전체)이면 그대로 둡니다
                continue
            new = ans                         # 답이 그 내용을 안 썼으면 인용만 덤입니다 (65세 공원 이용 → 제20조)
        rest = [h for h in used if not label_pat.search(h["label"])]
        if rest:
            ans, used = new, rest
    return ans, used


def _remove_sentences(ans, pat, clauses=True):
    """pat이 든 문장을 뺍니다. 다른 내용과 한 문장이면 그 절만 뺍니다(clauses=False면 문장 전체).
    다 지워지거나 바뀐 게 없으면 None."""
    parts = re.split(r"((?<=[.!?])\s+)", ans)
    sents, seps = parts[0::2], parts[1::2] + [""]
    keep = []
    for s, sep in zip(sents, seps):
        if not pat.search(s):
            keep.append((s, sep))
        elif clauses and (t := _trim_clause(s, pat)):
            keep.append((t, sep))
    if keep == list(zip(sents, seps)) or not any(s.strip() for s, _ in keep):
        return None
    return "".join(s + sep for s, sep in keep).strip()


DELEGATE_SENT = re.compile(r"따로\s*정(합니다|해집니다|하며|하고)")


BOTH_OK = re.compile(r"(함께|같이|모두|둘\s*다|각각|동시에|중복(?:해서|하여)?)[^.]{0,30}?"
                     r"(받을|적용받을|해당할|적용될|이용할)\s*수\s*(?:는\s*)?있")


def _drop_contradicting_yes(ans):
    """'아니요'(결론: 하나만 적용)인데 앞부분이 "둘 다 받을 수 있다"고 말하면 그 앞부분을 뺍니다.
    "아니요, A와 B를 함께 받을 수 있습니다. 다만 … 하나만 적용됩니다" → "아니요, … 하나만 적용됩니다"
    "아니요, 두 할인 모두 받을 수 있지만, 같은 요금에서는 … 하나만 적용됩니다" → "아니요, 같은 요금에서는 …"
    낱말·조항을 지정하지 않고, 첫 문장의 긍정 단정과 뒤따르는 결론의 모양만 봅니다."""
    m = re.match(r"아니요,\s*", ans)
    if not m:
        return ans
    body = ans[m.end():]
    s1, *rest = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)
    rest = rest[0] if rest else ""
    sc = re.match(r"(\S+\s*의\s*경우)\s*,?\s*", s1)        # "주차요금의 경우"처럼 범위를 밝힌 말은 남깁니다
    scope = sc.group(1) + ", " if sc else ""
    # 한 문장 안: "…받을 수 있지만, <결론>"
    parts = re.split(r"(?<=있지만)\s*,?\s*|(?<=있으나)\s*,?\s*", s1, maxsplit=1)
    if len(parts) == 2 and BOTH_OK.search(parts[0]) and parts[1].strip():
        return "아니요, " + scope + parts[1].strip() + (" " + rest if rest else "")
    # 따로 선 문장: "…받을 수 있습니다." + "다만 <결론>"
    # "받을 수 있습니다."처럼 대상 없이 긍정만 한 짧은 첫 문장도 같습니다
    bare_ok = re.fullmatch(r"\s*(?:둘\s*다\s*)?(?:받을|적용받을|해당할)\s*수\s*(?:는\s*)?있습니다\.?\s*", s1)
    if (BOTH_OK.search(s1) or bare_ok) and not re.search(r"없|않|못|하나만", s1) and rest:
        return "아니요, " + scope + re.sub(r"^(?:다만|그러나|하지만)\s*,?\s*", "", rest)
    return ans


def _drop_ungrounded_delegation(ans, used):
    """"감면율과 운영 방식은 시장이 따로 정합니다" 문장은 인용 조항에 '따로 정'이 있을 때만 남깁니다.
    "장애인 혜택?" 답에 공공자전거 조항(제12조의3 제3항)의 문구가 섞여, 조례가 80%로 정한 주차 할인까지
    시장이 정하는 것처럼 읽힌 일이 있었습니다."""
    if not DELEGATE_SENT.search(ans) or any(re.search(r"따로\s*정", _label_text(h["label"]) or h["text"]) for h in used):
        return ans
    parts = re.split(r"((?<=[.!?])\s+)", ans)
    sents, seps = parts[0::2], parts[1::2] + [""]
    keep = [(s, sep) for s, sep in zip(sents, seps) if not DELEGATE_SENT.search(s)]
    return "".join(s + sep for s, sep in keep).strip() if any(s.strip() for s, _ in keep) else ans


CLAUSE_END = {"있": "있습니다", "없": "없습니다", "되": "됩니다", "하": "합니다", "이": "입니다"}


def _trim_clause(sent, pat):
    """'A는 ~할 수 있고, B는 ~설치됩니다.'에서 B 절만 빼고 'A는 ~할 수 있습니다.'로 끝맺습니다. 못 하면 None."""
    clauses = re.split(r"(?<=[고며]),\s*", sent)
    keep = [c for c in clauses if not pat.search(c)]
    if len(clauses) < 2 or not keep or len(keep) == len(clauses):
        return None
    out = ", ".join(keep)
    if not re.search(r"[.!?]\s*$", out):                   # 마지막 절을 뺐으면 앞 절의 '~고/~며'를 맺음말로
        m = re.search(r"(있|없|되|하|이)(?:으며|며|고)$", out.rstrip())
        if not m:
            return None
        out = out.rstrip()[:m.start()] + CLAUSE_END[m.group(1)] + "."
    return out


def _drop_unasked_facility(question, ans, used):
    terms = _person_terms(question)
    for entry in UNASKED_FACILITY:
        sent_pat, asked, label_pat = entry[:3]
        mode = entry[3] if len(entry) > 3 else ""
        if asked.search(question):
            continue
        if mode == "cited_whole":
            if not any(label_pat.search(h["label"]) for h in used):
                continue
            new = _remove_sentences(ans, sent_pat, clauses=False)
            rest = [h for h in used if not label_pat.search(h["label"])]
            if rest:
                ans, used = (new or ans), rest
            continue
        # 질문한 사람이 그 시설 근거에 대상으로 나오면 진짜 혜택입니다 ("다자녀 혜택?" → 공공자전거 감면 제12조의3 제2항 제8호)
        if terms and any(label_pat.search(h["label"])
                         and any(w in _without_law_names(_label_text(h["label"]) or h["text"]) for w in terms)
                         for h in used):
            continue
        parts = re.split(r"((?<=[.!?])\s+)", ans)          # [문장, 구분자, 문장, …] 줄바꿈·목록 모양 유지
        sents, seps = parts[0::2], parts[1::2] + [""]
        keep = []
        for s, sep in zip(sents, seps):
            if not sent_pat.search(s):
                keep.append((s, sep))
            elif (t := _trim_clause(s, sent_pat)):          # 다른 내용과 한 문장이면 그 절만 뺍니다
                keep.append((t, sep))
        if not any(s.strip() for s, _ in keep):
            continue
        if keep != list(zip(sents, seps)):
            ans = "".join(s + sep for s, sep in keep).strip()
        # 지운 문장이 없어도(앞 규칙에서 이미 지워졌거나 원래 안 썼으면) 그 시설의 인용은 덤이라 뺍니다
        rest = [h for h in used if not label_pat.search(h["label"])]
        if rest and rest != used:
            used = rest
    return ans, used


def _method_fix(sentence, used):
    """방식 바로잡기 판정: 그 문장이 말하는 행위·제도(예: '임시 보호하는 제도는')가 고른 근거에 실제로 있으면
    '규정 없음'이 아니라 방식이 다르다는 뜻 (U22). 근거에 없는 행위(U19 '주워가는 제도')는 False."""
    m = METHOD_ACK.search(sentence)
    if not m:
        return False
    used_text = " ".join(re.sub(r"\s+", "", h["text"]) for h in used)
    words = m.group(1).split()
    for cand in ("".join(words), words[-1]):                     # "데려가 임시보호하" → 둘 다, "임시보호"도 확인
        term = re.sub(r"하$", "", cand)                          # "임시 보호하" → "임시보호"
        if len(term) >= 3 and term in used_text:
            return True
    return False


EXCEPT_OF = re.compile(r"(?<![조\d])제(\d+)항(?:\s*본문|\s*각\s*호(?:의\s*규정)?)?에도\s*불구하고")


def _with_principle(used):
    """'제1항에도 불구하고 …'로 시작하는 예외 항을 인용하면, 그 원칙 항(같은 조 제1항)도 함께 인용합니다.

    예) H19 공원 상행위: 제22조 제2항(공익 행사는 허용)만 인용하면 '원칙적으로 금지'의 근거(제1항)가 빠집니다.
    전체 조각 중 해당 항은 3개뿐이라 영향 범위가 좁습니다.
    """
    out, have = [], {h["label"] for h in used}
    for h in used:
        out.append(h)
        m = EXCEPT_OF.search(h["text"].split("\n", 1)[-1][:120])
        if not m or not h.get("article"):
            continue
        label = f"{h['article']} 제{m.group(1)}항"
        if label in have:
            continue
        base = next((c for c in _load()[0] if c["label"] == label and not c.get("sub")), None)
        if base:
            out.append(base)
            have.add(label)
    return out


PROC_Q = re.compile(r"[하할]려면|하자면|절차|방법|어떻게\s*(신청|지정|설치|등록)|신청은\s*어떻게")
DUP_LIMIT = re.compile(r"둘\s*이상의\s*\S*\s*감면\s*사유|감면\s*사유가\s*둘\s*이상")


def _with_reduction_list(used):
    """'둘 이상의 감면사유에 해당하면 높은 하나만'(중복 제한) 항을 인용하면 같은 조의 감면 목록 항(제1항)도 함께 인용합니다.
    예) M1 "장애인·다둥이 할인 둘 다?" → 답이 비율(80%·50%)을 안 쓰면 제7조 제3항만 인용돼 제1항이 빠졌음(0.75점)."""
    out, have = list(used), {h["label"] for h in used}
    for h in used:
        if not h.get("article") or not DUP_LIMIT.search(h["text"]):
            continue
        label = f"{h['article']} 제1항"
        if label in have:
            continue
        base = next((c for c in _load()[0] if c["label"] == label and not c.get("sub")), None)
        if base:
            out.append(base)
            have.add(label)
    return out


def _with_zone_designation(used, hits):
    """'금주구역에서 음주하면 과태료' 항을 인용했으면 그 구역을 지정하는 조(제목에 '금주구역 지정')도 함께 인용합니다.

    예) M2 어린이공원 음주: 제23조 제4항(과태료 10만원)만 고르고 제22조의3(금주구역 지정)을 가끔 빠뜨림.
    검색 결과(hits) 안에 있는 지정 조만 붙이므로 엉뚱한 조를 끌어오지 않습니다.
    """
    fines = [h for h in used if "과태료" in h["text"]]
    have = {h["label"] for h in used}
    out = list(used)
    for f in fines:
        for z in {m.group(1) for m in ZONE_FINE.finditer(f["text"])}:
            def designates(h, z=z):
                return re.search(rf"\([^)]*{z}[^)]*지정[^)]*\)", h["text"].split("\n", 1)[0])
            if any(designates(h) for h in out):          # 이미 지정 조를 인용했으면 더 붙이지 않음 (S2 제5조 제4항 등)
                continue
            # 검색 결과에 없으면 같은 조례의 조각에서 찾습니다 (검색이 지정 조를 놓친 경우)
            pool = list(hits) + [c for c in _load()[0] if c.get("law") == f.get("law") and not c.get("sub")]
            for h in pool:
                if h["label"] not in have and designates(h) and not DISTRICT.search(h["label"]):
                    out.append(h)
                    have.add(h["label"])
                    break
    return out


def _target_in_evidence(target, used):
    """질문의 "OO 전용/우선"이 고른 근거에 실제로 'OO…전용/우선'으로 있는지.

    있음: 장애인 → '장애인전용주차구획', 국가유공자 → '국가유공자 우선주차구획',
          친환경차 → '환경친화적 자동차 전용주차구획' (글자가 '전용' 앞 20자 안에 모두 있으면 같은 것으로 봄)
    없음: 임산부 → 주차장 조례에는 '가족배려주차장'(이용대상에 임산부)뿐, 여성 → 어디에도 없음
    """
    need = set(target)
    for h in used:
        t = re.sub(r"\s+", "", h["text"])
        for m in re.finditer(r"전용|우선", t):
            if need <= set(t[max(0, m.start() - 20):m.start()]):
                return True
    return False


def _with_amount_sources(answer, used, hits):
    """답에 쓴 금액·비율이 인용한 조문 어디에도 없으면, 그 숫자가 적힌 검색 근거를 인용에 더합니다.

    예) M1 "장애인 100분의 80, 다둥이 100분의 50 → 높은 하나만": 제7조 제3항만 고르고
        비율이 적힌 제7조 제1항을 빠뜨림. 같은 조례의 근거를 먼저 봅니다.
    """
    norm = lambda t: re.sub(r"\s+", "", t)
    used_text = norm(" ".join(h["text"] for h in used))
    laws = {h.get("law") for h in used}
    out, have = list(used), {h["label"] for h in used}
    for m in AMOUNT_A.finditer(answer):
        amt = norm(m.group(0))
        if amt == "무료" or amt in used_text:
            continue
        for h in sorted(hits, key=lambda h: (h.get("law") not in laws, -h.get("score", 0))):
            if h["label"] not in have and amt in norm(h["text"]) and not DISTRICT.search(h["label"]):
                out.append(h)
                have.add(h["label"])
                used_text += norm(h["text"])
                break
    return out


def _parse_json(raw):
    """구조화 출력인데도 추론 모드에서 드물게 JSON 앞뒤에 글이 붙어 오는 경우가 있어, 가운데 {…}만 꺼내 읽습니다."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw or "", re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _fee_annex(h):
    """요금·사용료 별표인지 (제목에 요금·료·금액이 있는 별표)."""
    return "별표" in h["label"] and bool(re.search(r"요금|료|금액", h["text"][:120]))


CLAIM_STOP = {"소지자", "사람", "대상", "이상", "경우", "본인", "가족", "등록된", "등재된", "해당", "시민", "이용자",
              "소유자등", "소유자", "할인", "감면", "면제", "혜택", "요금", "입장료", "이용료"}


def _head_kind(text):
    """항 머리말(첫 호 앞)로 그 목록이 면제인지 감면인지."""
    head = re.sub(r"\[[^\]]*\]", " ", text)                  # 머리의 [조례 이름 제N조(조 제목)] 빼기
    head = re.split(r"\n\s*1\.\s", head, maxsplit=1)[0]
    if "면제" in head and not re.search(r"감면|할인", head):
        return "면제"
    if re.search(r"감면|할인", head) and "면제" not in head:
        return "감면"
    return ""


def _locate(word, hits):
    """근거 조각들 중 그 낱말이 적힌 (라벨, 호 번호, 면제/감면) 목록."""
    locs, seen = [], set()
    for h in hits:
        if h["label"] in seen:
            continue
        seen.add(h["label"])
        text = h["text"]                                    # 줄 구조가 살아 있는 항 조각 (호 번호를 읽으려고)
        flat = re.sub(r"\s+", "", text)
        if word not in flat:
            continue
        # 호 하나에 '가. 나. 다.' 목이 줄을 바꿔 딸려 있으면(주차장 조례 제7조 제1항 제1호) 그 줄들도 그 호의 글입니다
        blocks, cur = {}, None
        for ln in text.splitlines():
            if HO_LINE.match(ln):
                cur = ln.strip().split(".", 1)[0]
            if cur is not None:
                blocks[cur] = blocks.get(cur, "") + re.sub(r"\s+", "", ln)
        nums = [n for n, body in blocks.items() if word in body]
        for n in dict.fromkeys(nums or [""]):
            if (h["label"], n) not in {(a, b) for a, b, _ in locs}:
                locs.append((h["label"], n, _head_kind(text)))
    return locs


SUBJ_NOT_LISTED = re.compile(r"(목록|대상|항목)에[^.]{0,30}?(없|포함되지\s*않|포함되어\s*있지\s*않|들어\s*있지\s*않|적혀\s*있지\s*않|정해져\s*있지\s*않|들어가지\s*않)"
                             r"|대상이\s*아니|해당하지\s*않|감면받을\s*수\s*없")
_last_claim_facts = threading.local()   # 주장 검증이 끝내 안 맞았을 때 확인된 사실 (2차 시도에 넘김)


def _check_claims(claims, hits, answer=""):
    """모델이 적은 주장(대상→근거·호·분류)을 조문과 대조합니다. 조항·주제를 지정하지 않는 계산입니다.

    - 대상 낱말이 근거 조각 어딘가에 있는데, 모델이 고른 조항(또는 호)에는 없으면 잘못 짚은 것
    - 면제/감면 분류가 그 항의 머리말과 다르면 잘못 분류한 것
    예) "국가유공자 + 30명 단체" → 국가유공자를 감면 항(제2항)에 넣음. 실제로는 면제 항(제1항) 제5호.
    문제가 있으면 (False, 확인된 사실 문장), 없으면 (True, "").
    """
    facts = []
    by_label = {h["label"]: h for h in hits}
    for c in claims or []:
        subj, label, ho, kind = c.get("subject", ""), c.get("label", ""), str(c.get("ho", "")).strip(), c.get("kind", "")
        # 대상(누가 면제·감면 받나)을 목록에서 읽은 주장만 봅니다. '맹견 출입 금지'처럼 규제 대상을 말한 주장,
        # 조문과 말이 다른 행위('임시 무대')는 낱말 위치로 판단할 수 없어 건드리지 않습니다.
        h0 = by_label.get(label)
        if kind not in ("면제", "감면") or not h0 or not any(HO_LINE.match(ln) for ln in h0["text"].splitlines()):
            continue
        words = [w for w in _hangul_words(subj) if len(w) >= 3 and w not in CLAIM_STOP]
        # 답이 그 대상을 "목록에 없다/대상이 아니다"라고 말했으면, 모델이 kind를 감면으로 적었어도 부정 판단이라 검사하지 않음
        # ("다둥이 운동시설 감면되나요?"에 "아니요, 감면 목록에 없습니다"라고 맞게 답하고도 불일치로 떨어지던 일)
        if words and any(any(w in s for w in words) and SUBJ_NOT_LISTED.search(s)
                         for s in re.split(r"(?<=[.!?])\s+|,\s*", answer)):
            continue
        # 호는 "7호"뿐 아니라 "1호, 7호"·"제1호·제7호"처럼 여럿을 함께 적기도 합니다 → 번호 집합으로 비교
        # (예전에는 "1호, 7"을 통째로 번호와 비교해, 맞게 짚은 주장도 늘 '잘못 짚음'으로 판정했음)
        # "7의3"·"제7호의3"은 한 호(7의3)입니다 (7과 3으로 쪼개면 맞게 짚은 임산부 7의3호도 불일치로 판정했음)
        ho_nums = set(re.findall(r"\d+(?:의\d+)?", re.sub(r"호\s*의", "의", ho)))
        bad = False
        for w in words:
            locs = _locate(w, hits)
            if not locs:
                continue
            here = [l for l in locs if l[0] == label]
            if not here or (ho_nums and here[0][1] and not any(l[1] in ho_nums for l in here)):
                bad = True
            elif kind in ("면제", "감면") and here[0][2] and here[0][2] != kind:
                bad = True
            if bad:
                where = "; ".join(f"{lb.replace('서울특별시 ', '')}{(' ' + n + '호') if n else ''}"
                                  f"{(' (' + k + ' 목록)') if k else ''}" for lb, n, k in locs[:4])
                # 어디에 있는지뿐 아니라 '고른 곳에는 없다'도 알려 줍니다. 위치만 알려 주면 같은 대상을
                # 그 항(예: 운동시설 감면 목록)에 또 넣는 일이 있었습니다.
                picked = f"{label.replace('서울특별시 ', '')}{(' ' + ho) if ho else ''}"
                if not here:
                    facts.append(f"- '{subj}'은(는) {picked}에는 없다. 근거에서 {where}에 적혀 있다. "
                                 f"{picked}의 대상으로 쓰지 말 것.")
                else:
                    facts.append(f"- '{subj}'은(는) 근거에서 {where}에 적혀 있다.")
                break
    return (not facts), "\n".join(facts)


FEE_HEAD = re.compile(r"(?:([가-힣]{2,})\s)?([가-힣]*(?:료|요금))\s*(?:을|를)\s*(?:면제|감면|할인)")
SUBJ_SUFFIX = re.compile(r"(이랑|할인|감면|면제|혜택|과|와|을|를|이|가|은|는|도|의|에|랑)$")
GROUP_REF = re.compile(r"두\s*경우|둘\s*다|모두|각각|두\s*(?:사람|대상|할인|혜택)")


RATE_RE = re.compile(r"\d+\s*분의\s*\d+|\d+\s*(?:%|퍼센트)")


def _norm_rate(t):
    """"100분의 80"·"80%"·"80퍼센트"를 같은 꼴(100분의80)로."""
    t = re.sub(r"\s+", "", t)
    return re.sub(r"(\d+)(?:%|퍼센트)", r"100분의\1", t)


def _ho_block(h, num, head=False):
    """항 조각에서 그 호(딸린 '가. 나.' 줄 포함)의 글. head=True면 항 머리말(첫 호 앞)도 붙임."""
    lines, cur, out = h["text"].splitlines(), None, []
    head_lines = []
    for ln in lines:
        if HO_LINE.match(ln):
            cur = ln.strip().split(".", 1)[0]
        if cur is None:
            head_lines.append(ln)
        elif cur == num:
            out.append(ln)
    return ("\n".join(head_lines) + "\n" if head else "") + "\n".join(out)


def _list_lists(hits):
    """근거 중 면제·감면 목록 항: [(label, 요금 이름, 앞 수식어 또는 None, 면제/감면)].
    요금 이름·종류는 항 머리말("…별표 2의 운동시설 이용료를 감면할 수 있다")에서 읽습니다."""
    out, seen = [], set()
    for h in hits:
        if h["label"] in seen or not any(HO_LINE.match(ln) for ln in h["text"].splitlines()):
            continue
        seen.add(h["label"])
        kind = _head_kind(h["text"])
        head = re.split(r"\n\s*1\.\s", re.sub(r"\[[^\]]*\]", " ", h["text"]), maxsplit=1)[0]
        m = FEE_HEAD.search(head)
        if kind and m:
            qual = m.group(1) if m.group(1) and not re.search(r"(는|은|에|를|을|의|고|면|서|도|여)$", m.group(1)) else None
            out.append((h["label"], m.group(2), qual, kind))
    return out


def _question_subjects(question, hits):
    """질문 속 대상 이름 중 근거 목록(호)에 실제로 적힌 것. "국가유공자할인과" → "국가유공자".
    괄호 안 단서("(다만, 서울대공원은 …)")에만 나오는 장소 이름은 대상이 아니라서 괄호를 지우고 찾습니다."""
    bare = [{**h, "text": re.sub(r"\([^()]*\)", "", h["text"])} for h in hits]
    subs = []
    for tok in re.findall(r"[가-힣]{2,}", question):
        forms, w = [tok], tok
        while True:                                   # 조사·'할인' 등을 하나씩 떼며 후보를 만듦 (긴 것부터)
            w2 = SUBJ_SUFFIX.sub("", w)
            if w2 == w or len(w2) < 2:
                break
            forms.append(w2)
            w = w2
        for f in reversed(forms):                 # 기본형부터 ("장애인이"보다 "장애인")
            if len(f) >= 3 and f not in CLAIM_STOP and any(n for _, n, _ in _locate(f, bare)):
                if f not in subs:
                    subs.append(f)
                break
    # 목록 머리말에 나오는 말(운동시설·주차요금 등 요금 이름)은 대상이 아님
    heads = "".join(re.sub(r"\s+", "", _ho_block(h, "", head=True)) for h in hits
                    if any(HO_LINE.match(ln) for ln in h["text"].splitlines()) and _head_kind(h["text"]))
    return [w for w in subs if w not in heads]


def _check_list_sentences(question, answer, hits):
    """답 문장이 말한 '대상 → 요금 목록(면제/감면)'을 조문 목록과 대조합니다 (모델이 claims에 안 적은 곁가지도 잡음).
    예) "운동시설 이용료는 두 경우 모두 감면 대상" → 다둥이는 운동시설 감면 목록에 없음
        "국가유공자는 입장료 감면 대상" → 국가유공자는 입장료 '면제' 목록에 있음
    돌려줌: [(문장 번호, 확인된 사실)]. 판단할 수 없는 문장(요금이 둘 이상, 부정문, 종류 불명)은 건드리지 않습니다."""
    lists = _list_lists(hits)
    subs = _question_subjects(question, hits)
    if not lists or not subs:
        return []
    bad = []
    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    all_cores = {core for _, core, _, _ in lists}
    by_label = {}
    for h in hits:
        by_label.setdefault(h["label"], h)
    # 목록 머리말에 나오는 말(주차요금·입장료 등 요금 이름)은 대상이 아님
    heads = "".join(re.sub(r"\s+", "", _ho_block(by_label[lb], "", head=True)) for lb, _, _, _ in lists if lb in by_label)
    subs = [w for w in subs if w not in heads]
    if not subs:
        return []
    for i, s in enumerate(sents):
        s_cores = {c for c in all_cores if c in s}
        if not s_cores:
            continue
        flagged, prev_named = False, []
        for clause in re.split(r",\s*|(?<=고)\s+|(?<=며)\s+", s):
            here_named = [w for w in subs if w in clause]
            if flagged or re.search(r"없|아니|않|제외|빠져", clause):
                prev_named = here_named or prev_named
                continue
            kind = "면제" if "면제" in clause else "감면" if re.search(r"감면|할인", clause) else None
            if not kind:
                prev_named = here_named or prev_named
                continue
            # 이 절이 말하는 요금: 절 안에 적힌 요금들(“운동시설 이용료와 주차요금은 둘 다 감면”), 없으면 문장의 유일한 요금
            cores = {c for c in s_cores if c in clause} or (s_cores if len(s_cores) == 1 else set())
            named = here_named
            if not named and GROUP_REF.search(clause):
                named = subs
            # 주어를 생략한 절은 앞 절의 대상을 이어받습니다
            # ("국가유공자나 다둥이는 입장료를 면제받을 수 있고, 운동시설 이용료는 100분의 50 감면받을 수 있습니다")
            if not named and prev_named:
                named = prev_named
            prev_named = named or prev_named
            for core in cores:
                quals = [q for lb, c, q, k in lists if c == core and k == kind]
                cands = [lb for lb, c, q, k in lists if c == core and k == kind and (q is None or q in s)]
                if not named or not cands or (len(cores) > 1 and len(quals) != len(cands)):
                    continue
                for w in named:
                    locs = _locate(w, hits)
                    if locs and not any(lb in cands for lb, _, _ in locs):
                        where = "; ".join(f"{lb.replace('서울특별시 ', '')}{(' ' + n + '호') if n else ''}"
                                          f"{(' (' + k + ' 목록)') if k else ''}" for lb, n, k in locs[:4])
                        bad.append((i, f"- '{w}'은(는) {core} {kind} 목록({cands[0].replace('서울특별시 ', '')})에 없다. "
                                       f"근거에서 {where}에 적혀 있다."))
                        flagged = True
                        break
                if flagged:
                    break
            # 감면율 대조: "임산부 할인과 장애인 할인은 각각 100분의 50" → 장애인 호에는 100분의 80
            rates = {_norm_rate(r) for r in RATE_RE.findall(clause)}
            if flagged or kind != "감면" or not rates or not named:
                continue
            labels = {lb for lb, c, q, k in lists if not cores or (c in cores and (q is None or q in s))}
            # 대상마다 그 이름 뒤 ~ 다음 대상 이름 앞 구간의 감면율만 그 대상의 것으로 봅니다
            # ("다둥이 할인은 100분의 50이므로 장애인 할인만 적용" → 50은 다둥이 것)
            pos = sorted((clause.find(w), w) for w in named if w in clause)
            for w in named:
                if pos:
                    k = next((j for j, (_, x) in enumerate(pos) if x == w), None)
                    if k is None:
                        continue
                    seg = clause[pos[k][0]:pos[k + 1][0] if k + 1 < len(pos) else len(clause)]
                    w_rates = {_norm_rate(r) for r in RATE_RE.findall(seg)}
                else:
                    w_rates = rates                 # "두 경우 모두 100분의 50"처럼 이름 없이 묶어 말한 경우
                if not w_rates:
                    continue
                spots = [(lb, n) for lb, n, _ in _locate(w, hits) if lb in labels and n]
                if not spots:
                    continue
                ok_texts = [_norm_rate(_ho_block(by_label[lb], n, head=True)) for lb, n in spots if lb in by_label]
                if ok_texts and not any(r in t for r in w_rates for t in ok_texts):
                    lb, n = spots[0]
                    real = sorted({_norm_rate(x) for x in RATE_RE.findall(_ho_block(by_label[lb], n, head=True))})
                    bad.append((i, f"- '{w}'의 감면율은 근거({lb.replace('서울특별시 ', '')} {n}호)에서 "
                                   f"{', '.join(real) or '따로 적혀 있지 않다'}이다. 다른 대상의 감면율을 쓰지 말 것."))
                    flagged = True
                    break
    return bad


def _claims_still_stated(claims, answer, hits):
    """문장을 뺀 뒤에도 답에 남아 있는 주장만 (뺀 문장의 요금 목록을 가리키는 주장은 검사하지 않음)."""
    info = {lb: (c, k) for lb, c, _, k in _list_lists(hits)}
    sents = re.split(r"(?<=[.!?])\s+", answer)
    keep = []
    for c in claims or []:
        lk = info.get(c.get("label", ""))
        if lk and not any(lk[0] in s and (lk[1] in s or (lk[1] == "감면" and "할인" in s)) for s in sents):
            continue
        keep.append(c)
    return keep


def _attempt(question, hits, reasoning_effort=None, timeout=None, convert=False, strict=False, feedback="", rounds=0):
    """Solar 답변 1회 + 검증. 성공하면 (결과, None), 실패하면 (None, 실패 이유)."""
    import upstage
    raw = upstage.chat(
        [{"role": "system", "content": SYSTEM_LEAN if "prompt" in LEAN else SYSTEM},
         {"role": "user", "content": f"[근거]\n{_context(hits, question)}\n\n[질문]\n{question}{_unit_hint(question, hits, convert)}"
                                     + (f"\n\n[확인된 사실]\n{feedback}\n위 사실에 맞게 대상별 분류와 근거를 바로잡아 다시 답하라." if feedback else "")}],
        response_format=_schema(hits), reasoning_effort=reasoning_effort, timeout=timeout)
    out = _parse_json(raw)
    picked = out.get("evidence", [])
    if config.EVIDENCE_KEY == "index":
        used = [hits[i - 1] for i in picked if isinstance(i, int) and 1 <= i <= len(hits)]
    else:                       # 라벨 → 그 라벨의 첫 조각 (별표처럼 조각이 여럿이어도 링크·시행일은 같음)
        by_label = {}
        for h in hits:
            by_label.setdefault(h["label"], h)
        used = [by_label[lb] for lb in dict.fromkeys(picked) if lb in by_label]
    # 주장 단위 검증: 대상을 엉뚱한 조항·호에 넣거나 면제/감면을 거꾸로 적었으면, 확인된 사실을 주고 한 번 다시 쓰게 합니다
    # 문장 단위 목록 대조: claims에 안 적은 곁가지("운동시설도 둘 다 감면")까지 조문 목록과 맞춰 봅니다.
    # 틀린 곁가지 문장은 빼고, 결론(첫 문장)이 틀렸으면 확인된 사실을 주고 다시 쓰게 합니다.
    if CLAIM_CHECK and out.get("found"):
        bad = _check_list_sentences(question, out["answer"], hits)
        # 질문이 특정 요금(예: 운동시설 이용료)을 물었는데 결론(첫 문장)이 다른 요금(입장료)만 말하면 다른 질문에 답한 것
        lists_q = _list_lists(hits)
        s1_ans = re.split(r"(?<=[.!?])\s+", out["answer"].strip(), maxsplit=1)[0]
        q_fee = {(c, q) for _, c, q, _ in lists_q if c in question and (q is None or q in question)}
        a_fee = {(c, q) for _, c, q, _ in lists_q if c in s1_ans and (q is None or q in s1_ans)}
        if q_fee and a_fee and not (q_fee & a_fee):
            asked = ", ".join(f"{q + ' ' if q else ''}{c}" for c, q in q_fee)
            bad = [(0, f"- 질문은 {asked}을(를) 묻는다. 그 요금의 목록 기준으로 결론을 쓸 것.")] + bad
        # 중복 제한 규칙이 근거(인용한 조례)에 없는데 "한 번만 적용 / 두 번 받지 않는다"고 단정한 문장은 추론입니다
        # ("65세+장애인 운동시설 이용료 두 번?" → "두 사유가 겹쳐도 감면은 한 번만 적용됩니다")
        laws_used = {h.get("law") for h in used}
        has_rule = any(RULE_TEXT.search(_label_text(h["label"]) or h["text"]) for h in hits if h.get("law") in laws_used)
        if not has_rule and re.search(r"할인|감면|면제|요금|료|혜택", question):
            for i, s in enumerate(re.split(r"(?<=[.!?])\s+", out["answer"].strip())):
                if DUP_INFER.search(s) or RULE_CLAIM.search(s):
                    bad.append((i, "- 근거에 둘 이상의 감면·면제 사유를 겹쳐 적용하는지 정한 규정이 없다. "
                                   "'한 번만 적용된다'고 단정하지 말고, 겹쳐 적용하는지는 조례에 정해져 있지 않다고 쓸 것."))
        # 요금·혜택을 겹쳐 받는지 묻는 질문만 ("미끄럼 방지시설도 같이 보나요?"의 '같이'는 해당 없음)
        if is_multi_q(question) and re.search(r"할인|감면|면제|요금|료|혜택", question):
            subs_m = _question_subjects(question, hits)
            # 중복 제한 규정도 없는데 "둘 다/두 번 받을 수 있나요?"에 "네"로 답하면, 겹쳐 받는다는 근거 없는 결론
            if not has_rule and out["answer"].lstrip().startswith("네"):
                bad = [(0, "- 근거에 둘 이상의 사유를 겹쳐 적용한다는 규정이 없다. '네'로 단정하지 말고, 각 대상이 어느 목록에 "
                           "있는지 쓴 뒤 겹쳐 적용하는지는 조례에 정해져 있지 않다고 쓸 것 "
                           "(둘 다 면제 대상이면 하나만 해당해도 면제라고 쓸 것).")] + bad
            # 질문한 대상이 모두 들어 있는 목록이 있는데, 답이 그런 목록(요금·면제/감면)을 하나도 말하지 않으면 핵심이 빠진 답
            # ("서울대공원 국가유공자+다둥이 둘 다?"에 두 대상이 모두 있는 입장료 면제 목록 대신 운동시설 감면만 말함)
            if len(subs_m) >= 2:
                full = [(lb, c, q, k) for lb, c, q, k in lists_q
                        if all(any(l[0] == lb for l in _locate(w, hits)) for w in subs_m)]
                kw = {"면제": r"면제", "감면": r"감면|할인"}
                if full and not any(c in out["answer"] and re.search(kw[k], out["answer"]) and (q is None or q in out["answer"])
                                    for _, c, q, k in full):
                    lb, c, q, k = full[0]
                    bad = [(0, f"- 질문한 대상 {', '.join(subs_m)}은(는) 모두 {q + ' ' if q else ''}{c} {k} 목록"
                               f"({lb.replace('서울특별시 ', '')})에 있다. 이 목록을 기준으로 결론을 쓸 것.")] + bad
            # 질문한 대상이 모두 같은 요금 목록에 있는데 "서로 다른 요금에 대한 혜택"이라고 하면 틀린 설명
            if len(subs_m) >= 2 and re.search(r"서로\s*다른\s*요금", out["answer"]):
                same = [lb for lb, _, _, _ in lists_q
                        if all(any(l[0] == lb for l in _locate(w, hits)) for w in subs_m)]
                if same:
                    i_bad = next(i for i, s in enumerate(re.split(r"(?<=[.!?])\s+", out["answer"].strip()))
                                 if re.search(r"서로\s*다른\s*요금", s))
                    bad.append((i_bad, f"- 질문한 대상 {', '.join(subs_m)}은(는) 모두 같은 목록"
                                       f"({same[0].replace('서울특별시 ', '')})에 있다. 서로 다른 요금의 혜택이 아니다."))
        # "국가유공자는 입장료 감면 목록에 없다"고만 하고, 같은 요금의 면제 목록에 있다는 걸 빼면 틀린 설명이 됨
        subs_n = _question_subjects(question, hits)
        sents_n = re.split(r"(?<=[.!?])\s+", out["answer"].strip())
        for i, s in enumerate(sents_n):
            for clause in re.split(r",\s*|(?<=고)\s+|(?<=며)\s+", s):
                if not SUBJ_NOT_LISTED.search(clause):
                    continue
                for w in [x for x in subs_n if x in clause]:
                    for lb, core, q, kind in lists_q:
                        if core not in s or (q and q not in s):
                            continue
                        spot = [l for l in _locate(w, hits) if l[0] == lb and l[1]]
                        other_kind = "면제" if kind == "면제" else "감면"
                        if spot and not any(w in s2 and core in s2 and other_kind in s2 and not SUBJ_NOT_LISTED.search(s2)
                                            for s2 in sents_n):
                            bad.append((i, f"- '{w}'은(는) {core} {kind} 목록({lb.replace('서울특별시 ', '')} {spot[0][1]}호)에 있다. "
                                           f"그 사실을 빼고 '목록에 없다'고만 쓰지 말 것."))
                            break
        # "네"로 시작했는데 답 스스로 질문한 대상이 목록에 없다고 말하면 결론이 거꾸로 된 것
        # ("네, 감면될 수 있습니다. … 다만 감면 대상 목록에는 다둥이 행복카드가 적혀 있지 않아 …")
        if out["answer"].lstrip().startswith("네"):
            subs_q = _question_subjects(question, hits)
            for s in re.split(r"(?<=[.!?])\s+", out["answer"])[1:]:
                gone = [w for w in subs_q if w in s and SUBJ_NOT_LISTED.search(s)]
                if gone:
                    bad = [(0, f"- '{gone[0]}'은(는) 질문한 요금의 목록에 없다. 결론은 '아니요'로 쓸 것.")] + bad
                    break
        if bad:
            sents_l = [s for s in re.split(r"(?<=[.!?])\s+", out["answer"].strip()) if s]
            drop = {i for i, _ in bad}
            merged = "\n".join(dict.fromkeys((feedback + "\n" + "\n".join(f for _, f in bad)).strip().splitlines()))
            left_l = [s for i, s in enumerate(sents_l) if i not in drop and not BARE_YES.match(s)
                      and not re.search(r"(따라|마다)\s*(다릅니다|달라집니다)\.?\s*$", s)]
            # 결론이 틀렸거나, 빼고 나면 내용이 안 남거나, "네, 감면될 수 있습니다."처럼 짧은 결론의 근거 문장(둘째)을 빼면 다시 쓰기
            if 0 in drop or not left_l or (1 in drop and BARE_YES.match(sents_l[0])):
                if rounds >= CLAIM_ROUNDS:
                    _last_claim_facts.v = merged
                    return None, "claim_mismatch"
                return _attempt(question, hits, reasoning_effort, timeout, convert, strict, feedback=merged, rounds=rounds + 1)
            removed = " ".join(s for i, s in enumerate(sents_l) if i in drop)
            out["answer"] = " ".join(s for i, s in enumerate(sents_l) if i not in drop)
            used = _drop_removed_citations(out["answer"], removed, used)
            out["claims"] = _claims_still_stated(out.get("claims"), out["answer"], hits)
    if CLAIM_CHECK and out.get("found") and out.get("claims"):
        ok, facts = _check_claims(out["claims"], hits, out["answer"])
        if not ok:
            if rounds >= CLAIM_ROUNDS:
                _last_claim_facts.v = "\n".join(dict.fromkeys((feedback + "\n" + facts).strip().splitlines()))
                return None, "claim_mismatch"
            # 앞 회차에서 알려 준 사실도 함께 줍니다 (새로 틀린 주장이 나와도 앞의 바로잡기를 잃지 않게)
            facts = "\n".join(dict.fromkeys((feedback + "\n" + facts).strip().splitlines()))
            return _attempt(question, hits, reasoning_effort, timeout, convert, strict, feedback=facts, rounds=rounds + 1)
    # 예·아니요(해도 되나요) 질문을 별표(요금표)만으로 답했으면, 허용·금지를 정한 조문 없이 요금이 있다는 것만으로
    # 추론한 것 → 확인 불가. (U14 "공원에 텐트 쳐도 되나요?" → 캠핑장 텐트 대여료만 보고 "네")
    # 금액·몇·얼마 질문은 해당 없음 (S1처럼 별표가 바로 답).
    yes_no = not is_content_q(question) and not is_amount_q(question) and not UNIT_Q.search(question)
    # 별표와 함께 무관한 본문 조항을 곁들여도 마찬가지입니다. (U14가 별표 2 + 제10조 제1항 '공원점용허가 기준
    # (노외주차장·가설건축물)'을 인용해 이 검사를 피했음) → 답과 낱말이 2개 이상 겹치는 본문 조항이 없으면 별표만으로 답한 것
    if _G and out.get("found") and used and yes_no and any(_fee_annex(h) for h in used):
        aw = {w for w in _hangul_words(out["answer"]) if len(w) >= 3}
        # '모든 시민에게 개방'(도시공원 제5조 제1항) 같은 일반 원칙은 그 행위를 허용하는 근거가 아닙니다.
        # (U14 텐트: 제5조 제1항 + 캠핑장 요금표만으로 "네"라고 답해 14번 중 7번 샜음)
        body = [h for h in used if "별표" not in h["label"]
                and not (USE_TOPIC_FILTERS and GENERIC_BASIS.search(h["label"]))]
        if all(len(aw & {w for w in _hangul_words(_label_text(h["label"]) or h["text"]) if len(w) >= 3}) < 2
               for h in body):
            return None, "annex_only"
    used = _with_principle(used)
    if USE_TOPIC_FILTERS:
        used = _with_reduction_list(used)
    used = _with_zone_designation(used, hits)     # 구조 규칙: 과태료 항이 가리키는 구역의 지정 조
    used = [h for h in used if not DISTRICT.search(h["label"])]
    if not used:
        return None, "not_found"
    if not out.get("found"):
        # found=false로 냈어도 "개인이 바로 임시 보호하는 제도는 조례에 정해져 있지 않습니다. 조례는 센터 동물을
        # 자원봉사자에게 임시보호하게 할 수 있도록 정하고 있으며…"처럼 말한 제도(임시보호)가 고른 근거에 실제로 있으면
        # 규정 없음이 아니라 방식이 다르다는 답입니다 (U22: 1차의 절반 가까이가 이렇게 found=false). 아래 검사를 그대로 거칩니다.
        # 근거에 없는 행위(U19 "고양이를 주워가는 제도")는 그대로 확인 불가.
        first = (re.split(r"(?<=[.!?])\s+", (out.get("answer") or "").strip()) or [""])[0]
        if not _method_fix(first, used):
            return None, "not_found"
    # 첫 문장이 "규정이 없다"로 결론 나는 답은 근거로 답한 것이 아니므로 '확인 불가' (프롬프트만으로는 잘 안 지켜짐)
    # 전제 바로잡기: "5분 단위가 아니라 1시간당 200원"처럼 질문의 단위를 고치고 실제 금액을 댄 답은
    # '○○ 단위 요금은 없다'는 말이 섞여도 규정 없음이 아닙니다. (검증기 판정은 그대로 받습니다)
    premise_fix = bool(PREMISE_FIX.search(out["answer"]) and AMOUNT_A.search(out["answer"]))
    # 첫 문장(결론)은 그대로 보고, 뒤 문장은 '조례·근거 자체에 규정이 없다'는 결론일 때만 봅니다.
    #   막을 것: D4 "따라서 … 과태료가 부과된다는 규정은 이 근거에 없습니다", D9 "… 금지하는 규정은 조례에 따로 없습니다"
    #   살릴 것: H13 "… 감면받을 수 있습니다. … 감면율은 따로 정해져 있지 않습니다." (세부 사항에 대한 덧붙임)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", out["answer"].strip()) if s] or [""]
    # 명칭 바꿔치기 차단: "OO 전용/우선 …"을 물었는데 답이 "그런 명칭은 없고 △△(가족배려주차장 등)가 있다"로 옮겨 가면
    # 다른 제도로 대신 답한 것 → 확인 불가. 그 이름의 제도는 6개 조례 밖 다른 조례에 있을 수 있습니다.
    #   예) 「서울특별시 임산부 전용주차구역 설치·운영에 관한 조례」가 따로 있는데, 주차장 조례 가족배려주차장으로 답했었음 (U28)
    #   (답이 명칭 차이를 말하지 않으면 검사하지 않음: F24 '친환경차 전용' ↔ 조문 '환경친화적 자동차 전용주차구획')
    tq = TARGET_Q.search(question)
    name_ack = False
    if _G and tq and (NAME_ACK.search(out["answer"]) or not _target_in_evidence(tq.group(1), used)):
        return None, "name_mismatch"
    # 방식 바로잡기: "임시 보호하는 제도는 조례에 정해져 있지 않습니다. 조례는 센터 동물을 자원봉사자에게…"처럼
    # 말한 제도(임시보호)가 고른 근거에 실제로 있으면 '규정 없음'이 아니라 방식이 다르다는 뜻 (U22).
    # 근거에 없는 행위(U19 "주워가는 제도")는 해당 없음.
    def method_fix(s):
        return _method_fix(s, used)

    # 목록을 보고 "그 대상은 이 목록에 없다"고 답한 것은 규정 없음이 아니라 목록에 따른 답입니다
    # ("다둥이 행복카드 있으면 운동시설 이용료 감면?" → "아니요, 감면 대상에 포함되어 있지 않습니다").
    # 대상 이름이 근거의 다른 목록에는 실제로 있을 때만 (어느 목록에도 없는 대상은 지금처럼 확인 불가)
    list_negative = bool(SUBJ_NOT_LISTED.search(sents[0]) and _list_lists(used)
                         and any(w in sents[0] for w in _question_subjects(question, hits)))
    head_no_rule = (NO_RULE.search(sents[0]) and not name_ack and not method_fix(sents[0]) and not list_negative
                    and not (re.search(r"둘\s*다|중복|함께|같이|동시에|겹", question)
                             and re.search(r"중복|함께|같이|동시에|겹", sents[0]) and len(sents) > 1)) \
        or STRONG_NO_RULE.search(out["answer"])      # "허용하는 규정은 아니고" 류는 어느 문장에 있어도 결론
    # 뒤 문장의 "다만 구체적인 대상·금액 기준은 조례에 직접 정해져 있지 않고 …"는 세부 사항 덧붙임이라 결론이 아님 (청년 월세)
    ctx_flat = re.sub(r"\s+", "", " ".join(h["text"] for h in hits))
    absent_q = [w for w in _hangul_words(question) if len(w) >= 2 and w[:2] not in ctx_flat
                and not re.search(r"(요|까|야|해|니다|는지|면|고|도|나)$", w)]     # 질문 명사 중 근거에 없는 것 (예: 헬멧)

    both_q_ = is_multi_q(question)

    def names_absent(x):
        if both_q_ and re.search(r"중복|함께|같이|동시에|겹|둘\s*다|두\s*(감면|면제|할인|혜택)", x):
            return True                         # 중복 질문에 '겹쳐 적용하는 규정은 없다'는 것이 바로 답
        return any(w[:2] in x for w in absent_q)
    tail_no_rule = any(NO_RULE.search(s) and re.search(r"^따라서|근거|조례", s) and not method_fix(s)
                       and not re.search(r"구체적|세부", s) and not names_absent(s) for s in sents[1:])
    if _G and head_no_rule and not premise_fix:
        return None, "no_rule_answer"
    # 결론(첫 문장)은 답했는데 뒤에서 '조례에 따로 정해져 있지 않다'로 뒤집은 경우: 막되 2차는 허용합니다.
    # (F13 "네, 신용카드로 낼 수 있습니다. … 다만 입장료 자체를 신용카드로 내는 방법은 조례에 따로 정해져 있지 않습니다.")
    if _G and tail_no_rule and not premise_fix:
        # 첫 문장이 실질적인 답이고, 뒤 문장이 '다만 …은 정해져 있지 않습니다'처럼 세부를 덧붙인 것이면 그 문장만 떼고
        # 나머지를 검증기에 맡깁니다 (맞게 답하고도 습관적인 덧붙임 한 문장 때문에 버려지던 일).
        # "따라서 …"처럼 결론을 뒤집는 문장이면 지금처럼 막습니다.
        tails = [x for x in sents[1:] if NO_RULE.search(x) and not method_fix(x) and not names_absent(x)]
        # 단, "… 이 근거만으로 확인할 수 없습니다"처럼 질문의 핵심에 답할 수 없다는 고백이면 떼지 않고 막습니다 (함정 방어)
        # 또는 질문·결론(첫 문장)에 없는 다른 대상을 주어로 세운 문장("자전거 이용요금은 … 정해져 있지 않습니다")이면
        # 묻지 않은 것에 대한 덧붙임이라 그 문장만 뗍니다. 주어가 질문·결론에 나오는 것이면 지금처럼 막습니다.
        def other_topic(x):
            m = re.match(r"\s*([가-힣·\s]{2,20}?)(?:은|는)\s", x)
            if not m:
                return False
            words = [w for w in _hangul_words(m.group(1)) if len(w) >= 2]
            return bool(words) and not any(w[:2] in question or w[:2] in sents[0] for w in words)
        if TAIL_STRIP and tails and all(re.match(r"\s*(다만|단,|단\s)", x) or other_topic(x) for x in tails) \
                and not any(re.search(r"(답할|확인할|판단할|알|말할)\s*수\s*(가\s*)?없|확인되지\s*않|근거만으로는", x) for x in tails):
            out["answer"] = " ".join(x for x in sents if x not in tails)
            sents = [x for x in sents if x not in tails]
        else:
            return None, "no_rule_tail"
    # '얼마' 질문인데 답에 구체적인 금액·비율이 없으면, 조례에 금액이 없어서 돌려 말한 것 → '확인 불가'
    # (예: 입양 비용이 얼마냐는 질문에 "보호기간에 따라 달라지며 청구할 수 있다"처럼 금액 없이 답하는 경우)
    if _G and is_amount_q(question) and (not AMOUNT_A.search(out["answer"])
                                      or (NO_AMOUNT.search(out["answer"]) and not premise_fix)):
        return None, "no_amount"
    # 비용·요금을 물었는데(할인·감면율 질문 제외) 답에 '○원'이나 '무료'가 없고 비율만 있으면, 다른 항목의
    # 감면율로 대신 답한 것 (D6 "입양 비용 얼마?" → 등록수수료 50퍼센트 감면·보호비용 청구 가능만 나열)
    if _G and is_amount_q(question) and COST_Q.search(question) and not RATE_Q.search(question) \
            and not WON_A.search(out["answer"]):
        return None, "no_amount"
    # '언제/시기/기간'을 물었는데 답이 바로 그 시기·기간이 정해져 있지 않다고 하면 규정 없음 (금액 질문의 no_amount와 같은 방식)
    # D8 "청년 월세 지원은 언제 신청할 수 있어요?" → "조례에는 구체적인 신청 시기나 기간이 명시되어 있지 않으며 …" (2차에서 샜음)
    if _G and is_when_q(question) and NO_WHEN.search(out["answer"]) and not TIME_A.search(out["answer"]):
        return None, "no_rule_answer"

    # 절차 질문("~하려면/할려면", 절차, 방법)은 절차·요건만: 과태료·사무 위임 문장을 검증 전에 뺍니다.
    # U5 "공원에 금연구역을 설치할려면?" 답 끝에 "과태료 부과·징수 사무는 … 위임합니다"가 붙으면 검증기가
    # offTopic으로 두 번 떨어뜨려(재시도도 없음) 확인 불가가 되던 일. 프롬프트 규칙만으로는 가끔 안 지켜짐.
    if _G and is_proc_q(question) and not re.search(r"과태료|위반|벌", question):
        new = _remove_sentences(out["answer"], re.compile(r"과태료|위임"))
        if new:
            out["answer"] = new
            rest = [h for h in used if not re.search(r"과태료|위임", h["text"].split("\n", 1)[-1][:80])]
            used = rest or used

    # 검증 ②: 답이 근거에 실제로 기반하는지 Groundedness Check
    # 고른 근거만 주면, 답에 곁들인 부수 내용(이의 절차 등)이 다른 근거에서 왔을 때 멀쩡한 답을 떨어뜨립니다.
    # 그래서 Solar에 넘긴 근거 전체를 기준으로 '지어낸 내용이 있는가'만 봅니다.
    if config.USE_VERIFY:
        vctx = _context(hits) + (_unit_note(question, hits) if convert else "")
        verdict, unneeded = upstage.groundedness(vctx, out["answer"], question, detail=True)
        # 탈락 판정이면 한 번 더 물어 두 번 다 탈락일 때만 떨어뜨립니다.
        # 멀쩡한 답이 우연히 떨어지는 일(채점 1회당 1~2건: U4·U5·F03·F25·H18)은 두 번 연속 나오기 어렵고,
        # 함정 답은 두 번 다 탈락이 나옵니다.
        # strict(1차가 '근거 없음'이었던 재시도): 한 번이라도 탈락·애매면 버림. 함정에서 2차가 그럴듯한 답을
        # 만들어 통과하던 경로(처음 보는 함정 80회 중 14회)를 막습니다. 처음 보는 일반 문항은 이 경로가 필요 없었음.
        if strict and verdict != "grounded":
            return None, "strict_reject"
        if verdict in ("notGrounded", "offTopic") and config.VERIFY_SECOND_OPINION:
            second, unneeded2 = upstage.groundedness(vctx, out["answer"], question, detail=True)
            if second not in ("notGrounded", "offTopic"):
                verdict, unneeded = second, unneeded2
        if verdict == "notGrounded":            # notSure는 통과시킵니다 (떨어뜨리면 멀쩡한 답이 더 많이 사라짐)
            return None, "not_grounded"
        if verdict == "offTopic":               # 근거는 맞지만 질문과 다른 대상을 답함 → 함정 방어
            return None, "off_topic"

        # 질문에 답하는 데 필요 없는 문장(검증기 판정)을 뺍니다. 첫 문장(결론)은 남깁니다.
        # 조항별 필터(맹견·임산부 등) 대신 쓰는 일반 장치: 처음 보는 조항에도 같은 기준이 적용됩니다.
        if TRIM_UNNEEDED:
            try:                                   # 편집 단계: 질문이 묻지 않은 문장 (실패해도 답은 그대로)
                unneeded = list(unneeded) + upstage.unasked_sentences(out["answer"], question)
            except Exception:  # noqa: BLE001
                pass
        if unneeded and TRIM_UNNEEDED:
            sents_v = upstage.split_sentences(out["answer"])
            drop = set(unneeded) - {1}
            if len(sents_v) > 1 and BARE_YES.match(sents_v[0]):
                drop -= {2}                        # "네, 이용할 수 있습니다."만 남지 않게 그 근거 문장은 둡니다
            if is_multi_q(question):              # "같이 받을 수 있나요?"의 결론(중복 제한) 문장은 빼지 않습니다
                drop -= {i for i, x in enumerate(sents_v, 1)
                         if RULE_CLAIM.search(x) or re.search(r"(함께|같이|둘\s*다|중복)[^.]{0,15}(받을\s*수\s*(는\s*)?없|적용되지\s*않)", x)}
            keep = [x for i, x in enumerate(sents_v, 1) if i not in drop]
            # 덜어낸 뒤 "네, 받을 수 있습니다. 다만 어떤 요금인지에 따라 다릅니다."처럼 내용 문장이 하나도 안 남으면
            # 편집이 답을 지운 것이라 덜어내지 않습니다.
            content = [x for x in keep if not BARE_YES.match(x)
                       and not re.search(r"(따라|마다)\s*(다릅니다|달라집니다)\.?\s*$", x)]
            if not content:
                keep = sents_v
            if len(keep) < len(sents_v):
                removed = [x for i, x in enumerate(sents_v, 1) if i in drop]
                out["answer"] = " ".join(keep)
                used = _drop_removed_citations(out["answer"], " ".join(removed), used)

    if _G and RULE_CLAIM.search(out["answer"]) and not any(
            RULE_TEXT.search(_label_text(h["label"]) or h["text"]) for h in used):
        return None, "unsupported_rule"     # 예) 공원 입장료에 조문에 없는 '감면율이 높은 하나만'을 붙임
    used = _with_amount_sources(out["answer"], used, hits)
    citation = list(dict.fromkeys(h["label"] for h in used))   # 순서 유지, 중복 제거
    first = used[0]
    ans = out["answer"].strip()
    if "post" in LEAN:                         # 후처리 없이 모델 답 그대로
        return {"answer": ans, "citation": citation,
                "source_url": first["source_url"], "effective_date": first["effective_date"]}, None
    if not is_content_q(question) and MUST_Q.search(question) and MUST_NOT.match(ans):
        ans = "아니요, " + ans                 # "반드시 사용해야 해?" → "반드시 사용해야 하는 것은 아니고 …"에 결론 붙이기
    both_q = is_multi_q(question)
    limit = re.compile(r"(하나만|한\s*가지만)(?!\s*해당)\s*(적용|받|인정)|함께\s*받을\s*수\s*(는\s*)?없|겹쳐\s*(받|적용)[^.]{0,6}(않|없)"
                       r"|중복(해서|하여)?\s*(받|적용)[^.]{0,6}(않|없)")
    if not is_content_q(question) and both_q and limit.search(ans):
        ans = re.sub(r"^\s*(네|예)\s*,\s*", "", ans)   # "네, 각각 대상입니다. 다만 함께 받을 수는 없습니다" → 결론은 '아니요'
        if not ans.startswith("아니요"):
            ans = "아니요, " + ans
        ans = _drop_contradicting_yes(ans)
    # 둘 다 면제 대상이면 '두 번 받는' 것이 아니므로 "네, 둘 다 받을 수 있습니다"로 시작하지 않습니다
    both_exempt = re.search(r"(모두|둘\s*다|각각)[^.]{0,40}면제\s*대상|면제\s*대상[^.]{0,30}(모두|둘\s*다)", ans) \
        or len(re.findall(r"면제\s*(대상|받)", ans)) >= 2 \
        or re.search(r"면제[^.]{0,40}하나만\s*해당해도[^.]{0,15}(내지\s*않|면제)", ans)   # "면제받을 수 있는 대상이라, 하나만 해당해도 내지 않습니다"
    if not is_content_q(question) and both_q and both_exempt and not limit.search(ans):
        ans = re.sub(r"^\s*네\s*,\s*(?:(?:두\s*할인\s*)?(?:둘\s*다|모두)\s*(?:받을|면제받을|적용받을)\s*수\s*있습니다\.\s*)?", "", ans)
        ans = re.sub(r"^\s*(?:두\s*(?:할인|혜택)\s*)?(?:둘\s*다|모두)\s*(?:받을|면제받을|적용받을)\s*수\s*있습니다\.\s*(?:다만\s*)?", "", ans)
        if not re.search(r"하나만\s*해당", ans):
            ans = ans.rstrip() + " 둘 중 하나만 해당해도 면제되므로 겹쳐 받을 할인은 없습니다."
    # "둘 다/두 번 받을 수 있나요?"인데 중복 제한 규정도 없고, 답이 겹쳐 적용 여부를 말하지 않았으면 그 사실을 덧붙임
    # ("65세 이상이고 장애인이면 운동시설 이용료 감면 두 번?" → 각각 감면 대상이라고만 답함)
    elif not is_content_q(question) and both_q and not limit.search(ans) \
            and re.search(r"할인|감면|면제|요금|료", question) and re.search(r"할인|감면|면제", ans) \
            and not any(RULE_TEXT.search(_label_text(h["label"]) or h["text"]) for h in used) \
            and not re.search(r"정해져\s*있지\s*않|정하고\s*있지\s*않|규정(이|은|을)?\s*(따로\s*)?(없|두고\s*있지\s*않)", ans):
        ans = ans.rstrip() + " 같은 요금에 두 사유를 겹쳐 적용하는지는 조례에 정해져 있지 않습니다."
    if not is_content_q(question) and not re.match(r"\s*(네|아니요|예)\s*[,.]", ans):
        s1 = re.split(r"(?<=[.!?])\s", ans, maxsplit=1)[0].strip()
        if NEG_END.search(s1) and re.search(r"(나요|까요|니까|되요|돼요|가요|어요|해요|수\s*있어|\?)\s*$", question.strip()):
            ans = "아니요, " + ans                 # "둘 다 받을 수 없습니다." → "아니요, 둘 다 받을 수 없습니다."
    if is_content_q(question):                 # '얼마/누구/언제…' 질문인데 "네,"로 시작하면 떼어 냅니다
        ans = re.sub(r"^(네|아니요|예)\s*[,.]\s*", "", ans)
        # 떼고 나면 "받을 수 있습니다."만 남는 첫 문장은 내용이 없어 뺍니다 ("지원은?" → "네, 받을 수 있습니다. 시장은 …")
        ans = re.sub(r"^(?:[가-힣]+\s)?(?:받을|할|이용할|신청할)\s*수\s*있습니다\.\s*(?=\S)", "", ans)
    elif re.match(r"^네\s*,", ans):            # "네, …할 수 없습니다"처럼 첫 문장이 부정이면 "아니요,"가 맞습니다
        s1 = re.split(r"(?<=[.!?])\s", ans, maxsplit=1)[0].strip()
        # "~내야 하나요?"에 "네, 내지 않아도 됩니다"도 방향이 틀린 경우입니다
        must_q = MUST_Q.search(question)
        if must_q and re.search(r"감면|면제", s1) and PAY_BUT.match(ans):
            # C3 "네, 내야 하지만 유기견을 입양하여 등록하면 전액 감면…" → 감면부터 말합니다
            ans = PAY_BUT.sub("", ans, count=1)
        elif NEG_END.search(s1) or (must_q and NOT_NEEDED.search(s1)):
            ans = re.sub(r"^네\s*,", "아니요,", ans)
        elif re.search(r"(지만|으나|나,)\s*[^.]*(하나만\s*적용|받을\s*수\s*없|적용되지\s*않|할\s*수\s*없)", s1):
            ans = re.sub(r"^네\s*,", "아니요,", ans)   # "네, 둘 다 해당할 수 있지만 하나만 적용됩니다" → 결론은 '아니요'

    if USE_TOPIC_FILTERS:   # 특정 조항·주제 문구 교정 (하드코딩)
        # 주차요금 감면(주차장 조례 제7조)은 공영주차장 요금에 대한 것입니다(제6조 제1항). 답이 주차장 종류 없이
        # "주차요금의 100분의 50을 할인"이라고만 쓰면 민간·부설주차장에서도 할인되는 것처럼 읽혀서 '공영주차장'을 붙입니다.
        if any(re.search(r"주차장 설치 및 관리 조례 제7조", h["label"]) for h in used) and "공영" not in ans \
                and re.search(r"할인|감면|면제", ans):
            ans = re.sub(r"(?<![가-힣])주차요금", "공영주차장 주차요금", ans, count=1)
        # 금주구역은 어린이공원만 지정할 수 있습니다(도시공원 조례 제22조의3). 답이 "금주구역으로 지정된 도시공원에서"처럼
        # 장소를 넓히거나, 장소 없이 "금주구역에서"만 쓰면 공원 어디서나 음주 과태료처럼 읽혀서 어린이공원으로 바로잡습니다.
        if any("어린이공원을 금주구역" in h["text"] for h in used) and "어린이공원" not in ans:
            ans = ans.replace("금주구역이나 금연구역으로 지정된 도시공원",
                              "금주구역으로 지정된 어린이공원이나 금연구역으로 지정된 도시공원")
            ans = re.sub(r"금주구역으로\s*지정(?:된|한|되어\s*있는)\s*(?:도시)?공원", "금주구역으로 지정된 어린이공원", ans)
            ans = re.sub(r"(?:도시)?공원\s*중\s*금주구역으로\s*지정(?:된|한)\s*곳", "금주구역으로 지정된 어린이공원", ans)
            if "어린이공원" not in ans:              # 장소 없이 "금주구역에서"만 쓴 답: 그 문장 뒤에 장소를 덧붙입니다
                sents = re.split(r"(?<=[.!?])\s+", ans)
                k = next((i for i, s in enumerate(sents) if "금주구역" in s), None)
                if k is not None:
                    sents.insert(k + 1, "금주구역은 시장이 어린이공원을 대상으로 지정합니다.")
                    ans = " ".join(sents)
        # 도시공원 제20조 제1항 제2호 "6세 미만 또는 65세 이상인 사람과 장애인"을 그대로 옮겨
        # "65세 이상인 사람과 함께 장애인은 면제"처럼 쓰면 동반해야 하는 것처럼 읽힙니다 → 물은 사람만 남깁니다
        if not re.search(r"65세|6세|어르신|노인|어린이|유아", question):
            ans = re.sub(r"(?:6세\s*미만\s*또는\s*)?65세\s*이상인\s*사람과\s*(?:함께\s*)?", "", ans)
        # "둘 다/같이 받을 수 있나요?"인데 결론이 '높은 하나만 적용'이면 답은 '아니요'입니다.
        # (M1 "네, 둘 다 적용받을 수는 있지만 실제로는 하나만…"처럼 '네'로 시작해 결론과 어긋났던 일)
        #   "둘 다 받을 수 있지만, … 하나만 적용됩니다"도 결론과 반대로 읽혀서 첫 문장을 통일합니다.
        if re.search(r"둘\s*다|두\s*번|모두|같이|함께|동시에|중복", question) and re.search(r"하나만\s*적용", ans) \
                and re.search(r"할인|감면", question + ans):
            s1, *rest = re.split(r"(?<=[.!?])\s+", ans, maxsplit=1)
            std = "아니요, 둘 다 받을 수는 없고 감면율이 높은 하나만 적용됩니다."
            if re.search(r"하나만\s*적용", s1):
                ans = std + (" " + rest[0] if rest else "")
            elif s1.startswith("아니요") and not re.search(r"없|않|못", s1):
                # "아니요, … 각각 적용받을 수 있는 요건은 갖추어져 있습니다. 다만 … 하나만 적용되므로 …" → 결론과 어긋난 첫 문장을 통일
                ans = std + (" " + re.sub(r"^다만\s*", "", rest[0]) if rest else "")
            elif not s1.startswith("아니요"):
                ans = "아니요, " + re.sub(r"^네\s*[,.]\s*", "", ans)
            # 묻지 않은 지하철 환승주차장 감면(최초 3시간 면제 등)은 중복 질문의 답이 아닙니다
            if not re.search(r"환승|지하철", question):
                ans = _remove_sentences(ans, re.compile(r"환승\s*주차장"), clauses=False) or ans
    # 조문 말투를 쉬운 말로: "다음 각 호의 어느 하나에 해당하는 사람" → "다음에 해당하는 사람".
    # 목록 번호 ①②③은 조례의 '항' 기호와 같아 제1항·제2항으로 읽힐 수 있어 1) 2) 3)으로 바꿉니다.
    ans = re.sub(r"다음\s*각\s*호의\s*어느\s*하나에\s*해당하는\s*", "다음에 해당하는 ", ans)
    ans = re.sub(r"다음\s*각\s*호의\s*", "다음 ", ans)
    ans = re.sub(r"[①-⑳]", lambda m: f"{ord(m.group()) - 0x2460 + 1}) ", ans).replace(")  ", ") ")
    # "임산부 전용 주차구역은 따로 없고 …"처럼 6개 조례에 없는 제도를 '없다'고 단정하는 문장(절)은 뺍니다.
    # 임산부 전용주차구역은 「서울특별시 임산부 전용주차구역 설치ㆍ운영에 관한 조례」에 따로 있습니다(6개 밖일 뿐).
    new = _remove_sentences(ans, NO_SUCH_FACILITY) if USE_TOPIC_FILTERS else None
    if new:
        ans = new
    # "공원은 근거에서 정한 맹견 출입금지 장소가 아닙니다"처럼 목록에 없다는 것만으로 '아니다'를 단정하는 문장은 뺍니다
    # (목록에는 어린이공원과 '시장이 지정·고시하는 장소'가 있어 공원 일반을 아니라고 할 근거가 없음)
    new = _remove_sentences(ans, NOT_LISTED, clauses=False)
    if new:
        ans = new
    # 근거에 없는 권고("이용하려는 공원의 안내를 확인하시는 것이 좋습니다", "~에 문의하세요")는 뺍니다 (프롬프트 금지 사항)
    new = _remove_sentences(ans, ADVICE)
    if new:
        ans = new
    new = _remove_sentences(ans, RESTATE, clauses=False)
    if new:
        ans = new
    new = _remove_sentences(ans, HEDGE, clauses=False)
    if new and not BARE_YES.match(new):        # 빼고 나서 "네, 받을 수 있습니다."만 남으면 내용이 사라진 것이라 두지 않습니다
        ans = new
    # "네, 받을 수 있습니다. 시장은 … 지원할 수 있습니다." → "네, 시장은 …" (재량 지원을 '받을 수 있다'로 단정하지 않게)
    ans = re.sub(r"^(네|아니요)\s*,\s*(?:받을|신청할)\s*수\s*(?:있|없)습니다\.\s+(?!다만|단,|그러나|하지만)(?=\S)", r"\1, ", ans)
    ans = ART_REF.sub("", ans)
    ans = LAW_REF.sub(lambda m: m.group(1), ans)
    new = _remove_sentences(ans, INFER, clauses=False)
    if new:
        ans = new
    # "…조례 제7조의3은" → "조례는" / 문장 중간의 "…조례 제20조 제2항에서"는 통째로 지웁니다 ("모두 조례는 별표…"처럼 꼬이던 일)
    ans = ART_SUBJ.sub(lambda m: "" if m.group(1) == "에서" else "조례" + {"은": "는"}.get(m.group(1), m.group(1)), ans)
    ans = re.sub(r"\s*\(질문과 관련된 호:[^)]*\)", "", ans)     # 근거에 붙인 표시가 답에 옮겨진 경우
    ans = re.sub(r"\s{2,}", " ", ans).strip()
    ans = re.sub(r"\s*\((?:[^()]*?\s)?제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?[^()]*\)", "", ans)   # "…조례 제7조의3은" → "조례는"
    if USE_TOPIC_FILTERS:
        ans, used = _drop_unasked_facility(question, ans, used)
        ans, used = _drop_unasked_facets(question, ans, used)
        ans, used = _drop_unasked_prevention(question, ans, used)
        ans, used = _drop_unasked_harm(question, ans, used)
        ans, used = _drop_other_object(question, ans, used)
    ans = _drop_absent_inference(question, ans, used)
    ans = _dedupe_clauses(ans)
    used = _drop_unused_citations(ans, used)
    used = _drop_other_law_citations(ans, used)
    if USE_TOPIC_FILTERS:
        ans, used = _drop_other_person_benefits(question, ans, used)
    ans = _drop_ungrounded_delegation(ans, used)
    if USE_TOPIC_FILTERS:
        used = _drop_offtopic_paragraphs(ans, used)
    # 중복 제한 규칙("높은 하나만")을 말했는데, 인용 정리 뒤 남은 근거에 그 규칙이 없으면 근거 없는 규칙입니다.
    # (서울대공원 입장료 답에 "같은 요금 안에서는 높은 하나만"이 남고 주차장 조례 제7조 제3항 인용만 빠진 일)
    if _G and RULE_CLAIM.search(ans) and not any(RULE_TEXT.search(_label_text(h["label"]) or h["text"]) for h in used):
        return None, "unsupported_rule"
    # 편집·정리 단계에서 핵심 문장이 빠졌는지 마지막에 한 번 더: 질문한 대상이 모두 들어 있는 목록(예: 입장료 면제)이
    # 있는데 최종 답이 그 목록을 하나도 말하지 않으면 2차로 다시 씁니다 (1차 검사 뒤 편집이 그 문장을 지운 일)
    if _G and is_multi_q(question) and re.search(r"할인|감면|면제|요금|료|혜택", question):
        subs_f = _question_subjects(question, hits)
        if len(subs_f) >= 2:
            kw_f = {"면제": r"면제", "감면": r"감면|할인"}
            full_f = [(c, q, k) for lb, c, q, k in _list_lists(hits)
                      if all(any(l[0] == lb for l in _locate(w, hits)) for w in subs_f)]
            if full_f and not any(c in ans and re.search(kw_f[k], ans) and (q is None or q in ans) for c, q, k in full_f):
                return None, "key_list_dropped"
    # 정리 끝에 "아니요, 둘 다 받을 수는 없습니다."처럼 결론 한 줄만 남으면(근거 문장·인용이 빠짐) 2차로 다시 씁니다
    if _G and BARE_YES.match(ans.strip()) and len(re.split(r"(?<=[.!?])\s+", ans.strip())) == 1:
        return None, "bare_answer"
    used = _cap_citations(ans, used)
    citation = list(dict.fromkeys(h["label"] for h in used))
    first = used[0]
    return {"answer": ans, "citation": citation,
            "source_url": first["source_url"], "effective_date": first["effective_date"]}, None


_oos_cache = {}


def out_of_scope(question):
    """질문이 6개 대상 조례 밖의 다른 서울시 조례에만 있는 용어를 쓰면 그 조례 이름을 돌려줍니다.

    예) "임산부 전용주차구역" → 「서울특별시 임산부 전용주차구역 설치·운영에 관한 조례」
        6개 조례로 비슷한 제도(가족배려주차장)를 끌어다 답하면 틀린 답이 됩니다 → 확인 불가.
    용어 목록: data/out_of_scope_terms.json. 6개 조례 본문에 있는 용어는 여기서 자동으로 빼서,
    범위 안 질문을 막지 않습니다.
    """
    if "terms" not in _oos_cache:
        path = config.DATA / "out_of_scope_terms.json"
        raw = json.loads(path.read_text(encoding="utf-8"))["terms"] if path.exists() else {}
        corpus = re.sub(r"\s+", "", " ".join(c["text"] + " " + c["label"] for c in _load()[0]))
        _oos_cache["terms"] = {re.sub(r"\s+", "", t): law for t, law in raw.items()
                               if re.sub(r"\s+", "", t) not in corpus}
    q = re.sub(r"\s+", "", question)
    for t, law in _oos_cache["terms"].items():
        if t in q:
            return law
    return None


PARK_Q = re.compile(r"공원|입장료")
CAR_Q = re.compile(r"주차|차량|자동차|차를|차\s*세|세워")


def _drop_cross_domain(question, hits):
    """공원·입장료를 묻는데(차·주차 말 없음) 주차장 조례가 근거에 끼면 뺍니다.
    "서울대공원 국가유공자할인과 다둥이 행복카드 할인 둘 다?"에 주차장 조례 제7조 제3항(감면사유 둘 이상이면
    높은 하나만)을 끌어와 공원 입장료에 적용한 일이 있었습니다. 공원 입장료 감면은 도시공원 조례 제20조입니다."""
    if USE_TOPIC_FILTERS and "post" not in LEAN and PARK_Q.search(question) and not CAR_Q.search(question):
        kept = [h for h in hits if "주차장 설치 및 관리 조례" not in h["label"]]
        if kept:
            return kept
    return hits


DUP_INFER = re.compile(r"한\s*번만\s*(적용|감면|할인|받)|두\s*번\s*(감면|할인|받|적용)[^.]{0,10}(않|없|못|아니)"
                       r"|중복(하여|해서)?\s*(적용|감면)되지\s*않")
RACE_NO_RETRY = ("name_mismatch", "annex_only", "off_topic", "no_rule_answer", "no_amount")


def _race(fn, n):
    """fn()을 n번 동시에 돌려 먼저 성공한 (결과, 이유, 사실)을 돌려줍니다. Solar 답이 실행마다 흔들려
    같은 질문이 10번 중 1~3번 확인 불가로 끝나던 것을 줄이려는 장치 (동시에 돌리므로 시간은 거의 같고 호출 수만 늘어남).
    모두 실패면 다음 단계로 이어갈 수 있는 실패(재시도 가능, 확인된 사실 있음)를 고릅니다."""
    if n <= 1:
        return fn()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    pool = ThreadPoolExecutor(n)
    outs = []
    try:
        for fut in as_completed([pool.submit(fn) for _ in range(n)]):
            outs.append(fut.result())
            wins = [o for o in outs if o[0]]
            if len(wins) >= 2:                  # 둘 이상 통과하면 바로 씀
                return wins[0]
    finally:
        pool.shutdown(wait=False)
    wins = [o for o in outs if o[0]]
    # 하나만 통과했을 때: 나머지가 '근거 없음' 신호(못 찾음·규정 없음·질문과 다름·이름 다름 등)를 냈으면
    # 함정 질문에서 한 번 샌 것일 수 있어 쓰지 않습니다 (먼저 통과한 것을 쓰면 함정 54개 중 3→5개로 샜음).
    # 나머지가 검증 탈락·주장 불일치처럼 '답은 있는데 다듬다 떨어진' 경우면 통과한 답을 씁니다.
    hard = [o for o in outs if not o[0] and (o[1] in RACE_NO_RETRY or o[1] in ("not_found", "low_score"))]
    if len(wins) == 1 and not hard:
        return wins[0]
    fails = [o for o in outs if not o[0]]
    fails.sort(key=lambda o: (o[1] in RACE_NO_RETRY, not o[2]))
    if hard and len(wins) == 1:
        return hard[0]
    return fails[0]


_law_flat = {}


def _anchor_law_only(question, hits):
    """중복(둘 다·같이) 질문에서, 질문 낱말 중 6개 조례 가운데 한 조례에만 나오는 말(예: 서울대공원·입장료 → 도시공원 조례,
    공영주차장 → 주차장 조례)이 있고 그 조례가 하나로 모이면 근거를 그 조례로 좁힙니다.
    '서울대공원 국가유공자+다둥이 둘 다?'에 주차장 조례 제7조(주차요금 감면·높은 하나만)가 섞여 주차요금으로 답이 새던 일.
    한 조례로 모이지 않으면(장애인·다둥이처럼 여러 조례에 나오는 말뿐이면) 그대로 둡니다."""
    if not is_multi_q(question):
        return hits
    if not _law_flat:
        for c in _load()[0]:
            _law_flat[c["law"]] = _law_flat.get(c["law"], "") + re.sub(r"\s+", "", c["text"])
    anchors, word_laws = set(), []
    for tok in re.findall(r"[가-힣]{3,}", question):
        forms, w = [tok], tok
        while True:
            w2 = SUBJ_SUFFIX.sub("", w)
            if w2 == w or len(w2) < 3:
                break
            forms.append(w2)
            w = w2
        for f in reversed(forms):                 # 조사를 뗀 기본형부터 ("장애인이"는 주차장 조례에만 있지만 "장애인"은 여러 조례)
            if f in CLAIM_STOP or re.search(r"(이|가|은|는|을|를|면|서|도|요|고|에|의|과|와|랑|며)$", f):
                break                             # 조사·어미가 붙은 꼴("감면이")은 조례 하나를 가리키는 이름이 아님
            laws = [lw for lw, t in _law_flat.items() if f in t]
            if laws:
                word_laws.append(set(laws))
                if len(laws) == 1:
                    anchors.add(laws[0])
                break
    if len(anchors) != 1:
        return hits
    law = anchors.pop()
    # 질문의 다른 말이 그 조례에 없으면(“서울대공원 입장료랑 주차요금”의 주차요금) 여러 조례를 묻는 질문이라 좁히지 않음
    if any(law not in ls for ls in word_laws):
        return hits
    kept = [h for h in hits if h.get("law") == law]
    return kept or hits


# ------------------------------------------------------------ 위임 조항 답변 (2026-11-21 운영 예시 반영)
# "반려견 놀이터 운영 시간"처럼 조례가 세부를 직접 정하지 않고 "그 밖에 ~에 필요한 사항은 시장이 따로 정한다"로 맡긴 경우,
# 확인 불가가 아니라 그 위임 조항을 근거로 "조례에는 정해져 있지 않고 시장이 따로 정한다"고 답합니다.
# 위임 조항이 질문한 바로 그 시설·제도를 맡긴 경우에만 (대상이 다르면 지금처럼 확인 불가).
DELEGATE = re.compile(r"(필요한|세부|구체적인|그\s*밖의|그\s*밖에)[^.。\n]{0,60}?(사항|기준|방법|절차|금액|요금|시간)[은는]?\s*"
                      r"(시장|구청장|시행규칙|규칙)(이|으로|에서)?\s*(따로\s*|별도로\s*)?정한다")
DELEGATE_SYSTEM = """너는 조례 해석 도우미다. 시민 질문과, 조례에서 '그 밖에 필요한 사항은 시장(또는 규칙)이 따로 정한다'고 맡긴 조항 후보를 본다.
covers=true는 다음을 모두 만족할 때만이다.
1) 질문이 묻는 세부 사항(시간·금액·기준·방법 등)이 근거 조항 어디에도 직접 적혀 있지 않다.
2) 후보 조항이 질문한 바로 그 시설·제도·사무(예: 반려견 놀이터의 설치·운영)에 필요한 사항을 맡긴다. 대상이 다르면 false.
3) 질문이 법률·국가 사무나 다른 기관 소관이 아니다.
asked: 질문이 물은 세부를 명사구로 (예: "반려견 놀이터 운영 시간").
matter: 조항이 맡긴 대상을 조항 원문 표현으로 (예: "반려견 놀이터의 설치 및 운영").
check_where: 시민이 구체적인 내용을 확인할 곳을 일반적인 표현으로 (예: "서울시나 해당 놀이터 안내"). 기관명·전화번호를 지어내지 않는다."""
DELEGATE_EXCLUDE = ("out_of_scope", "district_question", "low_score", "name_mismatch", "annex_only", "gu_scope", "unspecified_article")
_DELEGATE_STOP = {"설치", "운영", "관리", "사항", "필요한", "이용", "시설", "등의", "및"}


def _delegate_schema(labels):
    return {"type": "json_schema", "json_schema": {"name": "delegation", "strict": True, "schema": {
        "type": "object", "properties": {
            "covers": {"type": "boolean"}, "label": {"type": "string", "enum": labels},
            "asked": {"type": "string"}, "matter": {"type": "string"},
            "delegate": {"type": "string", "enum": ["시장", "구청장", "규칙", "시행규칙"]},
            "check_where": {"type": "string"}},
        "required": ["covers", "label", "asked", "matter", "delegate", "check_where"], "additionalProperties": False}}}


def _josa(word, pair):
    ch = re.sub(r"[\s)\]」』\"'’”]+$", "", word)[-1:]      # 괄호·따옴표 뒤면 그 앞 글자로 판단
    if ch and "가" <= ch <= "힣":
        return pair[0] if (ord(ch) - 0xAC00) % 28 else pair[1]
    return pair[0]


def _delegation_answer(question, hits):
    """확인 불가로 끝난 질문에서, 질문 대상의 세부를 시장·규칙에 맡긴 위임 조항이 근거에 있으면 그 조항으로 답합니다."""
    cands = list({h["label"]: h for h in hits[:config.MAX_CONTEXT] if DELEGATE.search(h.get("text", ""))}.values())[:6]
    if not cands:
        return None
    import upstage
    ev = "\n\n".join(f"[{h['label']}]\n{h['text'][:900]}" for h in hits[:config.MAX_CONTEXT])
    try:
        out = json.loads(upstage.chat(
            [{"role": "system", "content": DELEGATE_SYSTEM},
             {"role": "user", "content": f"[질문]\n{question}\n\n[위임 조항 후보]\n" + "\n".join(f"- {h['label']}" for h in cands)
                                         + f"\n\n[근거 조항 전체]\n{ev}"}],
            response_format=_delegate_schema([h["label"] for h in cands]), timeout=30))
    except Exception:  # noqa: BLE001
        return None
    h = next((c for c in cands if c["label"] == out.get("label")), None)
    if not out.get("covers") or not h or out.get("delegate", "") not in h["text"]:
        return None
    # 맡긴 대상이 질문과 조항 원문에 실제로 나와야 함 (다른 시설의 위임 조항으로 대신 답하지 않게)
    words = {w for w in re.findall(r"[가-힣]{2,}", out.get("matter", "")) if w not in _DELEGATE_STOP}
    flat_q, flat_t = re.sub(r"\s+", "", question), re.sub(r"\s+", "", h["text"])
    if not words or not any(w[:3] in flat_q for w in words) or not any(w[:3] in flat_t for w in words):
        return None
    m = re.match(r"^(.*?)\s+((?:제\d+조|별표).*)$", h["label"])
    if not m:
        return None
    law = re.sub(r"\s*\(\d{4}-\d{2}-\d{2} 시행본\)", "", m.group(1)).strip()
    asked = out["asked"].strip()
    matter = re.sub(r"\s*(에\s*)?(그\s*밖에\s*)?(필요한\s*)?(사항|것)$", "", out["matter"].strip())   # "~에 필요한 사항" 중복 방지
    where = re.sub(r"\s*에서$", "", out["check_where"].strip()) or "서울시나 담당 부서 안내"
    dp = {"시장": "시장이", "구청장": "구청장이", "규칙": "규칙으로", "시행규칙": "시행규칙으로"}[out["delegate"]]
    ans = (f"조례에는 {asked}{_josa(asked, ('이', '가'))} 정해져 있지 않습니다. 「{law}」 {m.group(2)}에 따라 "
           f"{matter}에 필요한 사항은 {dp} 따로 정하므로, 구체적인 내용은 {where}에서 확인해 주세요.")
    return {"answer": ans, "citation": [h["label"]], "source_url": h.get("source_url", ""),
            "effective_date": h.get("effective_date", ""), "delegated": True}


def answer(question, trace=None, convert=False):
    """확인 불가로 끝나면 위임 조항 답변을 시도합니다 (조례가 세부를 시장·규칙에 맡긴 경우). 나머지는 _answer_core."""
    trace = [] if trace is None else trace
    t0 = time.time()
    box = {}
    res = _answer_core(question, trace, convert, box)
    if res["answer"] == "확인 불가" and box.get("hits") and config.USE_VERIFY             and not any(t.startswith(DELEGATE_EXCLUDE) for t in trace) and time.time() - t0 < 70:
        d = _delegation_answer(question, box["hits"])
        if d:
            trace.append("delegated")
            d.pop("delegated", None)
            return d
    return res


def _answer_core(question, trace=None, convert=False, box=None):
    """trace에 리스트를 넘기면 시도별 결과(성공/실패 이유)를 적어 줍니다. 진단용.
    convert=True는 웹 화면용: "10분 요금"처럼 요금표 단위의 배수를 물으면 환산한 금액으로 답합니다.
    제출용 POST /ask는 False(조례에 적힌 금액 그대로)."""
    trace = [] if trace is None else trace
    t0 = time.time()
    _last_claim_facts.v = ""

    # 6개 조례 밖 다른 조례 소관 질문은 검색 전에 '확인 불가' (빠르고, 비슷한 제도로 대신 답하는 일을 막음)
    if config.USE_VERIFY:
        law = out_of_scope(question)
        if law:
            trace.append(f"out_of_scope:{law}")
            return dict(UNKNOWN)
    if config.USE_VERIFY and GU_NAMES.search(question) and GU_RULE_Q.search(question):
        trace.append("district_question")
        return dict(UNKNOWN)
    hits = _anchor_law_only(question, _drop_cross_domain(question, retrieve(question)))
    if box is not None:
        box["hits"] = hits

    # 검증 ①: 검색 단계에서 이미 관련 조문이 없어 보이면 바로 '확인 불가'
    if config.USE_VERIFY and hits and hits[0]["cos"] is not None and hits[0]["cos"] < config.MIN_SCORE:
        trace.append("low_score")
        return dict(UNKNOWN)

    # 1차는 빠른 모드. 1차가 '확인 불가'로 끝나면 추론 모드(reasoning_effort)로 한 번 더 신중하게 답합니다.
    # Solar의 답이 실행마다 흔들려서, 근거를 찾고도 '확인 불가'로 뒤집히는 일을 줄이기 위한 장치입니다.
    # 함정 문항도 2차를 거치지만, 2차 역시 같은 검증을 통과해야 하므로 지어낸 답은 걸러집니다.
    # 추론 모드는 가끔 40~60초가 걸려서, 질문당 RETRY_DEADLINE_SEC 안에서만 2차를 기다립니다.
    # 1차를 FIRST_TRIES번 동시에 돌려, 검증을 먼저 통과한 답을 씁니다. Solar 답이 실행마다 흔들려
    # 같은 질문이 10번 중 1~3번 확인 불가로 끝나던 것을 줄이려는 장치 (동시에 돌리므로 시간은 거의 같고 호출 수만 늘어남).
    def _one():
        _last_claim_facts.v = ""
        try:
            r, why = _attempt(question, hits, convert=convert)
        except Exception as e:  # noqa: BLE001  응답 형식 오류·일시적 API 오류
            r, why = None, f"error:{type(e).__name__}"
        return r, why, getattr(_last_claim_facts, "v", "")

    result, reason, facts = _race(_one, FIRST_TRIES)
    _last_claim_facts.v = facts
    trace.append(reason or "ok")
    if result or not config.RETRY_ON_UNKNOWN:
        return result or dict(UNKNOWN)
    # 함정 신호(다른 대상, '규정 없음', 금액 없음)면 재시도하지 않음.
    # 단, "5분당 얼마"처럼 단위를 붙여 물은 금액 질문은 전제가 틀렸을 뿐 답이 있을 수 있어 2차를 허용합니다.
    # (검증기가 "5분당 금액을 안 댔다"며 off_topic으로 떨어뜨리는 일도 있어 함께 허용. 2차도 같은 검증을 통과해야 함)
    unit_q = UNIT_Q.search(question) and reason in ("no_rule_answer", "no_amount", "off_topic")
    # "둘 다/같이 받을 수 있나요?"는 요금마다 답이 갈려 답이 길어지고, 검증기가 "질문에 바로 답하지 않았다"(off_topic)로
    # 떨어뜨리는 일이 잦습니다 (같은 질문이 10번 중 1~2번). 2차를 허용하고, 2차도 같은 검증을 통과해야 합니다.
    unit_q = unit_q or (is_multi_q(question) and reason in ("off_topic", "no_rule_answer"))
    # 면제·감면 목록에 실제로 있는 대상을 물은 질문("다둥이 → 운동시설 감면?")도 off_topic이면 2차 허용 (2차도 같은 검증)
    unit_q = unit_q or (reason == "off_topic" and bool(_question_subjects(question, hits)))
    if reason in ("name_mismatch", "annex_only") or (reason in ("off_topic", "no_rule_answer", "no_amount") and not unit_q):
        return dict(UNKNOWN)
    # 2차는 검색도 다시 합니다. 질문 재작성이 실행마다 달라서, 가끔 정답 조가 1차 검색에서 빠집니다
    # (H19 '공원에서 물건 팔기': 대부분 제22조가 3~7위지만 가끔 아예 빠짐). 1차 근거에 새로 찾은 것만 덧붙입니다.
    if config.RETRY_RESEARCH:
        try:
            have = {h["id"] for h in hits}
            extra = [h for h in _anchor_law_only(question, _drop_cross_domain(question, retrieve(question))) if h["id"] not in have]
            hits = hits + extra[:max(0, config.MAX_CONTEXT - len(hits))]
            if box is not None:
                box["hits"] = hits
        except Exception:  # noqa: BLE001  재검색 실패면 1차 근거로 그대로 2차
            pass
    left = config.RETRY_DEADLINE_SEC - (time.time() - t0)
    if left < 10:                               # 남은 시간이 너무 짧으면 2차는 건너뜁니다
        trace.append("skip_retry")
        return dict(UNKNOWN)
    first_facts = getattr(_last_claim_facts, "v", "")

    def _retry_one():
        _last_claim_facts.v = ""
        try:
            if trace[0] == "claim_mismatch" and first_facts:
                # 1차가 대상을 목록에서 잘못 읽어 끝났으면, 확인된 사실을 들고 빠른 모드로 다시 씁니다.
                # 추론 모드는 이런 긴 답에서 40초를 넘겨 시간 초과로 끝나는 일이 잦았고, 사실 없이 다시 쓰면 같은 실수를 반복했습니다.
                # 다시 써도 안 맞으면 재질문 없이 바로 실패(rounds=CLAIM_ROUNDS).
                r, why = _attempt(question, hits, timeout=left, convert=convert,
                                  feedback=first_facts, rounds=CLAIM_ROUNDS)
            else:
                # 추론 모드는 1차가 근거를 못 찾았을 때(not_found)만, 남은 시간이 50초 이상일 때만 씁니다.
                # 검증 탈락 뒤 추론 모드 2차는 25~45초 걸려 시간 초과가 잦았음
                effort = config.RETRY_EFFORT if (left >= 50 and trace[0] == "not_found") else None
                r, why = _attempt(question, hits, effort, timeout=left, convert=convert,
                                  strict=STRICT_RETRY and trace[0] == "not_found")
        except Exception as e:  # noqa: BLE001  시간 초과 등 2차 실패는 '확인 불가'로
            r, why = None, f"retry_error:{type(e).__name__}"
        return r, why, getattr(_last_claim_facts, "v", "")

    result, reason, _ = _race(_retry_one, FIRST_TRIES)   # 2차도 동시에 여러 번 돌려 먼저 통과한 답
    trace.append(reason or "ok")
    # 2차가 응답 형식 오류로 끝났으면(드묾) 시간이 남을 때 빠른 모드로 한 번 더
    left = config.RETRY_DEADLINE_SEC - (time.time() - t0)
    if not result and reason in ("retry_error:JSONDecodeError", "retry_error:TypeError") and left >= 10:
        try:
            result, reason = _attempt(question, hits, timeout=left, convert=convert)
        except Exception as e:  # noqa: BLE001
            result, reason = None, f"retry_error:{type(e).__name__}"
        trace.append(reason or "ok")
    return result or dict(UNKNOWN)


def answer_without_rag(question):
    """시연용 '전' 장면: 검색 없이 LLM만으로 답하게 하면 어떻게 되는가."""
    import upstage
    return upstage.chat([
        {"role": "system", "content": "너는 서울시 조례 전문가다. 답과 함께 근거 조항(제○조 제○항)과 원문 링크, 시행일을 적어라."},
        {"role": "user", "content": question}], temperature=0.3)
