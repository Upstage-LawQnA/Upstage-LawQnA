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
import time
from collections import Counter

import numpy as np

import config

UNKNOWN = {"answer": "확인 불가", "citation": [], "source_url": "", "effective_date": ""}
DISTRICT = re.compile(r"서울특별시\s*\S+구\s")      # 'OO구' 자치구 조례 인용 방지
AMOUNT_Q = re.compile(r"얼마(?!\s*(?:동안|나\s*(?:자주|오래|동안|걸)))")            # 금액·비율을 묻는 질문 ('얼마 동안/얼마나 자주'는 제외)
AMOUNT_A = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:만\s*원|천\s*원|원)|\d+분의\s*\d+|\d+(?:\.\d+)?\s*(?:%|퍼센트)|무료")
NO_AMOUNT = re.compile(r"(금액|요금|비용|액수)[^.]{0,25}(정해져\s*있지\s*않|규정되어\s*있지\s*않|나와\s*있지\s*않|명시되어\s*있지\s*않|정하고\s*있지\s*않|알\s*수\s*없)"
                       # "충전요금이라는 별도의 요금이 없습니다 / 충전요금 자체를 따로 정하지 않고" → 다른 요금(주차요금)으로 대신 답한 것
                       r"|따로\s*정하지\s*않|별도의?\s*(요금|금액|비용|이용료|사용료)\S*\s*(은|는|이|가)?\s*없"
                       r"|(요금|금액|비용|이용료|사용료)(은|는|이|가)\s*(따로\s*|별도로\s*)?없")
COST_Q = re.compile(r"비용|요금|돈|이용료|입장료|사용료|수수료|과태료|벌금|얼마\s*(드|들|내)")   # 금액을 묻는 질문
RATE_Q = re.compile(r"할인|감면|면제|퍼센트|%|비율|몇\s*배")                                   # 비율을 묻는 질문
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
ZONE_FINE = re.compile(r"(금주구역|금연구역)에서")                                     # 과태료 항이 가리키는 구역
UNIT_Q = re.compile(r"\d+\s*(분|시간)\s*(당|에|마다|단위)|시간\s*(당|단위)")              # "5분당/10분에/1시간 단위/시간당"
WH_QUESTION = re.compile(r"얼마|몇|누구|누가|어디|언제|무엇|뭐|뭘|무슨|어떻게|어떤|며칠")   # 예·아니요로 답하지 않는 질문
NEG_END = re.compile(r"(없습니다|않습니다|안 됩니다|안됩니다|불가합니다|아닙니다|못합니다|금지됩니다|금지되어 있습니다)[.]?$")
NOT_NEEDED = re.compile(r"(않아도|안 해도|안 내도|없어도)\s*됩니다|필요(가|는)?\s*없")
# "네, (원칙적으로) 내야 하지만 / 내야 합니다만," 같은 머리말 (감면·면제 답에서 떼어 냄)
PAY_BUT = re.compile(r"^네\s*,\s*(?:원칙적으로\s*|기본적으로\s*|일반적으로\s*)?(?:[가-힣]+(?:은|는|을|를)\s*)?"
                     r"내야\s*(?:하지만|하나|합니다만|하는데)\s*,?\s*")
# 첫 문장이 '규정 없음'이면 확인 불가. 예) D10 "야간 정기권 이용 시간이 정해져 있지 않습니다", D2 "운영 시간은 조례에 정해져 있지 않습니다"
# '않은'(꾸밈말: "기간이 따로 정해져 있지 않은 경우")은 결론이 아니라서 뺍니다 (F12 오탐 방지)
NO_RULE = re.compile(r"(규정|조항|내용)(은|이|도)?\s*((이\s*)?(근거|조례)들?(에는|에)\s*)?(따로\s*)?(없|존재하지 않)"
                     r"|명시(되어|돼)?\s*있지 않(?!은)|정하고 있지 않(?!은)|규정하고 있지 않(?!은)"
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
- 질문에 없는 조례 이름이나 숫자는 지어내지 않는다.
- 검색어 하나는 15단어 이내로 짧게 쓴다.
결과는 JSON으로 낸다."""

REWRITE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "queries", "strict": True,
        "schema": {"type": "object",
                   "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                   "required": ["queries"], "additionalProperties": False},
    },
}


def rewrite(question):
    """질문을 조문 용어로 바꾼 검색어 목록(최대 3개). 실패하면 빈 목록(원래 질문만으로 검색)."""
    if not (config.USE_REWRITE and config.UPSTAGE_API_KEY):
        return []
    import upstage
    try:
        out = upstage.chat([{"role": "system", "content": REWRITE_SYSTEM},
                            {"role": "user", "content": question}], response_format=REWRITE_SCHEMA)
        return [q.strip()[:120] for q in json.loads(out).get("queries", []) if q.strip()][:3]
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
SYSTEM = """너는 서울특별시 조례 질의응답 도우미다.
규칙:
- 반드시 아래 [근거]에 적힌 내용만으로 답한다. 근거에 없는 내용은 추측하지 않는다.
- 질문이 "~되나요/~할 수 있나요/~받을 수 있나요/~내야 하나요/둘 다 되나요"처럼 예·아니요로 답할 질문이면 첫 단어를 "네," 또는 "아니요,"로 시작한다.
  네·아니요는 질문의 동사에 대한 답이다. 예) "수수료를 내야 하나요?"에 안 내도 되면 → "아니요, 내지 않아도 됩니다."
  "얼마/몇/어떻게/무엇" 질문은 네·아니요 없이 바로 답한다.
  · "~내야 하나요?"인데 근거가 그 사람에게 감면·면제를 정하고 있으면 "네, 내야 하지만"으로 시작하지 않는다.
    네·아니요 없이 감면·면제부터 쓰고, 감면에서 빠지는 비용은 다음 문장에 쓴다.
    예) "65세인데 공원 입장료를 내야 하나요?" → "65세 이상이면 입장료를 면제받을 수 있습니다."
  · 근거가 질문보다 좁거나 간접적이면 "네"로 단정하지 않는다. 첫 문장에 근거가 실제로 정한 방식(누가·누구에게·어떤 조건으로)을 쓴다.
    예) "서울시가 보험금을 주나요?"인데 근거는 구청 단체보험 가입비 지원 → "서울시가 보험금을 직접 주는 것은 아니고, 구청이 가입하는 자전거 단체보험의 가입비 일부를 지원할 수 있습니다."
    예) "발견한 유기동물을 개인이 임시 보호해도 되나요?"인데 근거는 센터 보호 동물을 시장이 자원봉사자에게 맡기는 제도
        → "발견한 동물을 바로 데려가 보호하는 제도는 아니고, 동물보호센터에서 보호 중인 유기동물을 시장이 자원봉사자에게 임시보호하게 할 수 있습니다."
  · 근거가 "시장은 ~ 대책(방안)을 수립·시행하여야 한다 / 노력하여야 한다"처럼 시의 정책 의무만 정하고 시민이 받을 권리·대상·금액을 정하지 않으면,
    "네, 받을 수 있습니다"로 단정하지 않는다. 첫 문장에 시가 무엇을 하도록 정했는지 쓰고, 구체적 대상·금액은 그 대책에서 정한다고 덧붙인다.
    예) "청년 취업 지원금 받을 수 있나요?"인데 근거는 '시장은 청년 고용 확대 정책을 시행하여야 한다'
        → "조례는 시장이 청년 고용을 늘리는 정책을 시행하도록 정하고 있어, 그 정책에 따라 지원을 받을 수 있습니다. 구체적인 대상과 금액은 시가 마련하는 사업에서 정합니다."
  · 근거에 이용대상·요건 목록이 있으면, 질문 속 사람이 목록에 그 이름으로 없더라도 목록 기준으로 답한다.
    "직접 정하고 있지 않다/정해져 있지 않다"로 끝내지 않는다.
    예) "가족배려주차장에 장애인이 주차 가능해?" → "장애인이라는 이유만으로 이용대상이 되지는 않고, 고령 등으로 이동이 불편한 사람이거나 그 사람을 동반한 경우에 이용할 수 있습니다."
  · 질문한 행위와 같은 이름의 제도(예: 질문 "임시 보호" ↔ 조례 "임시보호")가 있지만 방식이 다르면
    "~하는 제도는 아니고, 조례의 제도는 ~입니다"처럼 쓴다. "~는 조례에 정해져 있지 않습니다"로 시작하지 않는다.
    같은 이름의 제도가 없으면(예: "고양이를 주워가도 되나요?"에 대해 조례에는 시장의 포획·방사만 있음) 다른 제도를 소개하지 말고 found=false.
- 근거에 없는 권고·의견("~하는 것이 맞습니다/좋습니다", "~에 신고하세요")을 덧붙이지 않는다.
- 절차·방법을 묻는 질문("~하려면?", "어떻게 신청하나요?")은 절차와 요건만 답한다. 위반 시 과태료, 사무 위임처럼 절차가 아닌 내용은 쓰지 않고 인용하지 않는다.
  예) "공원에 금연구역을 설치하려면?" → 지정 주체(시장)·의견 수렴·고시·안내표지판만. 흡연 금지·과태료·과태료 사무 위임은 쓰지 않는다.
  첫 문장은 질문에 바로 답한다. 예) "공원에 금연구역을 두려면 시장이 그 공원을 금연구역으로 지정해야 합니다."
- 답은 일반 시민이 이해할 수 있는 쉬운 말로 2~4문장으로 쓴다. 금액·비율·나이 같은 숫자는 그대로 쓴다.
- 질문이 묻지 않은 부수 내용(이의제기 절차, 과태료 가중·감경 기준, 용어 정의 등)은 쓰지 않는다.
- 질문이 묻는 시설·대상(예: 주차장)과 다른 시설의 혜택(예: 공원 입장료 면제, 공공자전거 요금 감면)은 같은 사람에 관한 것이라도 쓰지 않고 evidence에 넣지 않는다.
  예) "임산부가 받을 수 있는 주차 혜택은?" → 가족배려주차장과 공영주차장 요금 할인만. 공원 입장료 면제는 쓰지 않는다.
  "임산부가 이용할 수 있는 주차장은?"처럼 이용만 물으면 이용대상과 있는 곳만 쓰고, 설치 위치 기준·요금 할인은 쓰지 않는다.
- 「」 안 법률 이름에 들어 있는 낱말만 보고 그 사람이 그 제도의 대상이라고 판단하지 않는다. 임산부 질문에 장애인전용주차구획을 쓰지 않는다.
- "○○의 혜택은 뭐가 있나요?"처럼 혜택을 나열하는 질문에서는, 그 사람에게 해당하지 않는 다른 제도를
  "~에는 정해져 있지 않습니다/해당하지 않습니다"처럼 덧붙이지 않는다. (이 규칙은 found 판단과 무관하다)
- 대상 목록의 한 호에 여러 사람이 함께 적혀 있어도(예: "6세 미만 또는 65세 이상인 사람과 장애인") 질문한 사람만 쓴다.
  "65세 이상인 사람과 함께 장애인이 면제받습니다"처럼 엮으면 동반해야 하는 것처럼 읽히므로 쓰지 않는다. 예) "장애인은 도시공원 입장료를 면제받을 수 있습니다."
- 주차요금 감면·할인(주차장 조례 제7조)은 공영주차장(시장이 설치한 노상·노외주차장) 요금에 대한 것이다. 감면을 쓸 때 "공영주차장 주차요금의 50% 할인"처럼 공영주차장임을 밝힌다.
- 답 본문에 조례 이름·조항 번호(제○조제○항제○호)를 늘어놓지 않는다. 인용 조항은 따로 붙으므로 본문은 내용만 쉬운 말로 쓴다.
- "어떻게 되나요" 질문은 해당 행위가 언제 금지되는지(지정 요건)와 위반하면 어떤 결과(과태료 금액 등)인지를 짧게 답한다.
  이때 evidence에는 지정 조항(예: 금주구역·금연구역 지정)과 과태료 조항을 둘 다 넣는다. 과태료 가중·감경, 이의제기 조항은 넣지 않는다.
- 구역에 따라 달라지는 금지·과태료(금주구역·금연구역)를 쓸 때는 지정 조항이 정한 장소를 그대로 쓴다. 더 넓은 장소로 바꾸지 않는다.
  예) 금주구역 지정 조항이 "어린이공원을 금주구역으로 지정할 수 있다"이면 → "금주구역으로 지정된 어린이공원에서 술을 마시면 과태료 10만원입니다."
      ("공원에서 술을 마시면", "금주구역으로 지정한 도시공원에서", "도시공원 안에서 술을 마시면 금주구역 음주로 보아"처럼 쓰지 않는다.
       공원에서 술을 마시는 것 자체가 과태료 대상인 것은 아니고, 금주구역으로 지정된 어린이공원에서 마실 때만 과태료다.)
- 원칙 조항만 보지 말고, 근거에 있는 예외·단서(다만)·중복 적용 제한·감면율·과태료 조항도 확인해 답에 반영한다.
  단서가 특정 호·경우에만 걸리면 그 경우에만 쓴다. 다른 경우에 대해 "단서가 적용되지 않는다/유예 없이 바로 부과한다"처럼
  근거에 없는 반대 결론을 추론해 쓰지 않는다. 예) 유예기간이 '주차쿠폰을 쓰지 않은 경우'에만 있으면, 요금을 안 내고 간 경우의 답에는 유예기간을 쓰지 않는다.
  예) 감면을 둘 이상 묻는 질문 → 중복 적용을 정한 항이 있는지 본다. 높은 하나만 적용되면 각 감면율과 실제로 적용되는 쪽을 쓴다
      ("장애인 80%, 다둥이 50%이므로 장애인 할인 80%만 적용됩니다"). 금지·지정을 묻는 질문 → 위반 시 과태료 항이 있는지 본다.
- evidence에는 답의 내용을 직접 뒷받침하는 근거만 넣는다. 질문의 답(날짜·금액·대상 등)이 실제로 적힌 조항을 고르고, 같은 낱말을 언급만 하는 옆 조항은 고르지 않는다. 답에 쓴 조항은 빠짐없이 넣되, 용어 정의·목적 조항처럼 답에 쓰지 않은 근거는 넣지 않는다.
- 요금·금액은 별표에 있다. 별표에서 금액을 가져왔다면 그 별표 번호도 넣는다.
- 질문이 묻는 내용이 근거에 없으면 found를 false로 한다. 비슷한 다른 내용으로 대신 답하지 않는다.
  · 질문의 대상(시설·요금 종류·위반행위)과 근거의 대상이 정확히 같은지 확인한다. 이름이 비슷한 다른 대상이면 found=false.
  · 질문한 행위와 근거가 규제하는 행위가 같은지 확인한다. 판매자·영업자(상행위·영업)에 대한 규정을 사 먹기·주문하기 같은 이용자의 행위에 적용하지 않는다. 다르면 found=false.
  · 행위 주체를 확인한다. 질문은 시민 개인이 어떤 행위를 해도 되는지 묻는데, 근거는 시장·구청장 같은 행정기관이 하는 일
    (포획·방사, 급식소 설치, 개체 수 관리 등)만 정하고 시민의 그 행위를 허용·금지하는 문장이 없으면 found=false.
    행정기관의 권한·의무를 "시민은 ~하면 안 됩니다/~해도 됩니다"로 바꾸지 않는다.
    예) "길고양이에게 밥을 줘도 되나요?" ↔ 근거는 시장·구청장의 급식소 설치뿐 → 시민의 밥 주기와 같은 제도가 아니므로 found=false.
    "허용되지 않습니다·금지됩니다"는 근거에 "누구든지/소유자등은 ~하여서는 아니 된다"처럼 그 행위를 금지하는 문장이 있을 때만 쓴다.
  · 근거가 다른 조례를 가리키기만 하고(예: "「서울특별시 출산 및 양육지원에 관한 조례」에 따른 다자녀 가족")
    질문이 묻는 기준·대상·금액을 직접 정하지 않으면 found=false. 다른 조례 이름을 답으로 내놓지 않는다.
    예) "다자녀 가족 지원 대상은 누구인가요?" ↔ 근거는 공공자전거 감면 대상에 '다른 조례에 따른 다자녀 가족'이 들어 있다는 것뿐 → found=false.
  · 질문이 근거에 없는 이름의 시설·제도(예: 임산부 전용주차구역, 여성전용주차장, 노인 전용 주차구역)를 물으면 found=false.
    비슷한 다른 제도(예: 가족배려주차장 주차구획)로 바꿔 답하지 않는다. 그 이름의 제도는 이 근거 밖의 다른 조례에 따로 있을 수 있다.
    예) "임산부 전용주차구역 설치 가능?" → 근거에는 가족배려주차장만 있고 임산부 전용주차구역 규정은 없음 → found=false.
  · 질문이 묻는 금액·시간·벌칙이 근거에 없어서 답이 "규정이 없다/명시되어 있지 않다/따로 정한다"가 될 상황이면 found=false.
  · 예외(전제 바로잡기): 질문이 같은 대상의 요금을 잘못된 단위·기준으로 물었을 뿐이고(예: 1시간 단위 요금을 "5분당 얼마"로 물음)
    근거에 바로 그 대상의 실제 요금·기준이 있으면 found=true로 하고 전제를 바로잡아 답한다.
    - 첫 문장에 근거의 실제 기준을 원칙→예외 순서로 쓴다. 예) "자전거주차장은 원칙적으로 무료이고, 요금을 받는 경우에도 5분 단위가 아니라 1시간당 200원 범위입니다."
    - "정해져 있지 않다/규정이 없다"로 쓰지 말고 "○○ 단위가 아니라 △△ 기준입니다"처럼 쓴다.
    - 근거 시간보다 짧은 시간을 물으면(예: 1시간당 200원인데 "10분에 얼마") 나누어 계산하지 않는다. "10분에 100원"처럼 근거에 없는 금액을 만들지 않는다.
    - [계산 안내]가 붙어 있으면 그 안내를 따른다.
    - 질문 대상의 요금 자체가 근거에 없고 다른 대상(예: 입양비를 물었는데 보호비용만 있음)의 요금만 있으면 예외가 아니다 → found=false.
- 질문이 특정 자치구(OO구)의 조례·규칙·요금·시설을 묻는 것이면 found=false. 서울특별시 조례로 대신 답하지 않는다.
- 자치구(OO구) 조례나 규칙은 근거로 쓰지 않는다.
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
                },
                "required": ["found", "answer", "evidence"],
                "additionalProperties": False,
            },
        },
    }


AMEND_TAG = re.compile(r"\s*<(?:개정|신설|전문개정)[^>]*>|\s*\[(?:본조신설|전문개정|제목개정|종전|제\d+조에서)[^\]]*\]")


def _context(hits):
    def body(t):          # 개정 이력 꼬리표는 답에 쓸 일이 없어 빼고 넣을 수 있게 합니다 (CLEAN_AMEND)
        return AMEND_TAG.sub("", t) if config.CLEAN_AMEND else t
    if config.EVIDENCE_KEY == "index":
        return "\n\n".join(f"[근거 {i}] {h['label']}\n{body(h['text'])}" for i, h in enumerate(hits, 1))
    return "\n\n".join(f"[근거: {h['label']}]\n{body(h['text'])}" for h in hits)


Q_TIME = re.compile(r"(\d+)\s*(분|시간)")                      # 질문의 "10분당 / 30분 세우면 / 1시간"
BASE_TIME = re.compile(r"(\d+)?\s*(분|시간)\s*당")                # 근거 요금표의 "5분당 / 1시간당 / 시간당"


def _unit_ratios(question, hits):
    """질문 시간(분)과 근거 요금표 단위(분)들. 금액·단위를 묻는 질문이 아니면 (None, [])."""
    m = Q_TIME.search(question)
    if not m or not (AMOUNT_Q.search(question) or UNIT_Q.search(question)):
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


# "임산부가 이용 가능한 주차장은?"처럼 '어디를 이용할 수 있나'를 물으면 이용대상(제25조의2 제2항)과
# 있는 곳(제1항)만 답합니다. 설치 위치 기준(제4항: 사각 없는 곳·CCTV·승강기 근처)·확장형 비율(제3항)·
# 요금 할인(제7조)은 묻지 않은 내용이라 문장(절)과 인용을 뺍니다. 요금·할인을 함께 물으면 그대로 둡니다.
USE_Q = re.compile(r"이용\s*(가능|할\s*수\s*있|해도\s*되)")
FEE_Q = re.compile(r"요금|할인|감면|돈|얼마|무료|비용|혜택")
UNASKED_FACETS = [   # (답 문장 속 표현, 그 내용의 인용 라벨)
    (re.compile(r"할인|감면|주차요금"), re.compile(r"주차장 설치 및 관리 조례 제7조")),
    (re.compile(r"사각|폐쇄회로|CCTV|승강기|계단|인접한|확장형|통행이\s*잦"),
     re.compile(r"주차장 설치 및 관리 조례 제25조의2 제[34]항")),
]


def _drop_unasked_facets(question, ans, used):
    if not USE_Q.search(question) or FEE_Q.search(question):
        return ans, used
    for sent_pat, label_pat in UNASKED_FACETS:
        if not any(label_pat.search(h["label"]) for h in used):
            continue
        new = _remove_sentences(ans, sent_pat)
        if new is None:
            continue
        rest = [h for h in used if not label_pat.search(h["label"])]
        if rest:
            ans, used = new, rest
    return ans, used


def _remove_sentences(ans, pat):
    """pat이 든 문장을 뺍니다. 다른 내용과 한 문장이면 그 절만 뺍니다. 다 지워지거나 바뀐 게 없으면 None."""
    parts = re.split(r"((?<=[.!?])\s+)", ans)
    sents, seps = parts[0::2], parts[1::2] + [""]
    keep = []
    for s, sep in zip(sents, seps):
        if not pat.search(s):
            keep.append((s, sep))
        elif (t := _trim_clause(s, pat)):
            keep.append((t, sep))
    if keep == list(zip(sents, seps)) or not any(s.strip() for s, _ in keep):
        return None
    return "".join(s + sep for s, sep in keep).strip()


DELEGATE_SENT = re.compile(r"따로\s*정(합니다|해집니다|하며|하고)")


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
    for sent_pat, asked, label_pat in UNASKED_FACILITY:
        if asked.search(question):
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
        if keep == list(zip(sents, seps)) or not any(s.strip() for s, _ in keep):
            continue
        ans = "".join(s + sep for s, sep in keep).strip()
        rest = [h for h in used if not label_pat.search(h["label"])]
        if rest:
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


def _attempt(question, hits, reasoning_effort=None, timeout=None, convert=False):
    """Solar 답변 1회 + 검증. 성공하면 (결과, None), 실패하면 (None, 실패 이유)."""
    import upstage
    raw = upstage.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"[근거]\n{_context(hits)}\n\n[질문]\n{question}{_unit_hint(question, hits, convert)}"}],
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
    # 예·아니요(해도 되나요) 질문을 별표(요금표)만으로 답했으면, 허용·금지를 정한 조문 없이 요금이 있다는 것만으로
    # 추론한 것 → 확인 불가. (U14 "공원에 텐트 쳐도 되나요?" → 캠핑장 텐트 대여료만 보고 "네")
    # 금액·몇·얼마 질문은 해당 없음 (S1처럼 별표가 바로 답).
    yes_no = not WH_QUESTION.search(question) and not AMOUNT_Q.search(question) and not UNIT_Q.search(question)
    # 별표와 함께 무관한 본문 조항을 곁들여도 마찬가지입니다. (U14가 별표 2 + 제10조 제1항 '공원점용허가 기준
    # (노외주차장·가설건축물)'을 인용해 이 검사를 피했음) → 답과 낱말이 2개 이상 겹치는 본문 조항이 없으면 별표만으로 답한 것
    if out.get("found") and used and yes_no and any("별표" in h["label"] for h in used):
        aw = {w for w in _hangul_words(out["answer"]) if len(w) >= 3}
        body = [h for h in used if "별표" not in h["label"]]
        if all(len(aw & {w for w in _hangul_words(_label_text(h["label"]) or h["text"]) if len(w) >= 3}) < 2
               for h in body):
            return None, "annex_only"
    used = _with_principle(used)
    used = _with_reduction_list(used)
    used = _with_zone_designation(used, hits)
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
    if tq and (NAME_ACK.search(out["answer"]) or not _target_in_evidence(tq.group(1), used)):
        return None, "name_mismatch"
    # 방식 바로잡기: "임시 보호하는 제도는 조례에 정해져 있지 않습니다. 조례는 센터 동물을 자원봉사자에게…"처럼
    # 말한 제도(임시보호)가 고른 근거에 실제로 있으면 '규정 없음'이 아니라 방식이 다르다는 뜻 (U22).
    # 근거에 없는 행위(U19 "주워가는 제도")는 해당 없음.
    def method_fix(s):
        return _method_fix(s, used)

    head_no_rule = (NO_RULE.search(sents[0]) and not name_ack and not method_fix(sents[0])) \
        or STRONG_NO_RULE.search(out["answer"])      # "허용하는 규정은 아니고" 류는 어느 문장에 있어도 결론
    # 뒤 문장의 "다만 구체적인 대상·금액 기준은 조례에 직접 정해져 있지 않고 …"는 세부 사항 덧붙임이라 결론이 아님 (청년 월세)
    tail_no_rule = any(NO_RULE.search(s) and re.search(r"^따라서|근거|조례", s) and not method_fix(s)
                       and not re.search(r"구체적|세부", s) for s in sents[1:])
    if head_no_rule and not premise_fix:
        return None, "no_rule_answer"
    # 결론(첫 문장)은 답했는데 뒤에서 '조례에 따로 정해져 있지 않다'로 뒤집은 경우: 막되 2차는 허용합니다.
    # (F13 "네, 신용카드로 낼 수 있습니다. … 다만 입장료 자체를 신용카드로 내는 방법은 조례에 따로 정해져 있지 않습니다.")
    if tail_no_rule and not premise_fix:
        return None, "no_rule_tail"
    # '얼마' 질문인데 답에 구체적인 금액·비율이 없으면, 조례에 금액이 없어서 돌려 말한 것 → '확인 불가'
    # (예: 입양 비용이 얼마냐는 질문에 "보호기간에 따라 달라지며 청구할 수 있다"처럼 금액 없이 답하는 경우)
    if AMOUNT_Q.search(question) and (not AMOUNT_A.search(out["answer"])
                                      or (NO_AMOUNT.search(out["answer"]) and not premise_fix)):
        return None, "no_amount"
    # 비용·요금을 물었는데(할인·감면율 질문 제외) 답에 '○원'이나 '무료'가 없고 비율만 있으면, 다른 항목의
    # 감면율로 대신 답한 것 (D6 "입양 비용 얼마?" → 등록수수료 50퍼센트 감면·보호비용 청구 가능만 나열)
    if AMOUNT_Q.search(question) and COST_Q.search(question) and not RATE_Q.search(question) \
            and not WON_A.search(out["answer"]):
        return None, "no_amount"

    # 절차 질문("~하려면/할려면", 절차, 방법)은 절차·요건만: 과태료·사무 위임 문장을 검증 전에 뺍니다.
    # U5 "공원에 금연구역을 설치할려면?" 답 끝에 "과태료 부과·징수 사무는 … 위임합니다"가 붙으면 검증기가
    # offTopic으로 두 번 떨어뜨려(재시도도 없음) 확인 불가가 되던 일. 프롬프트 규칙만으로는 가끔 안 지켜짐.
    if PROC_Q.search(question) and not re.search(r"과태료|위반|벌", question):
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
        verdict = upstage.groundedness(vctx, out["answer"], question)
        # 탈락 판정이면 한 번 더 물어 두 번 다 탈락일 때만 떨어뜨립니다.
        # 멀쩡한 답이 우연히 떨어지는 일(채점 1회당 1~2건: U4·U5·F03·F25·H18)은 두 번 연속 나오기 어렵고,
        # 함정 답은 두 번 다 탈락이 나옵니다.
        if verdict in ("notGrounded", "offTopic") and config.VERIFY_SECOND_OPINION:
            second = upstage.groundedness(vctx, out["answer"], question)
            if second not in ("notGrounded", "offTopic"):
                verdict = second
        if verdict == "notGrounded":            # notSure는 통과시킵니다 (떨어뜨리면 멀쩡한 답이 더 많이 사라짐)
            return None, "not_grounded"
        if verdict == "offTopic":               # 근거는 맞지만 질문과 다른 대상을 답함 → 함정 방어
            return None, "off_topic"

    used = _with_amount_sources(out["answer"], used, hits)
    citation = list(dict.fromkeys(h["label"] for h in used))   # 순서 유지, 중복 제거
    first = used[0]
    ans = out["answer"].strip()
    if WH_QUESTION.search(question):           # '얼마/누구/언제…' 질문인데 "네,"로 시작하면 떼어 냅니다
        ans = re.sub(r"^(네|아니요|예)\s*[,.]\s*", "", ans)
    elif re.match(r"^네\s*,", ans):            # "네, …할 수 없습니다"처럼 첫 문장이 부정이면 "아니요,"가 맞습니다
        s1 = re.split(r"(?<=[.!?])\s", ans, maxsplit=1)[0].strip()
        # "~내야 하나요?"에 "네, 내지 않아도 됩니다"도 방향이 틀린 경우입니다
        must_q = re.search(r"야\s*(하나요|해요|되나요|돼요|합니까)", question)
        if must_q and re.search(r"감면|면제", s1) and PAY_BUT.match(ans):
            # C3 "네, 내야 하지만 유기견을 입양하여 등록하면 전액 감면…" → 감면부터 말합니다
            ans = PAY_BUT.sub("", ans, count=1)
        elif NEG_END.search(s1) or (must_q and NOT_NEEDED.search(s1)):
            ans = re.sub(r"^네\s*,", "아니요,", ans)
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
        if re.search(r"하나만\s*적용", s1):
            ans = "아니요, 둘 다 받을 수는 없고 감면율이 높은 하나만 적용됩니다." + (" " + rest[0] if rest else "")
        elif not s1.startswith("아니요"):
            ans = "아니요, " + re.sub(r"^네\s*[,.]\s*", "", ans)
    # 조문 말투를 쉬운 말로: "다음 각 호의 어느 하나에 해당하는 사람" → "다음에 해당하는 사람".
    # 목록 번호 ①②③은 조례의 '항' 기호와 같아 제1항·제2항으로 읽힐 수 있어 1) 2) 3)으로 바꿉니다.
    ans = re.sub(r"다음\s*각\s*호의\s*어느\s*하나에\s*해당하는\s*", "다음에 해당하는 ", ans)
    ans = re.sub(r"다음\s*각\s*호의\s*", "다음 ", ans)
    ans = re.sub(r"[①-⑳]", lambda m: f"{ord(m.group()) - 0x2460 + 1}) ", ans).replace(")  ", ") ")
    ans, used = _drop_unasked_facility(question, ans, used)
    ans, used = _drop_unasked_facets(question, ans, used)
    ans, used = _drop_other_person_benefits(question, ans, used)
    ans = _drop_ungrounded_delegation(ans, used)
    used = _drop_offtopic_paragraphs(ans, used)
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


def answer(question, trace=None, convert=False):
    """trace에 리스트를 넘기면 시도별 결과(성공/실패 이유)를 적어 줍니다. 진단용.
    convert=True는 웹 화면용: "10분 요금"처럼 요금표 단위의 배수를 물으면 환산한 금액으로 답합니다.
    제출용 POST /ask는 False(조례에 적힌 금액 그대로)."""
    trace = [] if trace is None else trace
    t0 = time.time()

    # 6개 조례 밖 다른 조례 소관 질문은 검색 전에 '확인 불가' (빠르고, 비슷한 제도로 대신 답하는 일을 막음)
    if config.USE_VERIFY:
        law = out_of_scope(question)
        if law:
            trace.append(f"out_of_scope:{law}")
            return dict(UNKNOWN)
    hits = retrieve(question)

    # 검증 ①: 검색 단계에서 이미 관련 조문이 없어 보이면 바로 '확인 불가'
    if config.USE_VERIFY and hits and hits[0]["cos"] is not None and hits[0]["cos"] < config.MIN_SCORE:
        trace.append("low_score")
        return dict(UNKNOWN)

    # 1차는 빠른 모드. 1차가 '확인 불가'로 끝나면 추론 모드(reasoning_effort)로 한 번 더 신중하게 답합니다.
    # Solar의 답이 실행마다 흔들려서, 근거를 찾고도 '확인 불가'로 뒤집히는 일을 줄이기 위한 장치입니다.
    # 함정 문항도 2차를 거치지만, 2차 역시 같은 검증을 통과해야 하므로 지어낸 답은 걸러집니다.
    # 추론 모드는 가끔 40~60초가 걸려서, 질문당 RETRY_DEADLINE_SEC 안에서만 2차를 기다립니다.
    try:
        result, reason = _attempt(question, hits, convert=convert)
    except Exception as e:  # noqa: BLE001  응답 형식 오류(JSONDecodeError 등)·일시적 API 오류는 2차로 한 번 더
        result, reason = None, f"error:{type(e).__name__}"
    trace.append(reason or "ok")
    if result or not config.RETRY_ON_UNKNOWN:
        return result or dict(UNKNOWN)
    # 함정 신호(다른 대상, '규정 없음', 금액 없음)면 재시도하지 않음.
    # 단, "5분당 얼마"처럼 단위를 붙여 물은 금액 질문은 전제가 틀렸을 뿐 답이 있을 수 있어 2차를 허용합니다.
    # (검증기가 "5분당 금액을 안 댔다"며 off_topic으로 떨어뜨리는 일도 있어 함께 허용. 2차도 같은 검증을 통과해야 함)
    unit_q = UNIT_Q.search(question) and reason in ("no_rule_answer", "no_amount", "off_topic")
    if reason in ("name_mismatch", "annex_only") or (reason in ("off_topic", "no_rule_answer", "no_amount") and not unit_q):
        return dict(UNKNOWN)
    # 2차는 검색도 다시 합니다. 질문 재작성이 실행마다 달라서, 가끔 정답 조가 1차 검색에서 빠집니다
    # (H19 '공원에서 물건 팔기': 대부분 제22조가 3~7위지만 가끔 아예 빠짐). 1차 근거에 새로 찾은 것만 덧붙입니다.
    if config.RETRY_RESEARCH:
        try:
            have = {h["id"] for h in hits}
            extra = [h for h in retrieve(question) if h["id"] not in have]
            hits = hits + extra[:max(0, config.MAX_CONTEXT - len(hits))]
        except Exception:  # noqa: BLE001  재검색 실패면 1차 근거로 그대로 2차
            pass
    left = config.RETRY_DEADLINE_SEC - (time.time() - t0)
    if left < 10:                               # 남은 시간이 너무 짧으면 2차는 건너뜁니다
        trace.append("skip_retry")
        return dict(UNKNOWN)
    try:
        result, reason = _attempt(question, hits, config.RETRY_EFFORT, timeout=left, convert=convert)
    except Exception as e:  # noqa: BLE001  시간 초과 등 2차 실패는 '확인 불가'로
        result, reason = None, f"retry_error:{type(e).__name__}"
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
