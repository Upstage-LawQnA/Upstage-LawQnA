"""시연용 민원 처리 흐름: 질문 분석 → 자치법규 검색 → 답변(제출용과 같은 rag.answer) → 유형 분류 → 담당자 연결.

개발제안서 2-3·2-4의 흐름을 그대로 따릅니다.
  간단 질문  : 조례 한 조항에 답이 분명함 → 근거와 함께 바로 답변
  복잡 질문  : 여러 조항을 함께 봐야 함 → 답변하되 근거 묶음을 붙여 담당자 검토로 보낼 수 있음
  회색지대   : 조문은 있지만 세부가 정해져 있지 않아 해석이 갈림 → 담당자 확인 권장
  근거 없음  : 선택 지역 자치법규에 관련 규정이 없음 → 지어내지 않고 담당 부서로 연결
  앞단 분류  : 조례 질의 / 국가 법령 질의 / 사업·운영 문의 / 현장 민원 / 건의·제안 (현장 민원·건의·제안은 조례 검색 없이 담당 부서 연결,
               나머지는 엄격 답변 후 근거가 없으면 분류에 맞춰 국가 법령·담당 부서 안내)
"""
import json
import re
import threading
import time
from datetime import datetime

import config
import corpus
import rag

# ------------------------------------------------------------ 개인정보 가리기 (제안서 7장)
PII = [
    (re.compile(r"\d{6}\s*-\s*[1-4]\d{6}"), "[주민등록번호]"),
    (re.compile(r"01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}"), "[전화번호]"),
    (re.compile(r"0\d{1,2}-\d{3,4}-\d{4}"), "[전화번호]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[이메일]"),
    (re.compile(r"(?<!\d)\d{2,3}\s?[가-힣]\s?\d{4}(?!\d)"), "[차량번호]"),
]


def mask(text):
    hits = []
    for pat, rep in PII:
        if pat.search(text):
            hits.append(rep.strip("[]"))
            text = pat.sub(rep, text)
    return text, hits


# ------------------------------------------------------------ 1) 질문 분석
TRIAGE_SYSTEM = """너는 지방자치단체 민원 창구의 접수 도우미다. 시민이 쓴 글을 읽고 분류해 JSON으로 낸다.
- kind: 다섯 가지 중 하나
  "조례 질의" = 지방자치단체 조례·규칙에 정해질 내용을 묻는 질문 (요금·사용료·감면·면제·지원 대상·자격·허용·금지·과태료 금액·
     기준·정의·나이·시행일, 시 기관·부서의 업무와 권한). 국가 법률과 함께 걸려 있어도 조례·규칙이 정한 부분을 묻는다면 여기.
  "국가 법령 질의" = 법률·시행령·국가 시행규칙·고시가 정하는 내용을 묻는 질문 (예: 건축법·소방시설법·도로교통법·주택법상 기준,
     국가 자격·인허가 요건, 법률 위반 여부). 조례로는 정하지 않는 사항.
  "사업·운영 문의" = 법규가 아니라 사업·시설 운영 정보를 묻는 질문 (예: 사업 신청 기간·방법, 시행 계획·일정, 시설 운영시간·위치,
     관할 구역, 예산·진행 상황, 홈페이지·시스템 사용법).
  "현장 민원" = 현장 확인·조치가 필요한 신고·요청 (예: 불법 주정차 단속, 시설 고장·파손, 쓰레기·소음·악취, 청소, 안전 위험).
  "건의·제안" = 제도·정책을 바꾸거나 새로 만들어 달라는 의견, 시설 설치·개선 건의, 칭찬·불만.
- keywords: 관련 자치법규를 찾을 검색어 2~3개. 조례 제목이나 조문에 나올 짧은 명사(2~8자). 일상어는 조문 용어로 바꾼다.
  예) 차 세우는 데 → 주차장, 담배 → 금연, 강아지 → 동물보호, 벌금 → 과태료. 지역 이름·'조례'라는 말은 넣지 않는다.
  질문에 제도·카드·시설의 고유한 이름(예: 다둥이 행복카드, 서울형 키즈카페)이 있으면 바꾸지 말고 그 이름을 그대로 하나 넣는다.
  조례 질의가 아니어도 담당 부서를 찾을 수 있게 관련 분야 검색어를 넣는다 (예: 가로등 고장 → 가로등, 도로).
- title_words: 이 질문을 다루는 자치법규의 '제목'에 들어 있을 낱말 1~3개 (2~6자). 질문 낱말이 아니라 조례 이름에 쓰이는 말.
  조례 제목은 시설·제도의 가장 일반적인 이름을 쓰므로, 구체적인 말(공영주차장·노상주차장·주차요금·시립공원)이 아니라
  그 상위 낱말(주차장·공원)을 반드시 하나 넣는다. 특정 제도 이름이 따로 있을 법하면 그 이름도 함께 넣는다.
  예) 경차 주차요금 할인 → 주차장 / 공영주차장 요금 → 주차장 / 임산부 전용주차구역 → 임산부, 주차 / 담배 → 금연 /
      강아지 등록 → 동물보호 / 공원 입장료 → 공원 / 청년 나이 → 청년 / 가로등 고장 → 가로등, 도로
- title: 민원 제목으로 쓸 한 줄 요약 (30자 이내, 존댓말 아님).
- issue: 담당자가 알아야 할 핵심 요청 한 문장.
- question: 글이 낱말만 늘어놓았거나 서술어가 없으면(예: "공영주차장 요금"), 시민이 물으려는 뜻을 바꾸지 않고
  완전한 질문 한 문장으로 다시 쓴다 (예: "공영주차장 요금은 얼마인가요?"). 글에 없는 대상·조건·숫자는 더하지 않는다.
  이미 완전한 문장이면 그대로 옮긴다."""

TRIAGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "triage", "strict": True, "schema": {
        "type": "object",
        "properties": {"kind": {"type": "string", "enum": ["조례 질의", "국가 법령 질의", "사업·운영 문의", "현장 민원", "건의·제안"]},
                       "keywords": {"type": "array", "items": {"type": "string"}},
                       "title_words": {"type": "array", "items": {"type": "string"}},
                       "title": {"type": "string"}, "issue": {"type": "string"}, "question": {"type": "string"}},
        "required": ["kind", "keywords", "title_words", "title", "issue", "question"], "additionalProperties": False}},
}


# 문장으로 끝난 입력: 물음표·마침표, 또는 '~요/까/다/나/니/죠/지/냐/가/래/어/아/해/돼/되/야' 같은 어미
SENTENCE_END = re.compile(r"([?？.!]|(요|까|다|나|니|죠|지|냐|가|래|어|아|해|돼|되|야|임|음))\s*[?？.!~]*\s*$")


def triage(question):
    import upstage
    try:
        out = json.loads(upstage.chat([{"role": "system", "content": TRIAGE_SYSTEM},
                                       {"role": "user", "content": question}], response_format=TRIAGE_SCHEMA))
        kws = [k.strip() for k in out.get("keywords", []) if k.strip()][:3]
        tws = [k.strip() for k in out.get("title_words", []) if k.strip()][:3]
        # 서술어 없는 낱말 입력만 완전한 질문으로 바꿔 씁니다 (답변 단계는 완전한 질문을 전제로 만들어져 있어
        # "공영주차장 요금"처럼 물으면 근거를 찾고도 '확인 불가'로 끝났음). 문장으로 물은 질문은 그대로 둡니다.
        restated = (out.get("question") or "").strip()
        normalized = restated if (restated and not SENTENCE_END.search(question) and restated != question) else ""
        return {"kind": out.get("kind", "조례 질의"), "keywords": kws or [question[:20]], "title_words": tws,
                "title": out.get("title", question[:30]), "issue": out.get("issue", ""), "normalized": normalized}
    except Exception:  # noqa: BLE001  분석이 실패해도 질문 자체로 검색합니다
        return {"kind": "조례 질의", "keywords": [question[:20]], "title_words": [], "title": question[:30], "issue": "",
                "normalized": ""}


# 앞단 분류(5가지) → 처리. 질문 세 종류(조례·국가 법령·사업 운영)는 서로 자주 엇갈려서(따릉이 요금 → '사업·운영',
# 헬멧 과태료 → '국가 법령') 모두 조례 검색·엄격 답변을 거치고, 분류는 근거를 못 찾았을 때 안내(국가 법령·담당 부서)에만 씁니다.
# 현장 민원·건의·제안만 조례 검색 없이 담당 부서로 바로 넘깁니다 (응답소 표본에서 서울시 조례 질문이 이리로 간 일 0건).
ANSWER_KINDS = {"조례 질의", "국가 법령 질의", "사업·운영 문의"}
KIND_TYPE = {"현장 민원": "field", "건의·제안": "proposal"}
UNKNOWN_TYPE = {"국가 법령 질의": "national", "사업·운영 문의": "business"}
# 분류는 '막는 용도'가 아니라 '조심하라는 신호': 국가 법령·사업 운영으로 분류된 질문에 조례로 답이 나오면
# 답은 참고용으로 보여 주되 '담당자 확인 필요'로 표시하고 담당자에게 넘깁니다. (응답소 500건: 이 경우 답변 39건에서
# 틀린 답 5건 전부·빗나간 답 23건 중 20건이 나왔고, '조례 질의'로 분류된 답변 21건은 틀린 답 0건)
REVIEW_KINDS = {"국가 법령 질의", "사업·운영 문의"}
# 현장 민원·건의로 분류됐어도 문장이 질문이면 조례 검색을 거칩니다 (채점 문항 F10 "공공자전거가 고장 나서 못 탔으면
# 보상받을 수 있나요?"를 현장 민원으로 보고 바로 넘겨 놓친 일). 답이 나오면 분류 신호에 따라 '담당자 확인 필요'.
QUESTION_FORM = re.compile(r"(\?|？|나요|까요|가요|습니까|는지|인지|은지|던가요|문의|여부)\s*[?？.!~]*\s*$")
# 자치구 법규 사항: 시 조례로 대신 답하지 않고 '자치구 확인 필요' (채점 함정 Z1 '마포구 조례로 과태료', 응답소 632 '구 체육관 할인율')
SEOUL_GU = ("종로구|중구|용산구|성동구|광진구|동대문구|중랑구|성북구|강북구|도봉구|노원구|은평구|서대문구|마포구|양천구|강서구|"
            "구로구|금천구|영등포구|동작구|관악구|서초구|강남구|송파구|강동구")
GU_COMPARE = re.compile(r"구립|구\s*(체육관|체육센터|주차장|공원|시설)|구별로|구마다|자치구\s*(별|마다)|각\s*(자치)?구")
GU_FACILITY = re.compile(r"(?<![가-힣])(" + SEOUL_GU + r")\s*(?:에\s*있는\s*|의\s*)?[가-힣\s]{0,8}?"
                         r"(공원|주차장|체육관|체육센터|체육시설|도서관|수영장|문화센터|복지관|시설)")
GU_MONEY = re.compile(r"요금|입장료|이용료|사용료|할인|감면|면제|과태료|수수료")


def gu_scope(q):
    return bool(GU_RULE_Q.search(q) or GU_COMPARE.search(q) or (GU_FACILITY.search(q) and GU_MONEY.search(q)))

# ------------------------------------------------------------ 3) 유형 분류와 법 표현 표시
HEDGE = re.compile(r"정해져\s*있지\s*않|명시되어\s*있지\s*않|규정(이|은)\s*(따로\s*)?없|따로\s*정(하|합)|규칙으로\s*정|"
                   r"달라질\s*수|해석|확인이\s*필요")
TYPES = {
    "simple": ("간단 질문", "조례 한 조항에 답이 분명해서 바로 답했어요."),
    "complex": ("복잡 질문", "여러 조항을 함께 봐야 하는 질문이에요. 필요하면 근거 묶음을 담당자에게 보내 검토받을 수 있어요."),
    "gray": ("회색지대", "조문은 있지만 세부가 정해져 있지 않아 해석이 갈릴 수 있어요. 담당자 확인을 권해요."),
    "related": ("관련 조항 안내", "질문에 바로 답하는 문장은 찾지 못했지만 관련 조항을 찾았어요. 조문을 보시고, 정확한 적용은 담당자에게 확인하세요."),
    "no_basis": ("근거 없음", "선택한 지역의 자치법규에서 관련 규정을 찾지 못했어요. 답을 지어내지 않고 담당 부서로 연결해 드려요."),
    "handoff": ("일반 민원", "법령 해석이 필요한 질문이 아니라 담당 부서의 처리가 필요한 민원이에요."),
    "national": ("국가 법령 확인", "조례가 아니라 법률·시행령이 정하는 사항으로 보여요."),
    "business": ("사업·운영 문의", "조례가 아니라 사업·시설 운영 정보를 묻는 질문으로 보여요."),
    "field": ("현장 민원", "현장 확인과 조치가 필요한 민원이에요."),
    "proposal": ("건의·제안", "제도 개선이나 시설 설치에 관한 건의예요."),
    "gu": ("자치구 확인 필요", "자치구 조례가 정하는 사항일 수 있어 서울시 조례로 대신 답하지 않았어요. 해당 구청 확인이 필요해요."),
    "review": ("담당자 확인 필요", "관련 조례를 찾았지만 법률이나 사업 운영이 걸린 질문일 수 있어 담당자 확인으로 넘겼어요. 아래 답은 참고용이에요."),
}
# 제안서 4-2 '꼭 구분해야 할 법령 표현'
EXPRESSIONS = [
    ("노력 의무", re.compile(r"노력하여야\s*한다|노력해야\s*한다"), "의무처럼 보이지만 강제력이 약해요. 의무 위반으로 단정하지 않아요."),
    ("의무", re.compile(r"(?<!노력)하여야\s*한다|(?<!노력)해야\s*한다"), "반드시 해야 하는 일이에요."),
    ("재량", re.compile(r"할\s*수\s*있다"), "기관이 할지 말지 고를 수 있어요. 받는다고 보장되는 것은 아니에요."),
    ("금지", re.compile(r"(하여서는|해서는)\s*아니\s*된다|금지한다"), "하면 안 되는 일이에요."),
    ("예외(단서)", re.compile(r"다만[,\s]"), "'다만' 뒤의 예외까지 함께 읽어야 해요."),
    ("준용", re.compile(r"준용한다"), "다른 조항을 가져와 적용해요. 그 조항도 함께 봐야 해요."),
    ("위임", re.compile(r"(규칙|조례|시장이|구청장이)\s*(으로|로)?\s*(따로\s*)?정한다|따로\s*정한다"), "세부 내용은 다른 규정에 맡겨요."),
]


def expressions(text):
    body = re.sub(r"^\[[^\]]*\]\s*", "", text)
    out = []
    for name, pat, tip in EXPRESSIONS:
        if pat.search(body):
            if name == "의무" and any(o["name"] == "노력 의무" for o in out) and not re.search(r"(?<!노력)(하여|해)야\s*한다", body.replace("노력하여야", "")):
                continue
            out.append({"name": name, "tip": tip})
    return out


META_REF = re.compile(r"(?:제시된|주어진|위의?)\s*근거(?=만으로|에는|에서|에\s)")
META_SUFFIX = re.compile(r"(?<![가-힣])근거(만으로는|만으로|에는|에서는|에서)(?=\s)")


def citizen_text(ans):
    """답에 남은 처리 과정 표현("이 근거만으로는", "근거에는")을 시민 말("조례만으로는", "조례에는")로 바꿉니다.
    답변 단계 지시문이 '근거'라는 말로 쓰여 있어서 답에 그대로 새어 나오는 일이 있었음."""
    if not ans or ans == "확인 불가":
        return ans
    ans = META_REF.sub("조례", ans)
    ans = META_SUFFIX.sub(lambda m: "조례" + m.group(1), ans)
    return plain_structure(ans)


# 조례 문서 구조를 가리키는 말 → 시민 말. 답변 단계에서 한 번 다시 쓰게 한 뒤에도 남은 것만 여기서 바꿉니다.
# (뜻이 바뀌지 않는 표현만: 문서 위치를 가리키는 수식어를 빼고, '대상 목록'은 '대상'으로)
STRUCT_FIX = [
    (re.compile(r"근거(?:로|에)\s*(?:제시된|적힌|나온|주어진)\s*"), ""),
    (re.compile(r"(?:제공된|주어진|제시된)\s*(조례|자치법규|근거|자료|규정)"), "조례"),
    (re.compile(r"(?<![가-힣])근거(는|가)(?=\s)"), r"조례\1"),
    (re.compile(r"\s*\((?:[^()]*?(?:별표|(?<!조)제\s*\d+\s*호|제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?\s*관련)[^()]*?)\)"), ""),  # (별표 2 제1호)
    (re.compile(r"별표(?:\s*\d+)?의\s*범위"), "조례가 정한 범위"),
    (re.compile(r"(?:(?:같은|이|해당)\s*)?(?:조례(?:의)?\s*)?별표(?:\s*\d+)?\s*목록에\s*(?:직접\s*|따로\s*)?"), "따로 "),
    (re.compile(r"(?:(?:같은|이|해당)\s*)?(?:조례(?:의)?\s*)?별표(?:\s*\d+)?(?:\s*제\s*\d+\s*호)?(?:에서\s*정한|에\s*따라|의|에는|에)\s*"), ""),
    (re.compile(r"(?:그|위의?|이|해당)\s*대상\s*목록에\s*해당하는"), "감면·면제 대상에 해당하는"),
    (re.compile(r"(?:그|위의?|이|해당)\s*대상\s*중\s*(?:하나에\s*해당하는|어느\s*하나에\s*해당하는)"), "대상에 해당하는"),
    (re.compile(r"(?:그|위의?|이|해당)\s*대상\s*중\s*"), "대상 가운데 "),
    (re.compile(r"대상\s*목록"), "대상"),
    (re.compile(r"(?:이\s*)?목록에\s*(따로|직접)\s*"), r"\1 "),
    (re.compile(r"(?:이\s*)?목록에(?:는)?\s*"), "대상에 "),
]


def plain_structure(ans):
    for pat, rep in STRUCT_FIX:
        ans = pat.sub(rep, ans)
    return re.sub(r"\s{2,}", " ", ans).strip()


def classify(question, kind, res, asked=False):
    if kind in KIND_TYPE:
        # 질문형 현장 민원·건의: 조례로 답이 나왔으면 담당자 확인 필요, 아니면 원래대로 담당 부서 전달
        return "review" if asked and res["answer"] != "확인 불가" else KIND_TYPE[kind]
    if kind == "일반 민원":
        return "handoff"
    if res["answer"] == "확인 불가":
        return UNKNOWN_TYPE.get(kind, "no_basis")
    if res.get("delegated"):
        return "gray"          # 조문은 있지만 세부는 시장·규칙이 정함 → 회색지대(담당자 확인 권유)
    if kind in REVIEW_KINDS:
        return "review"
    if res.get("related"):
        return "related"
    if HEDGE.search(res["answer"]):
        return "gray"
    arts = {re.sub(r"\s*제\d+항$", "", c) for c in res["citation"]}
    if len(arts) >= 2 or rag.is_multi_q(question):
        return "complex"
    return "simple"


def _evidence(citations):
    out = []
    for lb in citations:
        law = corpus.law_of_label(lb) or {}
        text = corpus.label_text(lb)
        out.append({"label": lb, "law": law.get("name", ""), "text": text,
                    "source_url": law.get("source_url", ""), "effective_date": law.get("effective_date", ""),
                    "dept": law.get("dept", ""), "phone": law.get("phone", ""), "expressions": expressions(text)})
    return out


def _dept(evidence, laws, region):
    for src in ([e for e in evidence if e.get("dept")] + [lw for lw in laws if lw.get("dept")]):
        return {"dept": src["dept"], "phone": src.get("phone", ""), "org": region,
                "basis": src.get("law") or src.get("name", ""), "guessed": not evidence}
    return {"dept": "민원 총괄 부서", "phone": "120" if region.startswith("서울") else "", "org": region,
            "basis": "", "guessed": True}


# "언제부터 적용되나요 / 기준일 / 소급되나요 / ○○ 이후 혼인한 부부"처럼 시점을 묻는 질문
TEMPORAL = re.compile(r"언제부터|기준\s*일|적용\s*(시점|시기|기준|대상\s*시기)|소급|시행\s*일|시행\s*전|"
                      r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일\s*(이후|이전|부터)|(이후|이전)에\s*(혼인|출산|신청|입주|가입)")
AMEND_NOTE = re.compile(r"<((?:개정|신설|전문개정)[^>]*)>|\[((?:본조신설|전문개정|제목개정)[^\]]*)\]")


def temporal_hits(q, hits, max_laws=2, max_addenda=2):
    """시점 질문이면, 상위 근거 조례의 최근 부칙(시행일·적용례)과 상위 조항의 신설·개정 날짜 메모를 근거에 붙입니다.
    (답변 단계는 개정 꼬리표를 지우고 넣으므로 '본조신설 2025.7.14' 같은 날짜를 따로 적어 줍니다.)"""
    if not TEMPORAL.search(q):
        return []
    store = rag._load()[0]
    have = {h["id"] for h in hits}
    top_laws = list(dict.fromkeys(h["law"] for h in hits[:6]))[:max_laws]
    out = []
    for law in top_laws:
        add = [c for c in store if c.get("law") == law and " 부칙(" in c["label"] and c["id"] not in have]
        out += [{**c, "score": 0.0, "cos": None, "related": []} for c in add[-max_addenda:]]
        for art in dict.fromkeys(h["article"] for h in hits[:6] if h["law"] == law):
            tags = []
            for c in store:
                if c.get("article") == art and not c.get("sub"):
                    tags += [a or b for a, b in AMEND_NOTE.findall(c["text"])]
            tags = list(dict.fromkeys(t.strip() for t in tags))
            if tags:
                base = next(c for c in store if c.get("article") == art)
                out.append({**base, "id": f"note:{art}", "parent": f"note:{art}", "label": art, "score": 0.0, "cos": None,
                            "related": [], "text": f"[{art} 개정 이력]\n이 조는 다음과 같이 신설·개정되었다: {', '.join(tags)}"})
    return out


# 조문 안의 같은 자치법규 참조: "제48조제4호 및 제5호의 지역", "제12조제2항제3호" (법·영·「다른 법」 참조는 제외)
INNER_REF = re.compile(r"(?<![」법영\w]\s)(?<![」법영\w])(제\d+조(?:의\d+)?)\s*(제\d+항)?\s*(제\d+호(?:\s*(?:및|ㆍ|·|,|또는|부터)\s*제\d+호(?:까지)?)*)")
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"


def _item_line(text, num):
    m = re.search(rf"(?m)^\s*{num}\.\s*(.+)$", text)
    return re.sub(r"\s+", " ", m.group(1)).strip()[:90] if m else ""


def _ref_nums(s):
    """'제3호부터 제10호까지 및 제12호' → [3..10, 12]"""
    nums = []
    for m in re.finditer(r"제(\d+)호(?:\s*부터\s*제(\d+)호(?:\s*까지)?)?", s):
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        nums += list(range(a, min(b, a + 20) + 1))
    return nums


def expand_refs(hits, q="", limit=6):
    """근거 조각에 같은 자치법규의 다른 조 호 참조가 있으면 그 호의 내용을 풀어 덧붙입니다.
    예) 도시계획 조례 제51조 제2항 제9호 "제48조제4호 및 제5호의 지역" → (참조) 제48조제4호: 제2종일반주거지역 : 200퍼센트 이하
    질문은 "제2종일반주거지역"으로 묻는데 조문은 번호로만 가리켜, 맞는 조항을 찾고도 답을 못 하던 일(응답소 1385)."""
    store = rag._load()[0]
    by_article = {}
    for c in store:
        if not c.get("sub"):
            by_article.setdefault(c.get("article"), []).append(c)
    out = []
    for h in hits:
        notes = []
        for m in INNER_REF.finditer(h["text"]):
            art_key = f"{h['law']} {m.group(1)}"
            chunks = by_article.get(art_key, [])
            if m.group(2):
                chunks = [c for c in chunks if c["label"].endswith(m.group(2).replace("제", "제"))] or chunks
            body = "\n".join(c["text"] for c in chunks)
            for num in _ref_nums(m.group(3)):
                line = _item_line(body, num)
                note = f"{m.group(1)}{m.group(2) or ''}제{num}호: {line}"
                if line and note not in notes:
                    notes.append(note)
        if notes:
            qw = {w for w in re.findall(r"[가-힣]{2,}", q)}
            notes.sort(key=lambda n: -sum(1 for w in qw if w in n))     # 질문 낱말이 든 참조부터
            h = {**h, "text": h["text"] + "\n(조문 속 참조 풀이) " + " / ".join(notes[:limit])}
        out.append(h)
    return out


RELATED_SYSTEM = """시민 질문과 자치법규 조항 후보를 보고, 질문이 묻는 대상·행위·제도를 직접 정한 조항만 고른다. JSON으로만 답한다.
- 고르는 조항: 질문의 대상(시설·사람·요금·행위·제도)을 그 조항이 직접 다룸. 답이 그 조항에 다 적혀 있지 않아도 된다.
- 고르지 않는 조항: 같은 분야일 뿐 질문 대상과 다른 것(다른 시설·다른 요금·다른 위반행위), 목적·정의·위임처럼 일반적인 조항.
- related: 고른 후보 번호 (가장 직접적인 것부터, 최대 2개, 없으면 빈 배열)"""
RELATED_SCHEMA = {"type": "json_schema", "json_schema": {"name": "related", "strict": True, "schema": {
    "type": "object", "properties": {"related": {"type": "array", "items": {"type": "integer"}}},
    "required": ["related"], "additionalProperties": False}}}


def related_articles(q, hits, n=5):
    """확인 불가로 끝났을 때 보여 줄 '관련 조항': 검색 상위 n개 중 질문 대상을 직접 정한 조항 (Solar 판단)."""
    cands, seen = [], set()
    for h in hits:
        if h["label"] in seen or h.get("definition_of") or str(h.get("id", "")).startswith("note:"):
            continue
        seen.add(h["label"])
        cands.append(h)
        if len(cands) >= n:
            break
    if not cands:
        return []
    listing = "\n\n".join(f"[{i}] {c['label']}\n{c['text'][:700]}" for i, c in enumerate(cands, 1))
    try:
        import upstage
        out = json.loads(upstage.chat([{"role": "system", "content": RELATED_SYSTEM},
                                       {"role": "user", "content": f"[질문]\n{q}\n\n[조항 후보]\n{listing}"}],
                                      response_format=RELATED_SCHEMA, timeout=40))
    except Exception:  # noqa: BLE001
        return []
    return [cands[i - 1] for i in out.get("related", [])[:2] if isinstance(i, int) and 1 <= i <= len(cands)]


ADDENDA_RULE = re.compile(r"유효기간|적용례|경과조치|적용한다|까지\s*적용|부터\s*적용|소급")


def addenda_for(hits, top=6, max_items=2):
    """상위 근거 조항을 직접 언급하면서 유효기간·적용례·경과조치를 정한 부칙을 찾아 붙입니다 (질문 종류와 무관).
    예) 도시계획 조례 부칙 "제51조제2항제9호 규정은 2028년 5월 18일까지 … 적용한다" → 용적률 '한시' 완화의 근거."""
    have = {h["label"] for h in hits}
    out = []
    for h in hits[:top]:
        m = re.search(r"(제\d+조(?:의\d+)?)(?:\s*제(\d+)항)?$", h["label"])
        if not m:
            continue
        art, para = m.group(1), m.group(2)
        pat = re.compile(re.escape(art) + (rf"\s*(?:제{para}항)?" if para else "") + r"(?!의|\d)")
        for a in reversed(corpus.ADDENDA.get(h["law"], [])):        # 최근 부칙부터
            if a["label"] in have or not pat.search(a["text"]) or not ADDENDA_RULE.search(a["text"]):
                continue
            # 다른 조례를 고치는 부칙(타 조례의 개정)에서 조 번호가 우연히 겹친 경우는 뺌
            if re.search(r"(다른|타)\s*조례의\s*개정", a["text"]) and not re.search(r"유효기간|적용례|경과조치", a["text"]):
                continue
            have.add(a["label"])
            out.append({"law": h["law"], "label": a["label"], "article": a["label"], "text": a["text"][:1500],
                        "id": f"addenda:{a['label']}", "parent": f"addenda:{a['label']}", "score": 0.0, "cos": None,
                        "related": [], "effective_date": h.get("effective_date", ""), "source_url": h.get("source_url", "")})
            if len(out) >= max_items:
                return out
    return out


def complete_top_articles(hits, n_art=2, max_add=6):
    """상위 n_art개 조는 K 제한과 상관없이 모든 항을 근거에 넣습니다 (빠진 항만 그 조 바로 뒤에 덧붙임).
    조는 검색 2위인데 정답 항은 9위라 K=8에서 잘리던 일(응답소 1406 공공재개발 제46조 제1항)."""
    store = rag._load()[0]
    have = {h["label"] for h in hits}
    arts = list(dict.fromkeys(h["article"] for h in hits if h.get("article") and not h.get("definition_of")))[:n_art]
    out, added = list(hits), 0
    for art in arts:
        miss = [c for c in store if c.get("article") == art and not c.get("sub") and c["label"] not in have]
        for c in miss:
            if added >= max_add:
                return out
            pos = max(i for i, h in enumerate(out) if h.get("article") == art) + 1
            out.insert(pos, {**c, "score": 0.0, "cos": None, "related": []})
            have.add(c["label"])
            added += 1
    return out


def focus_hits(q, rel, hits, max_chunks=10):
    """관련 조항의 같은 조 전체(모든 항) + 원래 근거 중 같은 조례의 보조 근거(정의·개정 이력·부칙·참조 풀이)."""
    store = rag._load()[0]
    arts = list(dict.fromkeys(h["article"] for h in rel))
    out, seen = [], set()
    by_label = {h["label"]: h for h in hits}         # 참조 풀이가 붙은 조각은 그것을 씀
    for art in arts:
        for c in store:
            if c.get("article") == art and not c.get("sub") and c["label"] not in seen:
                seen.add(c["label"])
                out.append(by_label.get(c["label"]) or {**c, "score": 0.0, "cos": None, "related": []})
    laws = {h["law"] for h in rel}
    for h in hits:
        aux = h.get("definition_of") or str(h.get("id", "")).startswith("note:") or " 부칙(" in h["label"]
        if aux and h["label"] not in seen and (h["law"] in laws or h.get("definition_of")):
            seen.add(h["label"])
            out.append(h)
    return expand_refs(out[:max_chunks], q)


UNSPEC_ART = re.compile(r"(일부|해당|관련|특정|어떤|몇몇|이)\s*(조항|조문|규정)(?:의|에|에서|은|는|이|가)?\s*[^?]{0,20}?(성격|취지|해석|의미|적용\s*여부|위반\s*여부|효력)")

# 확인 불가일 때 '관련 조항 안내'를 시민에게 보여 줄지. 끔(사용자 결정 2026-11-21): 조례는 절대 틀리면 안 되고,
# 확실하지 않으면 확인 불가로 담당자에게 넘김. 확실하지 않은 조항을 보여 주면 그 조항이 틀렸을 때 곧 '틀린 조례'가 됨.
SHOW_RELATED = False

GU_RULE_Q = re.compile(r"(?:^|\s)[가-힣]{1,3}구\s+[가-힣\s]{0,20}?(조례|규칙)")
ABSENT_HEAD = re.compile(r"정해져\s*있지\s*않|규정(은|이|도)?\s*((근거|조례)(에는|에)\s*)?(따로\s*|별도로\s*)?없|직접\s*(정한\s*(내용|규정)이?\s*없|정하고\s*있지\s*않|적혀\s*있지\s*않)"
                         r"|명시(되어|돼)\s*있지\s*않|적혀\s*있지\s*않|별도의?\s*(기준|규정)(은|이)?\s*없"
                         r"|(내용|규정|기준|조항)(은|이|도)?\s*(따로\s*|별도로\s*)?없(습니다|다|어)"
                         r"|(명시적으로|직접적으로|따로|별도로)?\s*정하고\s*있지\s*않|정해지지\s*않"
                         r"|(정하지|규정하지|다루지|명시하지|언급하지)\s*않")
YES_NO_HEAD = re.compile(r"^\s*(네|예|아니요|아니오)\s*[,.]\s*")


def is_absent_answer(ans):
    """첫 문장이 '정해져 있지 않다·규정이 없다'로 결론 나는 답인지."""
    first = re.split(r"(?<=[.!?])\s+", ans.strip(), maxsplit=1)[0]
    return bool(ABSENT_HEAD.search(first))


def related_text(rel):
    """관련 조항 안내 문장: 어느 조례의 어느 조항인지 바로 말합니다 (예: 「서울특별시 생활임금 조례」 제7조 제3항)."""
    names = []
    for h in rel:
        m = re.match(r"(.+?)\s+((?:제\d+조(?:의\d+)?(?:\s*제\d+항)?)|별표.*|부칙.*)$", h["label"])
        names.append(f"「{m.group(1)}」 {m.group(2)}" if m else h["label"])
    return f"질문에 바로 답하는 문장은 조례에서 찾지 못했어요. 가장 관련 있는 조항은 {', '.join(names)}이에요. 아래 원문을 확인해 주세요."


def answer_with_definitions(q, region, laws, trace):
    """고른 자치법규 안에서 검색한 뒤, 근거에 나온 용어의 정의 조항(다른 자치법규)을 붙여 답합니다.
    laws 목록에 정의 조항을 가져온 자치법규를 덧붙입니다(화면의 '찾아본 자치법규'에 보이도록).
    확인 불가로 끝나면 검색 상위 조항 중 질문 대상을 직접 정한 조항을 '관련 조항'으로 돌려줍니다(res["related"])."""
    names = {lw["name"] for lw in laws}
    hits = rag._anchor_law_only(q, rag._drop_cross_domain(q, rag.retrieve(q, laws=names)))
    hits = expand_refs(complete_top_articles(hits), q)
    try:
        extra, new = corpus.definition_hits(q, hits[:config.TOP_K], region, names)
    except Exception:  # noqa: BLE001  정의 찾기는 보조 수단
        extra, new = [], []
    if extra:
        hits = hits[:6] + extra + hits[6:]
        names |= set(new)
        laws += [corpus.LAWS[n] for n in new if n in corpus.LAWS]
        trace.append("definition:" + ",".join(e["definition_of"] for e in extra))
    tmp = temporal_hits(q, hits)
    if tmp:
        hits = hits[:6] + tmp + hits[6:]
        trace.append(f"temporal:{len(tmp)}")
    add = addenda_for(hits)
    if add:
        hits = hits[:6] + add + hits[6:]
        trace.append(f"addenda:{len(add)}")
    res = rag.answer(q, trace=trace, convert=False, laws=names, hits=hits)
    gu_q = bool(GU_RULE_Q.search(q))              # "강남구 … 규칙에 따르면": 자치구 자치법규를 물음 → 시 조례로 대신하지 않음
    if res["answer"] != "확인 불가" and is_absent_answer(res["answer"]) and not gu_q:
        # "아니요, 조례에 정해져 있지 않습니다"처럼 '없음'으로 결론 낸 답: 관련 조항으로 좁혀 한 번 더 보고, 더 구체적인 답이면 바꿈
        rel = related_articles(q, hits)
        if rel:
            res2 = rag.answer(q, trace=trace, convert=False, laws=names, hits=focus_hits(q, rel, hits))
            if res2["answer"] != "확인 불가" and not is_absent_answer(res2["answer"]):
                trace.append("focused_absent")
                return res2
        # '조례에 정해져 있지 않다'는 것도 조례에 대한 단정이라 틀리면 틀린 조례 답과 같습니다 (응답소 1406·1365·1385에서 실제로 틀림).
        # 좁혀 다시 봐도 구체적인 답이 없으면 확인 불가로 담당자에게 넘깁니다.
        d = rag._delegation_answer(q, hits)      # '없음' 결론이어도 세부를 시장·규칙에 맡긴 조항이 있으면 그 조항으로 답함
        if d:
            trace.append("delegated")
            return d
        trace.append("absent_to_unknown")
        return {**dict(rag.UNKNOWN), "review_candidates": [h["label"] for h in rel] if rel else []}
    if res["answer"] == "확인 불가" and not gu_q and "stop_no_basis" not in trace:
        rel = related_articles(q, hits)
        if rel:
            # 좁혀서 다시 답하기: 질문 대상을 직접 정한 조항(같은 조의 다른 항 포함)과 그 조례의 보조 근거만 주고 한 번 더 답합니다.
            # 근거가 8개여도 다른 조항에 시선이 흩어져 '못 찾음'으로 끝나는 일이 많았음 (정답 조항이 검색 1위인데 확인 불가 35/187)
            focused = focus_hits(q, rel, hits)
            res2 = rag.answer(q, trace=trace, convert=False, laws=names, hits=focused)
            if res2["answer"] != "확인 불가":
                if not is_absent_answer(res2["answer"]):     # '없음'으로 결론 낸 답은 쓰지 않음 (아래에서 확인 불가)
                    trace.append("focused")
                    return res2
            trace.append(f"related:{len(rel)}")
            if SHOW_RELATED:
                res = {**res, "answer": related_text(rel), "citation": [h["label"] for h in rel], "related": True,
                       "effective_date": rel[0].get("effective_date", ""), "source_url": rel[0].get("source_url", "")}
            else:
                # 시민에게는 확인 불가(담당자 연결). 찾아본 후보 조항은 담당자 검토용으로만 남깁니다.
                res = {**res, "review_candidates": [h["label"] for h in rel]}
    if res["answer"] == "확인 불가" and not any(t.startswith(rag.DELEGATE_EXCLUDE) for t in trace):
        # 조례가 세부(시간·금액·기준)를 시장·규칙에 맡긴 경우: 확인 불가 대신 그 위임 조항을 근거로 답함 (운영 예시 반영)
        d = rag._delegation_answer(q, hits)
        if d:
            trace.append("delegated")
            return d
    if res["answer"] == "확인 불가" and not res.get("review_candidates"):
        # 일찍 멈춘 확인 불가(stop_no_basis 등)도 담당자에게 어느 조례·조항을 살펴봤는지 넘깁니다 (검색 상위 3개, 추가 호출 없음)
        res = {**res, "review_candidates": list(dict.fromkeys(h["label"] for h in hits[:3]))}
    return res


def _law_brief(lw):
    return {"name": lw["name"], "kind": lw.get("kind", "조례"), "dept": lw.get("dept", ""),
            "effective_date": lw.get("effective_date", ""), "source_url": lw.get("source_url", "")}


# ------------------------------------------------------------ 질문 하나 처리 (단계별로 내보냄)
def ask_stream(question, region="서울특별시", pinned=None, as_of=None):
    """단계별 이벤트(dict)를 차례로 내보냅니다: triage → laws → answer.
    as_of(YYYY-MM-DD): 그날 시행 중이던 자치법규 버전으로 답합니다 (응답소 공개민원 검증용)."""
    t0 = time.time()
    corpus.init()
    q, masked = mask(question.strip())
    tri = triage(q)
    yield {"stage": "triage", **tri, "masked": masked}
    q = tri["normalized"] or q
    try:
        laws, others = corpus.pick_laws(q, region, tri["keywords"], tri["title_words"], pinned=pinned)
    except Exception as e:  # noqa: BLE001  법제처 API 장애
        laws, others = [], []
        yield {"stage": "warn", "message": f"자치법규 검색 중 오류가 났어요 ({type(e).__name__})."}
    not_yet = []
    if as_of and laws:
        laws, not_yet = corpus.as_of(laws, region, as_of)
    yield {"stage": "laws", "laws": [_law_brief(lw) for lw in laws],
           "others": [{"name": c["name"], "kind": c.get("kind", "")} for c in others]}
    trace = []
    asked = tri["kind"] in KIND_TYPE and bool(QUESTION_FORM.search(q))
    if gu_scope(q):
        res = dict(rag.UNKNOWN)
        trace.append("gu_scope")
    elif (tri["kind"] not in ANSWER_KINDS and not asked) or not laws:
        res = dict(rag.UNKNOWN)
        trace.append(KIND_TYPE.get(tri["kind"], "handoff") if tri["kind"] not in ANSWER_KINDS else "no_laws")
    elif UNSPEC_ART.search(q) and not re.search(r"제\s*\d+\s*조", q):
        # "일부 조항의 성격", "해당 규정의 취지"처럼 조항을 가리키는데 조 번호가 없으면 어느 조항인지 확정할 수 없음 → 확인 불가(담당자)
        # (응답소 1322: 다른 조항이나 목적·정의 조항으로 일반론을 답하던 일)
        res = dict(rag.UNKNOWN)
        trace.append("unspecified_article")
    else:
        try:
            res = answer_with_definitions(q, region, laws, trace)
        except Exception as e:  # noqa: BLE001
            res = dict(rag.UNKNOWN)
            trace.append(f"error:{type(e).__name__}")
    res["answer"] = citizen_text(res["answer"])
    typ = "gu" if "gu_scope" in trace else classify(q, tri["kind"], res, asked)
    ev = _evidence(res["citation"])
    label, desc = TYPES[typ]
    yield {"stage": "answer", "question": q, "region": region, "type": typ, "type_label": label, "type_desc": desc,
           "answer": res["answer"], "citation": res["citation"], "effective_date": res.get("effective_date", ""),
           "source_url": res.get("source_url", ""), "evidence": ev, "dept": _dept(ev, laws, region),
           "laws": [_law_brief(lw) for lw in laws], "title": tri["title"], "issue": tri["issue"],
           "kind": tri["kind"], "sec": round(time.time() - t0, 1), "trace": "/".join(trace),
           "as_of": as_of, "not_yet": not_yet, "review_candidates": res.get("review_candidates", [])}


def ask(question, region="서울특별시", pinned=None, as_of=None):
    final = None
    for ev in ask_stream(question, region, pinned, as_of):
        if ev["stage"] == "answer":
            final = ev
    return final


def compare(question, regions):
    """같은 질문을 여러 지역에서 (제안서 5장). 질문 분석은 한 번만 합니다."""
    corpus.init()
    q, masked = mask(question.strip())
    tri = triage(q)
    q = tri["normalized"] or q
    from concurrent.futures import ThreadPoolExecutor

    def one(region):
        t0 = time.time()
        try:
            laws, _ = corpus.pick_laws(q, region, tri["keywords"], tri["title_words"])
        except Exception:  # noqa: BLE001
            laws = []
        res, trace = dict(rag.UNKNOWN), []
        if laws:
            try:
                res = answer_with_definitions(q, region, laws, trace)
            except Exception as e:  # noqa: BLE001
                trace.append(f"error:{type(e).__name__}")
        ev = _evidence(res["citation"])
        return {"region": region, "answer": citizen_text(res["answer"]), "found": res["answer"] != "확인 불가",
                "evidence": ev, "laws": [_law_brief(lw) for lw in laws], "dept": _dept(ev, laws, region),
                "sec": round(time.time() - t0, 1), "trace": "/".join(trace)}

    with ThreadPoolExecutor(min(4, len(regions))) as ex:
        rows = list(ex.map(one, regions))
    found = [r["region"] for r in rows if r["found"]]
    if len(found) == len(rows):
        summary = "모든 지역에 관련 조례가 있어요. 조항 내용을 나란히 비교해 보세요."
    elif not found:
        summary = "선택한 지역 모두에서 관련 조례를 찾지 못했어요. 상위 법령(법률·시행령) 확인이 필요해요."
    else:
        missing = [r["region"] for r in rows if not r["found"]]
        summary = (f"{', '.join(missing)}에는 관련 조례를 찾지 못했어요. "
                   f"{', '.join(found)} 사례를 참고해 조례 제정·개정을 건의하는 민원을 넣을 수 있어요.")
    return {"question": q, "normalized": tri["normalized"], "masked": masked, "keywords": tri["keywords"],
            "rows": rows, "summary": summary}


# ------------------------------------------------------------ 담당자 문의 (민원 접수)
TICKETS = config.DATA / "tickets.json"
_tlock = threading.Lock()
STEPS = ["민원등록", "기관지정", "담당자처리중", "답변완료"]

ISSUE_SYSTEM = """너는 지방자치단체 민원 담당 공무원을 돕는 도우미다. 시민 질문과 AI가 찾은 근거 조항을 보고,
담당자가 답변 전에 확인해야 할 쟁점을 2~3개 뽑는다. 근거에 없는 사실을 지어내지 않는다.
각 쟁점은 한 문장, 담당자가 확인할 행동이 드러나게 쓴다. JSON으로 낸다."""
ISSUE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "issues", "strict": True, "schema": {
    "type": "object", "properties": {"issues": {"type": "array", "items": {"type": "string"}}},
    "required": ["issues"], "additionalProperties": False}}}


def _load_tickets():
    try:
        return json.loads(TICKETS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def _save_tickets(rows):
    TICKETS.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def _issues(question, result):
    if not result or not result.get("evidence"):
        return []
    import upstage
    ev = "\n\n".join(f"[{e['label']}]\n{e['text'][:1200]}" for e in result["evidence"])
    try:
        out = json.loads(upstage.chat([{"role": "system", "content": ISSUE_SYSTEM},
                                       {"role": "user", "content": f"[질문]\n{question}\n\n[AI 답변 초안]\n{result.get('answer','')}\n\n[근거]\n{ev}"}],
                                      response_format=ISSUE_SCHEMA))
        return [s for s in out.get("issues", []) if s.strip()][:3]
    except Exception:  # noqa: BLE001
        return []


def create_ticket(payload):
    question, _ = mask(payload.get("question", "").strip())
    message, _ = mask(payload.get("message", "").strip())
    result = payload.get("result") or {}
    region = payload.get("region") or result.get("region") or "서울특별시"
    dept = result.get("dept") or _dept([], [], region)
    now = datetime.now()
    with _tlock:
        rows = _load_tickets()
        no = f"{now:%Y%m%d}-{len(rows) + 1:04d}"
    ticket = {
        "no": no, "created": now.isoformat(timespec="seconds"), "region": region,
        "title": payload.get("title") or result.get("title") or question[:30],
        "question": question, "message": message, "name": (payload.get("name") or "").strip()[:20],
        "reason": payload.get("reason", ""), "dept": dept, "status": "기관지정",
        "history": [{"step": "민원등록", "at": now.isoformat(timespec="seconds")},
                    {"step": "기관지정", "at": now.isoformat(timespec="seconds"),
                     "note": f"{dept.get('org', '')} {dept.get('dept', '')}".strip()}],
        "bundle": {"type": result.get("type"), "type_label": result.get("type_label"),
                   "ai_answer": result.get("answer"), "citation": result.get("citation", []),
                   "evidence": result.get("evidence", []), "laws": result.get("laws", []),
                   "issues": _issues(question, result)},
        "reply": "",
    }
    with _tlock:
        rows = _load_tickets()
        rows.insert(0, ticket)
        _save_tickets(rows)
    return ticket


def list_tickets(nos=None):
    rows = _load_tickets()
    return [t for t in rows if not nos or t["no"] in nos]


def update_ticket(no, payload):
    with _tlock:
        rows = _load_tickets()
        for t in rows:
            if t["no"] != no:
                continue
            now = datetime.now().isoformat(timespec="seconds")
            if payload.get("status") in STEPS and payload["status"] != t["status"]:
                t["status"] = payload["status"]
                t["history"].append({"step": payload["status"], "at": now})
            if "reply" in payload:
                t["reply"] = payload["reply"].strip()
                if t["reply"] and t["status"] != "답변완료":
                    t["status"] = "답변완료"
                    t["history"].append({"step": "답변완료", "at": now})
            _save_tickets(rows)
            return t
    return None
