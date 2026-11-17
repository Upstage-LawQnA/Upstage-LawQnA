"""시민용 화면에서 인용 조항을 눌렀을 때 보여 줄 원문을 찾습니다.

인용 라벨(예: "서울특별시 주차장 설치 및 관리 조례 제7조 제3항")을 받아
  - 조·항이면: 그 조 전체를 항 단위로 돌려주고, 인용된 항에 highlight 표시
  - 별표면: Document Parse로 읽은 Markdown 표 (없으면 법제처 원문 텍스트)
"""
import json
import re
from functools import lru_cache

import config
from parse_annex import annex_path

LABEL = re.compile(r"^(?P<law>서울특별시.+?(?:조례|조례\S*))\s+"
                   r"(?:(?P<art>제\d+조(?:의\d+)?)(?:\s*제(?P<para>\d+)항)?|(?P<annex>별표(?:\s*\d+)?(?:의\d+)?))\s*$")
AMEND = re.compile(r"\s*<(?:개정|신설|삭제|전문개정)[^>]*>|\s*\[(?:본조신설|전문개정|제목개정|종전|제\d+조에서)[^\]]*\]")


@lru_cache(maxsize=1)
def _laws():
    laws = json.loads((config.DATA / "ordinances.json").read_text(encoding="utf-8"))
    return {law["name"]: law for law in laws}


def _clean(text):
    """개정 이력 꼬리표(<개정 …>, [본조신설 …])를 지워 시민이 읽기 쉽게 합니다."""
    return re.sub(r"[ \t]+\n", "\n", AMEND.sub("", text)).strip()


def lookup(label):
    m = LABEL.match(label.strip())
    if not m:
        return None
    law = _laws().get(m["law"])
    if not law:
        return None
    base = {"label": label, "law": law["name"], "source_url": law["source_url"],
            "effective_date": law["effective_date"]}

    if m["annex"]:
        key = re.sub(r"\s+", " ", m["annex"]).strip()
        for ax in law["annexes"]:
            if ax["label"] == key:
                parsed = annex_path(law["mst"], ax["label"])
                body = parsed.read_text(encoding="utf-8") if parsed.exists() else ax["text"]
                body = re.sub(r"^# .*\n+", "", body, count=1)      # 파일 첫 줄 제목은 중복이라 뺍니다
                return {**base, "kind": "annex", "title": f"{ax['label']} {ax['title']}", "markdown": body}
        return None

    for a in law["articles"]:
        if a["article"] == m["art"]:
            para = int(m["para"]) if m["para"] else None
            paras = []
            for p in a["paragraphs"]:
                text = _clean(p["text"])
                if p["no"] is None:                    # 항 없는 조: 머리의 '제○조(제목)' 반복 제거
                    text = re.sub(r"^제\d+조(?:의\d+)?\([^)]*\)\s*", "", text)
                paras.append({"no": p["no"], "text": text, "highlight": para is not None and p["no"] == para})
            return {**base, "kind": "article", "title": f"{a['article']}({a['title']})", "paragraphs": paras}
    return None
