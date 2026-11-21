"""3단계: 조문을 조각(chunk)으로 나누고 임베딩해 검색 색인을 만듭니다.

조각마다 '인용 라벨'을 붙여 두는 것이 핵심입니다.
  예) 서울특별시 주차장 설치 및 관리 조례 제7조 제3항
LLM은 이 라벨을 고르기만 하고, 조항 표기를 직접 쓰지 않습니다.

실행:  python -X utf8 build_index.py             (CHUNK_UNIT=paragraph 기본)
       set CHUNK_UNIT=article & python -X utf8 build_index.py   (조 단위, 시연 '전' 장면)
결과:  data/chunks_<단위>.json, data/vectors_<단위>.npy
"""
import json
import re

import numpy as np

import config
from parse_annex import annex_path


HO = re.compile(r"^\s*\d+\.\s")      # 호(1. 2. …)로 시작하는 줄


def _ho_subchunks(meta, label, article_key, head, body):
    """긴 조·항 안의 호를 하나씩 보조 조각으로 만듭니다. 라벨은 소속 조·항 그대로 둡니다.

    예) 청년 기본 조례 제3조(정의)는 호가 8개라 '청년 = 19~39세'가 긴 조각 속에 묻힙니다.
        호 하나짜리 보조 조각을 따로 두면 검색에 걸리고, 답변 단계에는 원래 조·항 전체가 들어갑니다.
    """
    lines = body.splitlines()
    ho = [ln.strip() for ln in lines if HO.match(ln)]
    if len(ho) < 2:
        return []
    lead = lines[0].strip()[:120]           # 항·조의 머리 문장 (예: "다음 각 호의 사람에 대하여 …")
    return [{**meta, "label": label, "article": article_key, "sub": True,
             "text": f"[{head}] {lead}\n{h}"} for h in ho]


def make_chunks(laws, unit):
    chunks = []
    for law in laws:
        meta = {"law": law["name"], "effective_date": law["effective_date"], "source_url": law["source_url"]}
        for a in law["articles"]:
            head = f"{law['name']} {a['article']}({a['title']})"
            article_key = f"{law['name']} {a['article']}"
            if unit == "article" or len(a["paragraphs"]) <= 1 or a["paragraphs"][0]["no"] is None:
                chunks.append({**meta, "label": article_key, "article": article_key, "text": a["text"]})
                if unit != "article":
                    chunks += _ho_subchunks(meta, article_key, article_key, head, a["text"])
                continue
            for p in a["paragraphs"]:
                label = f"{article_key} 제{p['no']}항"
                chunks.append({**meta, "label": label, "article": article_key,
                               "text": f"[{head}]\n{p['text']}"})
                chunks += _ho_subchunks(meta, label, article_key, head, p["text"])
        for ax in law["annexes"]:
            if not ax["label"].startswith("별표"):
                continue
            parsed = annex_path(law["mst"], ax["label"])
            if config.USE_PARSED_ANNEX and parsed.exists():
                body, src = parsed.read_text(encoding="utf-8"), "document-parse"
            else:
                body, src = ax["text"], "law.go.kr 원문(괘선 표)"
            # 긴 별표는 잘라서 넣되 라벨은 같게 둡니다
            for i in range(0, max(len(body), 1), 1800):
                chunks.append({**meta, "label": f"{law['name']} {ax['label']}",
                               "article": f"{law['name']} {ax['label']}",
                               "text": f"[{law['name']} {ax['label']} {ax['title']}]\n{body[i:i + 1800]}",
                               "annex_source": src})
    # parent: 보조 조각(호)이 가리키는 원래 조·항 조각의 id. 원래 조각은 자기 자신.
    parent_of_label = {}
    for i, c in enumerate(chunks):
        c["id"] = i
        if not c.get("sub"):
            parent_of_label.setdefault(c["label"], i)
            c["parent"] = i
    for c in chunks:
        if c.get("sub"):
            c["parent"] = parent_of_label[c["label"]]
    return chunks


def main():
    laws = json.loads((config.DATA / "ordinances.json").read_text(encoding="utf-8"))
    unit = config.CHUNK_UNIT
    chunks = make_chunks(laws, unit)
    (config.DATA / f"chunks_{unit}.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"조각 {len(chunks)}개 ({unit} 단위)")

    if not config.UPSTAGE_API_KEY:
        print("⚠ API 키가 없어 임베딩은 건너뜁니다. 키워드 검색만 동작합니다.")
        return
    import upstage
    vecs = upstage.embed([c["text"] for c in chunks], config.EMBED_PASSAGE)
    np.save(config.DATA / f"vectors_{unit}.npy", np.array(vecs, dtype=np.float32))
    print("임베딩 저장 완료")


if __name__ == "__main__":
    main()
