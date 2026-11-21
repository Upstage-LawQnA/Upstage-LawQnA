"""시연용 자치법규 저장소.

제출용은 서울특별시 조례 6종을 미리 색인해 두지만, 시연용은 지역을 고르면 그 지역의 모든 자치법규(조례·규칙)가 대상입니다.
질문이 들어오면
  1) 법제처 OPEN API로 선택 지역의 관련 자치법규를 찾고 (제목 검색 + 본문 검색)
  2) 질문과 가까운 몇 개를 골라 원문을 받아 조·항 단위로 나누고 임베딩해 색인에 덧붙인 뒤
  3) 그 자치법규들 안에서만 제출용과 같은 방식(rag.answer)으로 답합니다.
받은 원문·조각·임베딩은 data/cache/에 저장해 두어 같은 조례는 다시 받지 않습니다.
기존 서울특별시 조례 6종은 제출용 색인(별표 Document Parse 포함)을 그대로 씁니다.
"""
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import requests

import config
import rag
from build_index import make_chunks
from fetch_ordinances import PAGE_URL, SEARCH_URL, SERVICE_URL, parse_xml

CACHE = config.DATA / "cache"
PICK = 6            # 질문마다 답변 근거로 쓸 자치법규 수 (시연용: 4→6, 제목에 시설명이 없는 조례를 놓치는 일 줄임)
SEARCH_TTL = 3600   # 검색 결과를 메모리에 두는 시간(초)

# 법제처 OPEN API의 시·도 기관코드 (검색 결과의 지자체기관명으로 확인한 값)
REGIONS = {
    "서울특별시": "6110000", "부산광역시": "6260000", "대구광역시": "6270000", "인천광역시": "6280000",
    "대전광역시": "6300000", "울산광역시": "6310000", "세종특별자치시": "5690000", "경기도": "6410000",
    "강원특별자치도": "6530000", "충청북도": "6430000", "충청남도": "6440000", "전북특별자치도": "6540000",
    "전남광주통합특별시": "6130000",   # 2026-07-01 광주광역시·전라남도 통합 (법제처 기관코드)
    "경상북도": "6470000", "경상남도": "6480000", "제주특별자치도": "6500000",
}
_lock = threading.Lock()
LAWS = {}          # 자치법규명 → 정보(mst, 지역, 종류, 담당부서, 전화, 시행일, 원문 링크)
_loaded = set()    # 색인에 들어간 자치법규명
_search_cache = {}
_title_vec = {}


def region_list():
    """선택할 수 있는 지역: 시·도만 (시·군·구는 넣지 않음). 시·도를 고르면 그 시·도 자체의 자치법규만 씁니다."""
    return [{"name": r, "group": "시·도"} for r in REGIONS]


def _top(region):
    return region.split()[0]


# ------------------------------------------------------------ 저장소 (rag._store)
def _info_from_xml(xml_bytes):
    info = ET.fromstring(xml_bytes).find("자치법규기본정보")
    g = (lambda k: (info.findtext(k) or "").strip()) if info is not None else (lambda k: "")
    return {"dept": g("담당부서명"), "phone": g("전화번호"), "org": g("지자체기관명"),
            "kind": "규칙" if g("자치법규종류").startswith("R") or g("자치법규명").endswith("규칙") else "조례"}


def _install(chunks, vecs):
    """조각을 저장소 끝에 덧붙입니다 (id·parent를 이어 붙인 위치로 옮김). _lock 안에서 부릅니다."""
    old = rag._store
    base = len(old[0]) if old else 0
    moved = []
    for c in chunks:
        c = dict(c)
        c["id"] = c["id"] + base
        c["parent"] = c["parent"] + base
        moved.append(c)
    docs_new = [Counter(rag._tokens(c["label"] + " " + c["text"])) for c in moved]
    v = np.asarray(vecs, dtype=np.float32)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    if old:
        all_chunks = old[0] + moved
        all_vecs = np.vstack([old[1], v])
        docs = old[2] + docs_new
    else:
        all_chunks, all_vecs, docs = moved, v, docs_new
    df = Counter(g for d in docs for g in d)
    avg = sum(sum(d.values()) for d in docs) / max(len(docs), 1)
    rag._store = (all_chunks, all_vecs, docs, df, avg)
    rag.reset_caches()


def init():
    """기존 서울특별시 조례 6종(제출용 색인)을 저장소에 넣습니다."""
    if rag._store is not None:
        return
    CACHE.mkdir(exist_ok=True)
    for d in ("raw", "chunks"):
        (CACHE / d).mkdir(exist_ok=True)
    chunks = json.loads((config.DATA / "chunks_paragraph.json").read_text(encoding="utf-8"))
    vecs = np.load(config.DATA / "vectors_paragraph.npy")
    for o in json.loads((config.DATA / "ordinances.json").read_text(encoding="utf-8")):
        raw = config.DATA / "raw" / f"{o['mst']}.xml"
        extra = _info_from_xml(raw.read_bytes()) if raw.exists() else {}
        LAWS[o["name"]] = {"name": o["name"], "mst": o["mst"], "region": "서울특별시", "kind": "조례",
                           "effective_date": o["effective_date"], "source_url": o["source_url"],
                           "dept": extra.get("dept", ""), "phone": extra.get("phone", "")}
        _loaded.add(o["name"])
    with _lock:
        _install(chunks, vecs)
    # 6종에도 부칙을 덧붙입니다 (제출용 색인 파일은 건드리지 않고 시연용 캐시에 따로 저장)
    for o in json.loads((config.DATA / "ordinances.json").read_text(encoding="utf-8")):
        raw = config.DATA / "raw" / f"{o['mst']}.xml"
        cpath, vpath = CACHE / "chunks" / f"addenda_{o['mst']}.json", CACHE / "chunks" / f"addenda_{o['mst']}.npy"
        try:
            if cpath.exists() and vpath.exists():
                add, av = json.loads(cpath.read_text(encoding="utf-8")), np.load(vpath)
            elif raw.exists():
                first = next(c for c in chunks if c.get("law") == o["name"])
                meta = {k: first[k] for k in ("law", "effective_date", "source_url") if k in first}
                add = _addenda_chunks(raw.read_bytes(), o["name"], meta)
                if not add:
                    continue
                import upstage
                av = np.array(upstage.embed([c["text"][:3000] for c in add], config.EMBED_PASSAGE), dtype=np.float32)
                for i, c in enumerate(add):
                    c["id"] = c["parent"] = i
                _save(cpath, json.dumps(add, ensure_ascii=False).encode("utf-8"))
                np.save(vpath, av)
            else:
                continue
            with _lock:
                _install(add, av)
        except Exception:  # noqa: BLE001
            continue
        finally:
            if raw.exists():
                try:
                    remember_addenda(raw.read_bytes(), o["name"])
                except Exception:  # noqa: BLE001
                    pass


ANNEX_LABEL = re.compile(r"\[?\s*(별표)\s*(\d+)?\s*(?:의\s*(\d+))?\s*\]?")
BOX_TABLE = re.compile(r"[┌│├└┼┬┴┤]")
ANNEX_USEFUL = re.compile(r"요금|감면|면제|할인|사용료|이용료|입장료|수수료|과태료|부과|기준|금액|대상|단가|보상|지원")


def _parse_annex_file(url, ftype, mst, label, clean_title):
    from parse_annex import annex_path
    path = annex_path(mst, label)
    try:
        src = config.DATA / "annex_files" / f"{mst}_{label.replace(' ', '')}.{ftype or 'hwp'}"
        src.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            r = requests.get(url, timeout=(6, 60))
            r.raise_for_status()
            src.write_bytes(r.content)
        import upstage
        md = upstage.document_parse(src)
        if md.strip():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {label} {clean_title}\n\n{md}", encoding="utf-8")
    except Exception:  # noqa: BLE001  별표를 못 읽어도 조문으로는 답할 수 있음
        pass


def _annexes(xml_bytes, mst):
    """별표를 읽습니다. 최근 자치법규는 별표 본문이 비어 있고 첨부파일(hwp·hwpx·pdf)만 있어서,
    요금·감면·기준처럼 답에 쓰일 별표만 파일을 받아 Document Parse로 읽습니다 (제출용 parse_annex와 같은 방식).
    표지판·서식 그림 같은 별표는 건너뜁니다. 읽은 결과는 data/annex/에 저장해 다시 읽지 않습니다."""
    from parse_annex import annex_path
    rows, jobs = [], []
    for b in ET.fromstring(xml_bytes).iter("별표단위"):
        title = (b.findtext("별표제목") or "").strip()
        if "별지" in title[:8]:
            continue
        m = ANNEX_LABEL.match(title)
        if m:                                 # 제목이 "[별표 1] …"로 시작하면 그 번호
            no, sub = int(m.group(2) or 0), int(m.group(3) or 0)
        elif (b.findtext("별표구분") or "").strip() == "별표":
            # 제목에 [별표 N] 표시가 없는 별표 (예: "서울형 키즈카페 이용료 감면 기준") → 별표번호·가지번호 칸으로
            # (이걸 서식으로 보고 건너뛰어, '감면 대상은 별표 1과 같다'는 조례에서 별표가 빠졌었음)
            no, sub = int(b.findtext("별표번호") or 0), int(b.findtext("별표가지번호") or 0)
        else:
            continue                          # 별표 표시 없는 서식(신청서·고지서 등)
        label = "별표" + (f" {no}" if no else "") + (f"의{sub}" if sub else "")
        clean_title = re.sub(r"^\[[^\]]*\]\s*", "", title)
        body = (b.findtext("별표내용") or "").strip()
        url, ftype = (b.findtext("별표첨부파일명") or "").strip(), (b.findtext("별표첨부파일구분") or "hwp").strip()
        # 본문이 없거나, 괘선(┌│) 표라 칸이 줄마다 쪼개진 별표는 원본 파일을 Document Parse로 읽습니다
        # (예: 키즈카페 감면 기준표의 '입장료 / 전액 / 감면'이 세 줄로 나뉘어 '전액 감면'을 못 읽었음)
        if (not body or BOX_TABLE.search(body)) and not annex_path(mst, label).exists() and url \
                and ANNEX_USEFUL.search(clean_title):
            jobs.append((url, ftype, mst, label, clean_title))
        rows.append((label, clean_title, body, url, ftype))
    if jobs:
        with ThreadPoolExecutor(min(4, len(jobs))) as ex:
            list(ex.map(lambda j: _parse_annex_file(*j), jobs))
    return [{"label": lb, "title": t, "text": body, "file_url": url, "file_type": ft}
            for lb, t, body, url, ft in rows if body or annex_path(mst, lb).exists()]


ADDENDA_KEEP = 3   # 최근 부칙 몇 개를 넣을지 (시행일·적용례·경과조치: "언제부터 적용되나요?"의 근거)


def _addenda_chunks(xml_bytes, name, meta):
    """부칙을 조각으로 만듭니다. 원문 XML의 <부칙>은 공포일자·공포번호·내용이 차례로 반복됩니다."""
    b = ET.fromstring(xml_bytes).find("부칙")
    if b is None:
        return []
    items, cur = [], None
    for c in b:
        if c.tag == "부칙공포일자":
            cur = {"date": (c.text or "").strip()}
            items.append(cur)
        elif cur is not None and c.tag == "부칙공포번호":
            cur["no"] = (c.text or "").strip()
        elif cur is not None and c.tag == "부칙내용":
            cur["text"] = (cur.get("text", "") + "\n" + (c.text or "")).strip()
    out = []
    for it in [i for i in items if i.get("text") and len(i.get("date", "")) == 8][-ADDENDA_KEEP:]:
        d = it["date"]
        label = f"{name} 부칙({d[:4]}. {int(d[4:6])}. {int(d[6:])}." + (f" 제{it['no']}호" if it.get("no") else "") + ")"
        text = re.sub(r"[ \t]+\n", "\n", it["text"]).strip()
        out.append({**meta, "label": label, "article": label, "text": f"[{label}]\n{text}"})
    return out


ADDENDA = {}   # 자치법규명 → 모든 부칙 [{label, text, date}] (검색 색인에는 최근 3개만, 나머지는 조항 언급으로 붙임)


def _parse_addenda(xml_bytes):
    b = ET.fromstring(xml_bytes).find("부칙")
    if b is None:
        return []
    items, cur = [], None
    for c in b:
        if c.tag == "부칙공포일자":
            cur = {"date": (c.text or "").strip()}
            items.append(cur)
        elif cur is not None and c.tag == "부칙공포번호":
            cur["no"] = (c.text or "").strip()
        elif cur is not None and c.tag == "부칙내용":
            cur["text"] = (cur.get("text", "") + "\n" + (c.text or "")).strip()
    return [i for i in items if i.get("text") and len(i.get("date", "")) == 8]


def remember_addenda(xml_bytes, name):
    """그 자치법규의 모든 부칙을 ADDENDA에 보관합니다 (임베딩 없음)."""
    if name in ADDENDA:
        return
    out = []
    for it in _parse_addenda(xml_bytes):
        d = it["date"]
        label = f"{name} 부칙({d[:4]}. {int(d[4:6])}. {int(d[6:])}." + (f" 제{it['no']}호" if it.get("no") else "") + ")"
        out.append({"label": label, "date": d, "text": f"[{label}]\n" + re.sub(r"[ \t]+\n", "\n", it["text"]).strip()})
    ADDENDA[name] = out


def _with_addenda(xml_bytes, name, chunks, vecs, save_to=None):
    """조각에 부칙이 없으면 만들어 덧붙이고(임베딩 포함) 저장합니다."""
    if not chunks or any(" 부칙(" in c["label"] for c in chunks):
        return chunks, vecs
    meta = {k: chunks[0][k] for k in ("law", "effective_date", "source_url") if k in chunks[0]}
    add = _addenda_chunks(xml_bytes, name, meta)
    if not add:
        return chunks, vecs
    import upstage
    av = np.array(upstage.embed([c["text"][:3000] for c in add], config.EMBED_PASSAGE), dtype=np.float32)
    base = len(chunks)
    for i, c in enumerate(add):
        c["id"] = c["parent"] = base + i
    chunks, vecs = chunks + add, np.vstack([vecs, av])
    if save_to:
        cpath, vpath = save_to
        _save(cpath, json.dumps(chunks, ensure_ascii=False).encode("utf-8"))
        np.save(vpath, vecs)
    return chunks, vecs


def _fetch_law(cand):
    """원문을 받아(또는 캐시에서 읽어) 조각과 임베딩을 만듭니다."""
    mst = cand["mst"]
    raw = CACHE / "raw" / f"{mst}.xml"
    if not raw.exists():
        for attempt in range(3):
            try:
                r = requests.get(SERVICE_URL, params=dict(OC=config.LAW_OC, target="ordin", MST=mst, type="XML"),
                                 timeout=(6, 40))
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(0.5)
        _save(raw, r.content)
    xml_bytes = raw.read_bytes()
    extra = _info_from_xml(xml_bytes)
    cpath, vpath = CACHE / "chunks" / f"{mst}.json", CACHE / "chunks" / f"{mst}.npy"
    if cpath.exists() and vpath.exists():
        chunks, vecs = json.loads(cpath.read_text(encoding="utf-8")), np.load(vpath)
    else:
        law = parse_xml(xml_bytes, cand["name"], mst)
        law["annexes"] = _annexes(xml_bytes, mst)
        chunks = make_chunks([law], "paragraph")
        if not chunks:
            return cand, extra, [], None
        import upstage
        vecs = np.array(upstage.embed([c["text"][:3000] for c in chunks], config.EMBED_PASSAGE), dtype=np.float32)
        cpath.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        np.save(vpath, vecs)
    try:
        chunks, vecs = _with_addenda(xml_bytes, cand["name"], chunks, vecs, save_to=(cpath, vpath))
        remember_addenda(xml_bytes, cand["name"])
    except Exception:  # noqa: BLE001  부칙은 보조 근거라 실패해도 본문은 그대로
        pass
    return cand, extra, chunks, vecs


def ensure(cands):
    """고른 자치법규를 색인에 넣습니다 (이미 있으면 건너뜀)."""
    todo = [c for c in cands if c["name"] not in _loaded]
    if not todo:
        return
    with ThreadPoolExecutor(min(4, len(todo))) as ex:
        results = list(ex.map(_fetch_law, todo))
    with _lock:
        for cand, extra, chunks, vecs in results:
            if cand["name"] in _loaded or not chunks:
                continue
            LAWS[cand["name"]] = {**cand, "dept": extra.get("dept", ""), "phone": extra.get("phone", ""),
                                  "kind": extra.get("kind") or cand.get("kind", "조례")}
            _install(chunks, vecs)
            _loaded.add(cand["name"])


# ------------------------------------------------------------ 과거 시점 기준 (응답소 민원 검증용)
_versions_cache = {}


def versions(name, region):
    """자치법규의 연혁(과거 버전) 목록: [{mst, ef(YYYYMMDD)}] 시행일 순. 법제처 검색의 nw=2(연혁 포함)를 씁니다."""
    key = (name, region)
    if key in _versions_cache:
        return _versions_cache[key]
    code = REGIONS.get(_top(region))

    def page(p):          # 자치구 같은 이름 조례의 연혁이 앞쪽을 채우므로 모든 쪽을 봅니다 (최대 20쪽)
        params = dict(OC=config.LAW_OC, target="ordin", type="XML", display=100, page=p, query=name, nw=2)
        if code:
            params["org"] = code
        for attempt in range(3):
            try:
                r = requests.get(SEARCH_URL, params=params, timeout=(6, 30))
                r.raise_for_status()
                return ET.fromstring(r.content)
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(0.5)
    first = page(1)
    total = int(first.findtext("totalCnt") or 0)
    roots = [first]
    rest = list(range(2, min(20, (total + 99) // 100) + 1))
    if rest:
        with ThreadPoolExecutor(6) as ex:
            roots += list(ex.map(page, rest))
    out = []
    for root in roots:
        for law in root.findall("law"):
            if (law.findtext("자치법규명") or "").strip() == name and (law.findtext("지자체기관명") or "").strip() == region:
                out.append({"mst": law.findtext("자치법규일련번호"), "ef": law.findtext("시행일자") or ""})
    out = sorted({v["mst"]: v for v in out}.values(), key=lambda v: (v["ef"], int(v["mst"] or 0)))
    _versions_cache[key] = out
    return out


def as_of(laws, region, day):
    """고른 자치법규를 day(YYYY-MM-DD)에 시행 중이던 버전으로 바꿔 색인에 넣습니다.
    지금 버전과 다르면 이름 뒤에 '(YYYY-MM-DD 시행본)'을 붙여 지금 버전과 섞이지 않게 합니다.
    돌려주는 값: (바꾼 목록, 그날 아직 없던 자치법규 이름 목록)"""
    d = day.replace("-", "")
    out, missing = [], []
    for lw in laws:
        try:
            vs = [v for v in versions(lw["name"], region) if v["ef"] and v["ef"] <= d]
        except requests.RequestException:
            out.append(lw)
            continue
        if not vs:
            missing.append(lw["name"])
            continue
        v = vs[-1]
        if v["mst"] == str(lw.get("mst")):
            out.append(lw)
            continue
        ef = f"{v['ef'][:4]}-{v['ef'][4:6]}-{v['ef'][6:]}"
        out.append({"mst": v["mst"], "name": f"{lw['name']} ({ef} 시행본)", "region": region, "kind": lw.get("kind", "조례"),
                    "effective_date": ef, "source_url": PAGE_URL.format(mst=v["mst"]), "base_name": lw["name"]})
    ensure([c for c in out if c.get("base_name")])
    return [LAWS.get(c["name"], c) for c in out if c["name"] in LAWS], missing


# ------------------------------------------------------------ 검색 (법제처 OPEN API)
def _search(query, region, body=False, display=100, sort=None, page=1, more_pages=False):
    key = (query, region, body, display, sort, page, more_pages)
    hit = _search_cache.get(key)
    if hit and time.time() - hit[0] < SEARCH_TTL:
        return hit[1]
    params = dict(OC=config.LAW_OC, target="ordin", type="XML", display=display, page=page)
    if query:
        params["query"] = query
    if body:
        params["search"] = 2
    if sort:
        params["sort"] = sort
    code = REGIONS.get(_top(region))
    if code:
        params["org"] = code
    for attempt in range(3):                  # 법제처 서버가 가끔 연결을 늦게 받습니다
        try:
            r = requests.get(SEARCH_URL, params=params, timeout=(6, 20))
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(0.5)
    today = date.today().strftime("%Y%m%d")
    root = ET.fromstring(r.content)
    laws = root.findall("law")
    # 제목 검색은 결과를 모든 쪽 받습니다 (최대 5쪽). 시·도 기관코드 결과는 자치구 것이 이름순으로 앞을 채워서,
    # 첫 100개만 보면 '서울특별시 청년 기본 조례'처럼 시 조례가 빠졌음. 본문 검색은 결과가 너무 많아 첫 쪽만.
    if (not body or more_pages) and page == 1 and display == 100 and not sort:
        total = int(root.findtext("totalCnt") or 0)
        more = list(range(2, min(4 if body else 6, (total + 99) // 100 + 1)))
        if more:
            with ThreadPoolExecutor(len(more)) as ex:
                for rows in ex.map(lambda p: _search(query, region, body=body, display=100, page=p), more):
                    laws += rows
    out = []
    for law in laws:
        if isinstance(law, dict):             # 다른 쪽에서 이미 고른 결과
            out.append(law)
            continue
        if (law.findtext("지자체기관명") or "").strip() != region:
            continue
        eff = law.findtext("시행일자") or ""
        if eff > today:                       # 아직 시행 전(또는 99991231 같은 자리표시)인 것은 뺍니다
            continue
        mst = law.findtext("자치법규일련번호")
        out.append({"mst": mst, "name": (law.findtext("자치법규명") or "").strip(), "region": region,
                    "kind": (law.findtext("자치법규종류") or "조례").strip(),
                    "law_id": law.findtext("자치법규ID"),
                    "effective_date": f"{eff[:4]}-{eff[4:6]}-{eff[6:]}" if len(eff) == 8 else eff,
                    "source_url": PAGE_URL.format(mst=mst)})
    _search_cache[key] = (time.time(), out)
    return out


def _embed_titles(names):
    new = [n for n in names if n not in _title_vec]
    if new:
        import upstage
        for n, v in zip(new, upstage.embed(new, config.EMBED_PASSAGE)):
            v = np.asarray(v, dtype=np.float32)
            _title_vec[n] = v / np.linalg.norm(v)
    return np.array([_title_vec[n] for n in names])


def candidates(question, region, keywords, title_words=()):
    """선택 지역에서 질문과 관련 있어 보이는 자치법규 후보를 점수순으로 돌려줍니다.
    제목 검색(조례 이름에 쓰이는 말 + 검색어)이 주 신호이고, 본문 검색은 보조입니다
    (본문 검색은 시·도 전체 결과 100개가 이름순이라 자치구 것에 밀려 빠지는 일이 많음)."""
    tws = [k for k in dict.fromkeys(k.strip() for k in title_words) if k][:3]
    kws = [k for k in dict.fromkeys(k.strip() for k in keywords) if k][:3]
    jobs = list(dict.fromkeys([(q, False) for q in tws + kws] + [(q, True) for q in kws]))
    with ThreadPoolExecutor(max(1, len(jobs))) as ex:
        results = list(ex.map(lambda j: _safe_search(j[0], region, j[1]), jobs))
    cand, name_hits, body_hits, body_terms = {}, Counter(), Counter(), {}
    for (q, body), rows in zip(jobs, results):
        for c in rows:
            key = c["law_id"] or c["name"]
            if key not in cand or c["effective_date"] > cand[key]["effective_date"]:
                cand[key] = c
            (body_hits if body else name_hits)[key] += 1
            if body:
                body_terms.setdefault(key, []).append(q)
    if not cand:
        return []
    keys = list(cand)
    names = [cand[k]["name"] for k in keys]
    import upstage
    qv = np.asarray(upstage.embed([question], config.EMBED_QUERY)[0], dtype=np.float32)
    qv = qv / np.linalg.norm(qv)
    cos = _embed_titles(names) @ qv
    scored = []
    for k, cs in zip(keys, cos):
        c = cand[k]
        # 본문에 질문 검색어가 나오는 자치법규에 가산점 (제목에 시설 이름이 없는 조례: 예) 서울형 키즈카페 →
        # 「아동의 놀이권 보장을 위한 조례」. 제목만 보면 빠져서 6번 중 5번 엉뚱한 조례로 답했음)
        s = float(cs) + 0.08 * name_hits[k] + 0.05 * min(body_hits[k], 3) + (0.01 if c["kind"] == "조례" else 0)
        scored.append({**c, "score": round(s, 4), "body_terms": body_terms.get(k, [])})
    scored.sort(key=lambda c: -c["score"])
    # 서울특별시 6종은 제출용 색인(별표 Document Parse 포함)을 그대로 쓰도록 같은 이름이면 그 정보로 바꿉니다
    return [{**c, **{k: LAWS[c["name"]][k] for k in ("mst", "source_url", "effective_date")}} if c["name"] in LAWS else c
            for c in scored]


def _safe_search(q, region, body):
    try:
        return _search(q, region, body=body, more_pages=body)
    except requests.RequestException:
        return []


RERANK_SYSTEM = """너는 자치법규 검색 도우미다. 시민 질문과 자치법규 제목 목록을 보고,
질문에 답할 조문이 들어 있을 가능성이 높은 자치법규를 관련성 높은 순서로 최대 {k}개 고른다.
- 제목이 다루는 대상·시설·제도가 질문과 직접 관련된 것만 고른다. 특별회계·기금·위원회 운영처럼 질문과 관계없는 것은 고르지 않는다.
- 요금·감면·대상·금지·과태료를 묻는 질문은 그 시설·제도를 '설치·관리·운영'하는 기본 조례를 먼저 고른다.
- 제목에 시설·제도 이름이 없어도, 목록에 '(본문에 나옴: …)'으로 질문의 핵심 시설·제도 이름이 본문에 있다고 표시된 자치법규는
  그 시설·제도를 정한 조례일 가능성이 높으니 먼저 고른다.
- 세부 기준을 정한 시행규칙도 관련 있으면 고른다. 관련 있는 것이 없으면 빈 배열을 낸다. JSON으로 낸다."""


def _rerank(question, cands, k):
    """후보 제목 중 질문에 맞는 것을 LLM이 고릅니다. 실패하면 점수순 그대로."""
    pool = cands[:30]
    if len(pool) <= 1:
        return pool
    import upstage
    names = [c["name"] for c in pool]
    lines = [c["name"] + (f" (본문에 나옴: {', '.join(dict.fromkeys(c['body_terms']))})" if c.get("body_terms") else "")
             for c in pool]
    schema = {"type": "json_schema", "json_schema": {"name": "pick", "strict": True, "schema": {
        "type": "object", "properties": {"picks": {"type": "array", "items": {"type": "string", "enum": names}}},
        "required": ["picks"], "additionalProperties": False}}}
    try:
        out = json.loads(upstage.chat([{"role": "system", "content": RERANK_SYSTEM.format(k=k)},
                                       {"role": "user", "content": f"[질문]\n{question}\n\n[자치법규 제목]\n" + "\n".join(lines)}],
                                      response_format=schema))
        picks = [n for n in dict.fromkeys(out.get("picks", [])) if n in names][:k]
    except Exception:  # noqa: BLE001
        return pool[:k]
    by = {c["name"]: c for c in pool}
    return [by[n] for n in picks]


def pick_laws(question, region, keywords, title_words=(), pinned=None):
    """답변 근거로 쓸 자치법규를 고르고 색인에 넣습니다. pinned: '이 조례로 질문하기'로 고정한 자치법규명."""
    if pinned:
        cands = [c for c in _search(pinned, region) if c["name"] == pinned][:1]
        top, others = cands, []
    else:
        allc = candidates(question, region, keywords, title_words)
        top = _rerank(question, allc, PICK)
        chosen = {c["name"] for c in top}
        others = [c for c in allc if c["name"] not in chosen][:4]
    ensure(top)
    return [LAWS[c["name"]] | {"score": c.get("score")} for c in top if c["name"] in LAWS], others


_recent_cache = {}


def recent(region, n=6, days=150):
    """선택 지역에서 최근 시행된 자치법규 (시행일 내림차순).
    시·도 기관코드로 찾으면 자치구 것이 대부분이라, 최근 시행분(efYd)을 여러 쪽 받아 지역 이름이 같은 것만 남깁니다."""
    hit = _recent_cache.get(region)
    if hit and time.time() - hit[0] < SEARCH_TTL:
        return hit[1][:n]
    from datetime import timedelta
    today = date.today()
    ef = f"{(today - timedelta(days=days)):%Y%m%d}~{today:%Y%m%d}"
    code = REGIONS.get(_top(region))

    def page(p):
        params = dict(OC=config.LAW_OC, target="ordin", type="XML", display=100, page=p, sort="efdes", efYd=ef)
        if code:
            params["org"] = code
        try:
            r = requests.get(SEARCH_URL, params=params, timeout=(6, 20))
            r.raise_for_status()
            return ET.fromstring(r.content)
        except Exception:  # noqa: BLE001
            return None

    first = page(1)
    if first is None:
        return []
    total = int(first.findtext("totalCnt") or 0)
    roots = [first]
    pages = list(range(2, min(12, (total + 99) // 100 + 1)))
    if pages:
        with ThreadPoolExecutor(6) as ex:
            roots += [r for r in ex.map(page, pages) if r is not None]
    rows, seen = [], set()
    for root in roots:
        for law in root.findall("law"):
            name = (law.findtext("자치법규명") or "").strip()
            if (law.findtext("지자체기관명") or "").strip() != region or name in seen:
                continue
            seen.add(name)
            eff, mst = law.findtext("시행일자") or "", law.findtext("자치법규일련번호")
            rows.append({"mst": mst, "name": name, "region": region, "kind": (law.findtext("자치법규종류") or "").strip(),
                         "effective_date": f"{eff[:4]}-{eff[4:6]}-{eff[6:]}", "source_url": PAGE_URL.format(mst=mst)})
    rows.sort(key=lambda c: c["effective_date"], reverse=True)
    ords = [c for c in rows if c["kind"] == "조례"]      # 화면 제목이 '조례'라 조례를 먼저, 모자라면 규칙도
    rows = ords + [c for c in rows if c["kind"] != "조례"] if len(ords) < n else ords
    _recent_cache[region] = (time.time(), rows)
    return rows[:n]


def law_detail(mst):
    """조례 상세 화면용: 기본 정보 · 조항 목록(조 번호·제목) · 제·개정 이유. 원문 XML은 캐시를 씁니다(없으면 법제처에서 받음)."""
    root = ET.fromstring(_raw_xml(str(mst)).encode("utf-8"))
    info = root.find("자치법규기본정보")
    g = (lambda k: (info.findtext(k) or "").strip()) if info is not None else (lambda k: "")
    d = lambda v: f"{v[:4]}-{v[4:6]}-{v[6:]}" if len(v) == 8 else v  # noqa: E731
    arts = []
    jo_root = root.find("조문")
    for jo in (jo_root.findall("조") if jo_root is not None else []):
        if jo.findtext("조문여부") != "Y":
            continue
        body = (jo.findtext("조내용") or "").strip()
        m = re.match(r"(제\s*\d+\s*조(?:\s*의\s*\d+)?)\s*(?:\(([^)]*)\))?", body)
        if not m:
            continue
        label = re.sub(r"\s+", "", m.group(1))
        title = (m.group(2) or jo.findtext("조제목") or "").strip()
        if "삭제" in body[:40] and not title:
            continue
        arts.append({"label": label, "title": title})
    reason = ""
    zr = root.find("제개정이유")
    if zr is not None:
        reason = "\n".join((c.text or "").strip() for c in zr if (c.text or "").strip()) or (zr.text or "").strip()
    reason = re.sub(r"\n{3,}", "\n\n", reason).strip()
    name = g("자치법규명")
    return {"mst": str(mst), "name": name, "kind": "규칙" if g("자치법규종류").startswith("R") or name.endswith("규칙") else "조례",
            "region": g("지자체기관명"), "effective_date": d(g("시행일자")), "promulgated": d(g("공포일자")),
            "revision": g("제개정정보"), "dept": g("담당부서명"), "phone": g("전화번호"),
            "source_url": PAGE_URL.format(mst=mst), "articles": arts, "reason": reason[:2500], "tag": tag_of(name)}


TAGS = [("주차", "주차"), ("금연|흡연", "금연"), ("동물|반려", "동물"), ("자전거", "자전거"), ("공원|녹지", "공원"),
        ("청년", "청년"), ("교통|버스|도로|보행", "교통"), ("복지|장애|노인|아동|출산|임산부", "복지"),
        ("환경|폐기물|청소|기후", "환경"), ("세|수수료|요금|사용료", "세금·요금"), ("문화|관광|체육", "문화")]


def tag_of(name):
    for pat, tag in TAGS:
        if re.search(pat, re.sub(r"^\S+(?:\s\S+구)?\s", "", name)):
            return tag
    return "규칙" if name.endswith("규칙") else "조례"


def label_text(label):
    """인용 라벨의 원문 (항이 호별 조각으로 나뉘어 있어도 원래 조·항 조각 하나)."""
    chunks = rag._load()[0]
    parts = [c["text"] for c in chunks if c["label"] == label and not c.get("sub")]
    return "\n".join(dict.fromkeys(parts))


def law_of_label(label):
    best = ""
    for name in LAWS:
        if label.startswith(name) and len(name) > len(best):
            best = name
    return LAWS.get(best)


# ------------------------------------------------------------ 정의 조항 따라가기 (제안서 4-4 '여러 조항 결합')
QUOTED_TERM = re.compile(r"[“\"‘]([가-힣A-Za-z0-9·ㆍ ]{2,20}?)[”\"’]")


def _save(path, data):
    """임시 파일에 쓴 뒤 이름을 바꿉니다 (디스크가 차서 쓰다 끊겨도 깨진 캐시가 남지 않게)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _raw_xml(mst):
    raw = CACHE / "raw" / f"{mst}.xml"
    if not raw.exists():
        r = requests.get(SERVICE_URL, params=dict(OC=config.LAW_OC, target="ordin", MST=mst, type="XML"), timeout=(6, 40))
        r.raise_for_status()
        _save(raw, r.content)
    return raw.read_bytes().decode("utf-8", "replace")


def _defines(text, term):
    t = re.escape(term).replace("\ ", r"\s*")
    return re.search(r"[“\"]?" + t + r"[”\"]?\s*(?:이)?(?:란|이라\s*함은|라\s*함은)", text) is not None


_Q_STOP = {"서울", "서울시", "서울특별시", "조례", "경우", "규정", "기준", "관련", "대상", "어떻게", "무엇", "얼마", "있나요", "되나요", "문의"}


def _def_relevant(text, term, question):
    """정의문("○○"란 … 말한다)에 질문 낱말이 들어 있을 때만 그 정의를 근거로 붙입니다.
    예) 키즈카페 '다둥이 행복카드' 정의 "… 다자녀 가족을 위한 우대용 카드" ↔ 질문 '다자녀' → 붙임
        경차 질문에 주차장 조례의 '전기자동차' 정의 → 질문과 무관해서 안 붙임 (답 첫 문장에 전기자동차 얘기가 끼던 일)"""
    t = re.escape(term).replace(r"\ ", r"\s*")
    m = re.search(r"[“\"]?" + t + r"[”\"]?\s*(?:이)?(?:란|이라\s*함은|라\s*함은)(.{0,200}?)(?:말한다|뜻한다|$)", text)
    if not m:
        return False
    phrase = re.sub(r"\s+", "", m.group(1))
    words = {re.sub(r"(은|는|이|가|을|를|의|에|도|로|으로|와|과|인데|이면)$", "", w) for w in re.findall(r"[가-힣]{2,}", question)}
    return any(len(w) >= 2 and w not in _Q_STOP and w in phrase for w in words)


def definition_hits(question, hits, region, have_laws, max_terms=2):
    """근거에 따옴표로 적힌 용어(예: “다둥이 행복카드”)가 질문에 없고 고른 조례 안에서도 정의되지 않으면,
    같은 지역의 다른 자치법규에서 그 용어를 정의한 조항을 찾아 근거로 붙입니다.
    예) 키즈카페 감면 기준표의 '다둥이 행복카드를 소지한 가족' ↔ 질문의 '다자녀 가정':
        「서울특별시 출산 및 양육지원에 관한 조례」 제2조 "다둥이 행복카드란 … 다자녀 가족을 위한 우대용 카드"
    돌려주는 값: (붙일 조각 목록, 새로 쓴 자치법규 이름 목록)"""
    q = re.sub(r"\s+", "", question)
    have_text = " ".join(h["text"] for h in hits)
    terms = []
    for h in hits:
        for m in QUOTED_TERM.finditer(h["text"]):
            term = m.group(1).strip()
            if len(re.sub(r"\s+", "", term)) < 4 or re.sub(r"\s+", "", term) in q or term in terms:
                continue
            if _defines(have_text, term):              # 이미 근거 안에서 정의됨
                continue
            terms.append(term)
    out, new_laws = [], []
    have_ids = {h["id"] for h in hits}
    chunks_all = rag._load()[0]
    for term in terms[:max_terms]:
        # 이미 고른 자치법규가 정의하고 있는데 검색에서 그 조만 빠진 경우 (예: 출산 및 양육지원 조례를 골랐지만 제2조는 검색 밖)
        own = [ch for ch in chunks_all if ch.get("law") in have_laws and not ch.get("sub")
               and ch.get("id") not in have_ids and _defines(ch["text"], term)]
        own = [ch for ch in own if _def_relevant(ch["text"], term, question)]
        if own:
            out += [{**own[0], "score": 0.0, "cos": None, "related": [], "definition_of": term}]
            continue
        try:
            rows = _search(term, region, body=True, more_pages=True)
        except requests.RequestException:
            continue
        pool = [c for c in rows if c["name"] not in have_laws][:40]

        def check(c):
            try:
                return _defines(_raw_xml(c["mst"]), term)
            except requests.RequestException:
                return False
        with ThreadPoolExecutor(6) as ex:       # 원문은 캐시에 남아 다음부터는 바로 확인
            flags = list(ex.map(check, pool))
        for c, ok in zip(pool, flags):
            if not ok:
                continue
            ensure([c])
            chunks = rag._load()[0]
            found = [ch for ch in chunks if ch.get("law") == c["name"] and not ch.get("sub") and _defines(ch["text"], term)
                     and _def_relevant(ch["text"], term, question)]
            if found:
                out += [{**ch, "score": 0.0, "cos": None, "related": [], "definition_of": term} for ch in found[:1]]
                new_laws.append(c["name"])
                break
    return out, new_laws
