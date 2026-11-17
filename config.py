"""스타터 킷 공통 설정.

여기 값만 바꿔도 시연의 '전/후' 장면을 재현할 수 있습니다.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# .env 파일이 있으면 읽습니다 (python-dotenv가 없으면 환경변수만 사용)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY", "")
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"

# 법제처 OPEN API 인증값(OC). 'test'로도 응답하지만, 대회용으로는 open.law.go.kr에서 개인 OC를 신청하세요.
LAW_OC = os.environ.get("LAW_OC", "test")

# ---- 대상: 서울특별시 조례 6종 (자치구 조례는 대상이 아닙니다) ----
TARGET_ORDINANCES = [
    "서울특별시 주차장 설치 및 관리 조례",
    "서울특별시 동물보호 조례",
    "서울특별시 자전거이용 활성화에 관한 조례",
    "서울특별시 도시공원 조례",
    "서울특별시 청년 기본 조례",  # 확인 필요: 대회 운영진이 말한 '청년' 조례가 이것인지
    "서울특별시 금연환경 조성 및 간접흡연 피해방지조례",
]
TARGET_ORG = "서울특별시"      # 지자체기관명이 정확히 이것인 것만 받습니다
BASE_DATE = "20261001"          # 기준 시점: 2026년 10월 1일 시행 조례

# ---- 모델 ----
CHAT_MODEL = "solar-pro4"
EMBED_PASSAGE = "solar-embedding-2-passage"   # 조문 조각용
EMBED_QUERY = "solar-embedding-2-query"       # 질문용 (질문과 조각은 서로 다른 모델)

# ---- 시연 스위치 ----
# 2-4 시연: "article"(조 단위) → "paragraph"(항 단위)로 바꾸면 인용이 항까지 맞기 시작합니다
CHUNK_UNIT = os.environ.get("CHUNK_UNIT", "paragraph")
# 2-2 시연: 별표를 Document Parse 결과로 읽을지(True), 법제처 원문 괘선 텍스트로 읽을지(False)
USE_PARSED_ANNEX = os.environ.get("USE_PARSED_ANNEX", "1") == "1"
# 2-5 시연: 근거 검증 장치를 켤지
USE_VERIFY = os.environ.get("USE_VERIFY", "1") == "1"

TOP_K = int(os.environ.get("TOP_K", "10"))   # 6/8/10 비교: 10이 가장 좋음 (정답 조항이 7~10위에 걸리는 경우를 살림)
EMBED_WEIGHT = 0.9    # 검색 점수 = 0.9 × 의미(임베딩) + 0.1 × 키워드
# 검색 개선 스위치 (0으로 끄면 기준선과 비교할 수 있습니다)
USE_REWRITE = os.environ.get("USE_REWRITE", "1") == "1"        # 질문을 조문 용어로 재작성해 함께 검색
EXPAND_SIBLINGS = os.environ.get("EXPAND_SIBLINGS", "1") == "1"  # 상위 조의 다른 항을 함께 근거로
EXPAND_TOP = int(os.environ.get("EXPAND_TOP", "6"))     # 상위 몇 개 조까지 다른 항을 붙일지
MAX_CONTEXT = int(os.environ.get("MAX_CONTEXT", "24"))  # Solar에 넘기는 근거 조각 최대 개수
# 호가 많은 긴 항(예: 주차요금 감면 대상 15개 호)은 검색에 걸린 호만 남기고 나머지는 생략
TRIM_HO = os.environ.get("TRIM_HO", "0") == "1"
TRIM_MIN_CHARS = int(os.environ.get("TRIM_MIN_CHARS", "800"))   # 이보다 짧은 항은 그대로
TRIM_MARGIN = float(os.environ.get("TRIM_MARGIN", "0.05"))     # 최고 호 점수와 이만큼 차이 안에 든 호는 남김
TRIM_KEEP = int(os.environ.get("TRIM_KEEP", "3"))               # 남길 호 최대 개수
# LLM이 근거를 고르는 방식: "label" = 조항 라벨을 enum으로 고름 / "index" = [근거 N] 번호 (예전 방식)
# 번호 방식은 "[근거 1] … 제2항"에서 1·2를 헷갈려 옆 항을 고르는 일이 있었습니다 (F01: 8회 중 5~6회)
EVIDENCE_KEY = os.environ.get("EVIDENCE_KEY", "label")
CLEAN_AMEND = os.environ.get("CLEAN_AMEND", "0") == "1"          # <개정 …>·[본조신설 …] 꼬리표를 빼고 넣기
# 1차 답변이 '확인 불가'면 추론 모드로 한 번 더 시도 (근거를 찾고도 답이 뒤집히는 일 방지)
RETRY_ON_UNKNOWN = os.environ.get("RETRY_ON_UNKNOWN", "1") == "1"
RETRY_EFFORT = "low"
RETRY_DEADLINE_SEC = 60   # 질문 하나에 2차 시도까지 포함해 이 시간 안에 끝냅니다 (서버 상한 75초, 대회 제한 90초)
RETRY_RESEARCH = os.environ.get("RETRY_RESEARCH", "1") == "1"   # 2차 시도 때 검색도 다시 해서 빠진 조항을 보충
VERIFY_SECOND_OPINION = os.environ.get("VERIFY_SECOND_OPINION", "1") == "1"   # 검증 탈락이면 한 번 더 물어 둘 다 탈락일 때만 떨어뜨림
# 검색 점수가 이보다 낮으면 '확인 불가' 후보로 봅니다. TODO: 공개 예시로 직접 맞춰 보세요.
MIN_SCORE = 0.25

# 대회 제한은 문항당 90초. 여유를 두고 이 시간을 넘기면 '확인 불가'로 돌려줍니다.
ANSWER_TIMEOUT_SEC = 75
