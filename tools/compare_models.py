"""챗봇 모델 후보 비교.

AI 서버에서 돌립니다. 키는 이 파일에 없고 .env 에서 읽습니다.

    cd /home/ubuntu/14th-ISIX-ai
    PYTHONPATH=. venv/bin/python tools/compare_models.py gpt-4o-mini gpt-5.4-mini

**운영에 아무 영향이 없습니다.** .env 를 수정하지 않고 서비스도 재시작하지
않습니다. 프로세스 안에서만 모델 이름을 바꿔 호출합니다.

프롬프트와 요약 검증은 운영 모듈 그대로 씁니다.

    chat_service.build_messages   답변 프롬프트 (chat_prompt 3단 조립 포함)
    chat_summary._INSTRUCTION     요약 지시문
    chat_summary._transcript      요약에 넣는 대화
    chat_summary._parse           Profile 스키마 검증

**운영과 다른 점은 딱 하나, 토큰 상한 파라미터 이름입니다.**
GPT-5 계열은 ``max_tokens`` 를 받지 않고 ``max_completion_tokens`` 를 요구합니다.
운영 코드는 ``max_tokens`` 로 고정돼 있어서, 그대로 두면 후보 모델이 전부
호출 실패로 나와 품질 비교가 불가능합니다. 그래서 여기서만 자동으로 바꿔
붙입니다. 어느 이름을 썼는지는 결과에 그대로 찍습니다.

재는 것 다섯 가지입니다.

    첫 토큰 지연     음성이라 그대로 체감됩니다
    답변 길이        120자 이내여야 합니다
    가슴 수치 언급   profile 이 붙으면 판정 수치가 사라지던 증상
    1차 vs 2차       옷을 바꿨을 때 답변이 갈리는지
    요약 저장        실패하면 개인화가 조용히 죽습니다
"""

import asyncio
import json
import re
import sys
import time
from typing import Optional

from app.core.config import settings
from app.models.chat import ChatRequest, FitContext, Turn
from app.services import chat_prompt, chat_service, chat_summary

DEFAULT_MODELS = ["gpt-4o-mini"]

# 운영 상수를 그대로 씁니다. 다른 값을 쓰면 길이 비교가 무의미해집니다.
ANSWER_MAX_TOKENS = chat_service.MAX_TOKENS
SUMMARY_MAX_TOKENS = chat_summary.MAX_TOKENS

# 아바타는 격자 위가 아닌 실제 계측값에 가깝게 잡았습니다. 격자 정중앙에 두면
# 편차가 0 이 되어 "수치를 말하는지" 를 보기 어렵습니다.
MEASUREMENTS = {
    "shoulder_width": 45.5,
    "chest_circ": 92.0,
    "waist_circ": 72.0,
    "hip_circ": 96.0,
}

WARNINGS = ["팔이 몸에 붙어 어깨가 실제보다 넓게 측정되었을 수 있습니다."]

# 사용자가 어깨만 말했습니다. 그런데도 가슴 수치가 나와야 정상입니다 —
# 어깨는 옷이 바뀌어도 거의 같고, 갈리는 것은 가슴입니다.
PROFILE = {"용도": "출근", "신경쓰는부위": ["shoulder_width"]}

# 두 옷의 판정. 서버가 계산하는 값과 같은 방식으로 미리 넣었습니다.
# 어깨 편차는 두 옷이 같고(+0.5) 가슴 여유가 10.0 대 32.0 으로 갈립니다.
TURNS = [
    {
        "label": "1차 · 슬림 셔츠",
        "message": "출근용으로 셔츠 보고 있어요. 어깨가 늘 끼어서 신경 쓰여요.",
        "garment_id": "shirt_slim",
        "fit": "슬림",
        "fit_report": [
            {"part": "shoulder_width", "actual_ease": -8.5, "ref_ease": -9.0,
             "deviation": 0.5, "verdict": "good"},
            {"part": "chest_circ", "actual_ease": 10.0, "ref_ease": 8.0,
             "deviation": 2.0, "verdict": "good"},
        ],
    },
    {
        "label": "2차 · 오버핏 셔츠",
        "message": "이건 어때요?",
        "garment_id": "shirt_over",
        "fit": "오버핏",
        "fit_report": [
            {"part": "shoulder_width", "actual_ease": -8.5, "ref_ease": -9.0,
             "deviation": 0.5, "verdict": "good"},
            {"part": "chest_circ", "actual_ease": 32.0, "ref_ease": 30.0,
             "deviation": 2.0, "verdict": "good"},
        ],
    },
]

CHEST_NUMBER = re.compile(r"가슴[^.]{0,20}?\d")

# 모델별로 어느 파라미터를 받는지 기억합니다. 매번 실패를 반복하지 않도록.
_TOKEN_PARAM: dict[str, str] = {}


def build(turn: dict, history: list[Turn]) -> ChatRequest:
    return ChatRequest(
        mode="fitting",
        message=turn["message"],
        history=history,
        fit_context=FitContext(
            measurements=MEASUREMENTS,
            warnings=WARNINGS,
            garment_id=turn["garment_id"],
            size="m",
            recommended_size="m",
            fit=turn["fit"],
            fit_report=turn["fit_report"],
            profile=PROFILE,
            past_fittings=[],
        ),
    )


def client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=settings.openai_api_key, timeout=60)


async def create(model: str, limit: int, **kwargs):
    """토큰 상한 파라미터 이름을 맞춰가며 호출합니다.

    운영 코드는 ``max_tokens`` 로 고정입니다. GPT-5 계열은 그것을 거부하고
    ``max_completion_tokens`` 를 요구하므로, 여기서만 한 번 실패한 뒤 바꿔
    재시도하고 어느 쪽이었는지 기억합니다.
    """
    names = [_TOKEN_PARAM[model]] if model in _TOKEN_PARAM \
        else ["max_tokens", "max_completion_tokens"]

    last = None
    for name in names:
        try:
            result = await client().chat.completions.create(
                model=model, **{name: limit}, **kwargs)
            _TOKEN_PARAM[model] = name
            return result
        except Exception as error:
            last = error
            if "max_completion_tokens" not in str(error):
                raise
    raise last


async def answer_of(model: str, turn: dict, history: list[Turn]) -> tuple[str, float]:
    """답변과 첫 토큰 지연. 프롬프트는 운영 코드가 만듭니다."""
    messages = chat_service.build_messages(build(turn, history))

    started = time.monotonic()
    stream = await create(model, ANSWER_MAX_TOKENS, messages=messages, stream=True)

    ttft = 0.0
    chunks: list[str] = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        if not chunks:
            ttft = time.monotonic() - started
        chunks.append(delta)
    return "".join(chunks), ttft


async def summary_of(model: str, turn: dict, history: list[Turn],
                     answer: str) -> Optional[dict]:
    """요약. 지시문과 검증은 운영 코드 그대로입니다."""
    transcript = chat_summary._transcript(
        [t.model_dump() for t in history], turn["message"], answer)

    response = await create(
        model, SUMMARY_MAX_TOKENS,
        messages=[
            {"role": "system", "content": chat_summary._INSTRUCTION},
            {"role": "user", "content": transcript},
        ],
        response_format={"type": "json_object"},
    )
    if not response.choices:
        return None

    parsed = chat_summary._parse(response.choices[0].message.content or "")
    if parsed is None or not any(parsed.values()):
        return None
    return parsed


async def evaluate(model: str) -> dict:
    history: list[Turn] = []
    turns = []

    for turn in TURNS:
        try:
            answer, ttft = await answer_of(model, turn, history)
        except Exception as error:
            return {"model": model, "error": f"답변 호출 실패 — {str(error)[:150]}"}

        try:
            saved = await summary_of(model, turn, history, answer)
            summary_error = None
        except Exception as error:
            saved, summary_error = None, str(error)[:110]

        turns.append({"answer": answer, "ttft": ttft,
                      "summary": saved, "summary_error": summary_error})
        history = history + [
            Turn(role="user", content=turn["message"]),
            Turn(role="assistant", content=answer),
        ]

    return {"model": model, "turns": turns,
            "param": _TOKEN_PARAM.get(model, "?")}


def report(results: list[dict]) -> None:
    for r in results:
        print("=" * 74)
        print(f"■ {r['model']}")
        if "error" in r:
            print(f"  ❌ {r['error']}")
            continue

        param = r["param"]
        note = "운영 코드와 동일" if param == "max_tokens" \
            else "⚠️ 운영 코드는 max_tokens 라 이 모델로 바꾸면 챗봇이 502 로 죽습니다"
        print(f"  토큰 파라미터  {param} — {note}")

        for turn, result in zip(TURNS, r["turns"]):
            answer = result["answer"]
            over = "" if len(answer) <= 120 else " ❌ 초과"
            saved = result["summary"]
            if saved:
                summary = "✅ " + json.dumps(saved, ensure_ascii=False)
            elif result["summary_error"]:
                summary = f"❌ 호출 실패 — {result['summary_error']}"
            else:
                summary = "❌ 저장 안 됨 (형식 불일치 또는 전부 빈 값)"

            print()
            print(f"  {turn['label']}")
            print(f"    답변      {answer or '(빈 응답)'}")
            print(f"    길이      {len(answer)}자{over}")
            print(f"    첫 토큰   {result['ttft'] * 1000:.0f}ms")
            print(f"    가슴 수치 {'✅ 있음' if CHEST_NUMBER.search(answer) else '❌ 없음'}")
            print(f"    요약      {summary}")

        first, second = r["turns"]
        same = first["answer"].strip() == second["answer"].strip()
        print()
        print(f"  옷 바뀔 때 {'❌ 답변이 동일합니다' if same else '✅ 답변이 갈립니다'}")
    print("=" * 74)


def table(results: list[dict]) -> None:
    print()
    print("요약표")
    print(f"  {'모델':22} {'가슴':4} {'갈림':4} {'120자':5} {'요약':4} {'첫토큰':>7}  파라미터")
    for r in results:
        if "error" in r:
            print(f"  {r['model']:22} {'—':4} {'—':4} {'—':5} {'—':4} {'—':>7}  실행 실패")
            continue
        turns = r["turns"]
        chest = all(CHEST_NUMBER.search(t["answer"]) for t in turns)
        differ = turns[0]["answer"].strip() != turns[1]["answer"].strip()
        within = all(len(t["answer"]) <= 120 for t in turns)
        saved = all(t["summary"] for t in turns)
        ttft = max(t["ttft"] for t in turns) * 1000
        mark = lambda ok: " ✅ " if ok else " ❌ "
        print(f"  {r['model']:22}{mark(chest)}{mark(differ)}{mark(within)}  "
              f"{mark(saved)}{ttft:6.0f}ms  {r['param']}")


async def main() -> None:
    if not settings.openai_enabled:
        sys.exit("OPENAI_API_KEY 가 없습니다. /home/ubuntu/14th-ISIX-ai/.env 를 확인하세요.")

    models = sys.argv[1:] or DEFAULT_MODELS

    # 프롬프트 파일이 없으면 여기서 멈춥니다. 모델 탓으로 오해하지 않도록.
    prompt = chat_prompt.build_system_prompt(build(TURNS[0], []))
    print(f"프롬프트 {len(prompt)}자 · 후보 {len(models)}개")
    print("운영 .env 와 서비스는 건드리지 않습니다.\n")

    results = []
    for model in models:
        print(f"… {model}")
        results.append(await evaluate(model))
    print()
    report(results)
    table(results)


if __name__ == "__main__":
    asyncio.run(main())
