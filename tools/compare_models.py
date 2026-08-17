"""챗봇 모델 후보 비교.

AI 서버에서 돌립니다. 키는 이 파일에 없고 .env 에서 읽습니다.

    cd /home/ubuntu/14th-ISIX-ai
    python3 compare_models.py

운영 코드 경로를 그대로 씁니다 — 프롬프트는 chat_prompt.build_system_prompt,
요약은 chat_summary.extract 입니다. 별도로 재현한 것이 아니라서, 여기서 통과하면
운영에서도 같게 동작합니다.

재는 것 다섯 가지입니다.

    첫 토큰 지연     음성이라 그대로 체감됩니다
    답변 길이        120자 이내여야 합니다
    가슴 수치 언급   profile 이 붙으면 판정 수치가 사라지던 증상
    1차 vs 2차       옷을 바꿨을 때 답변이 갈리는지
    요약 저장        실패하면 개인화가 조용히 죽습니다

마지막이 중요합니다. 요약 호출이 실패해도 답변은 정상이라 화면에서는
멀쩡해 보이고, 2차 대화에서 기억을 못 하는 형태로만 드러납니다.
"""

import asyncio
import json
import re
import sys
import time

from app.core.config import settings
from app.models.chat import ChatRequest, FitContext, Turn
from app.services import chat_prompt, chat_service, chat_summary

# 후보. 인자로 넘기면 그것을 씁니다.
#
#     python3 compare_models.py gpt-4o-mini gpt-5.4-mini gpt-5.6-sol
#
# 모델 id 는 대시보드에서 그대로 복사해 오세요(Settings → Limits → Allow or
# block models). 이름이 틀리면 그 줄만 실패로 표시되고 나머지는 계속 돕니다.
DEFAULT_MODELS = ["gpt-4o-mini"]

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


async def run_turn(turn: dict, history: list[Turn]) -> dict:
    """한 턴을 돌리고 지연 · 답변 · 요약을 돌려줍니다."""
    request = build(turn, history)

    started = time.monotonic()
    first, rest = await chat_service.open_stream(request)
    ttft = time.monotonic() - started

    events = [first]
    async for event in rest:
        events.append(event)

    answer = "".join(
        json.loads(line[len("data: "):]).get("delta", "")
        for line in "".join(events).splitlines()
        if line.startswith("data: ")
    )

    summary = await chat_summary.extract(
        history=[t.model_dump() for t in history],
        message=turn["message"],
        answer=answer,
    )
    return {"ttft": ttft, "answer": answer, "summary": summary}


async def probe_max_tokens(model: str) -> str:
    """max_tokens 를 받는지 확인합니다.

    신형 모델 일부는 max_completion_tokens 를 요구합니다. 우리 코드가
    max_tokens 를 쓰고 있어서, 안 받으면 요약 호출이 400 으로 떨어집니다.
    답변은 정상이라 화면에서는 안 보이고 개인화만 죽습니다.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=20)
    try:
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=5,
        )
        return "max_tokens 사용 가능"
    except Exception as error:
        text = str(error)
        if "max_completion_tokens" in text:
            return "⚠️ max_completion_tokens 필요 — 코드 수정 없이는 요약이 죽습니다"
        return f"⚠️ {text[:90]}"


async def evaluate(model: str) -> dict:
    settings.openai_model = model
    chat_service.settings = settings
    chat_summary.settings = settings

    compat = await probe_max_tokens(model)

    history: list[Turn] = []
    turns = []
    for turn in TURNS:
        try:
            result = await run_turn(turn, history)
        except Exception as error:
            return {"model": model, "compat": compat, "error": str(error)[:160]}
        turns.append(result)
        history = history + [
            Turn(role="user", content=turn["message"]),
            Turn(role="assistant", content=result["answer"]),
        ]
    return {"model": model, "compat": compat, "turns": turns}


def report(results: list[dict], models: list[str]) -> None:
    for r in results:
        print("=" * 72)
        print(f"■ {r['model']}")
        print(f"  호환성   {r['compat']}")
        if "error" in r:
            print(f"  ❌ 실행 실패: {r['error']}")
            continue

        first, second = r["turns"]
        for turn, result in zip(TURNS, r["turns"]):
            answer = result["answer"]
            print()
            print(f"  {turn['label']}")
            print(f"    답변      {answer}")
            print(f"    길이      {len(answer)}자 {'✅' if len(answer) <= 120 else '❌ 초과'}")
            print(f"    첫 토큰   {result['ttft'] * 1000:.0f}ms")
            print(f"    가슴 수치 {'✅ 있음' if CHEST_NUMBER.search(answer) else '❌ 없음'}")
            saved = result["summary"]
            print(f"    요약      {'✅ ' + json.dumps(saved, ensure_ascii=False) if saved else '❌ 저장 안 됨'}")

        same = first["answer"].strip() == second["answer"].strip()
        print()
        print(f"  옷 바뀔 때 {'❌ 답변이 동일합니다' if same else '✅ 답변이 갈립니다'}")
    print("=" * 72)


async def main() -> None:
    if not settings.openai_enabled:
        sys.exit("OPENAI_API_KEY 가 없습니다. /home/ubuntu/14th-ISIX-ai/.env 를 확인하세요.")

    models = sys.argv[1:] or DEFAULT_MODELS

    # 프롬프트 파일이 없으면 여기서 바로 멈춥니다. 모델 탓으로 오해하지 않도록.
    prompt = chat_prompt.build_system_prompt(build(TURNS[0], []))
    print(f"프롬프트 {len(prompt)}자 · 후보 {len(models)}개\n")

    results = []
    for model in models:
        print(f"… {model} 실행 중")
        results.append(await evaluate(model))
    print()
    report(results, models)


if __name__ == "__main__":
    asyncio.run(main())
