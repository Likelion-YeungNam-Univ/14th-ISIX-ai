"""오류 코드 및 예외 정의.

오류 메시지는 원인과 개선 방법을 함께 제공합니다.
"어떤 사진이 문제인지" 알려주지 않으면 사용자가 다시 시도할 수 없습니다.
"""

from enum import Enum

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.response import ApiResponse


class ErrorCode(Enum):
    # 아바타 생성
    NO_PERSON_DETECTED = (422, "사진에서 사람을 찾지 못했습니다. 전신이 모두 나오도록 다시 촬영해주세요")
    POSE_NOT_FRONTAL = (422, "정면 자세가 아닙니다. 카메라를 정면으로 바라봐주세요")
    BODY_TRUNCATED = (422, "전신이 나오지 않았습니다. 머리끝부터 발끝까지 나오게 촬영해주세요")
    LOW_CONFIDENCE = (422, "인식 정확도가 낮습니다. 몸선이 드러나는 옷으로 다시 촬영해주세요")
    MULTIPLE_PERSONS = (422, "여러 명이 검출되었습니다. 한 명만 나오도록 촬영해주세요")

    # 입력
    FILE_TOO_LARGE = (413, "파일 크기가 너무 큽니다")
    UNSUPPORTED_FORMAT = (400, "JPEG 또는 PNG 파일만 지원합니다")
    INVALID_INPUT = (400, "입력값이 올바르지 않습니다")

    # 조회
    AVATAR_NOT_FOUND = (404, "아바타를 찾을 수 없습니다")
    # 아바타는 있는데 아직 쓸 수 없는 상태입니다. 백엔드 AVATAR_NOT_READY 와 같은 뜻입니다.
    AVATAR_NOT_READY = (409, "아바타 생성이 아직 끝나지 않았습니다. 잠시 후 다시 시도해주세요")
    FITTING_NOT_AVAILABLE = (404, "해당 조합은 준비 중입니다. 다른 사이즈를 선택해주세요")

    # 챗봇
    # mode=fitting 인데 fit_context 가 없는 경우입니다. 치수 없이 사이즈를
    # 답하게 하면 없는 수치를 지어냅니다.
    CHAT_CONTEXT_REQUIRED = (400, "치수 정보가 없어 사이즈를 안내할 수 없습니다")
    # 스트림을 열기 전에 실패한 경우입니다. 연 뒤에 끊기면 상태 코드를 바꿀 수
    # 없어 스트림 안의 error 이벤트로 내려갑니다. 코드 이름은 같습니다.
    CHAT_UPSTREAM_ERROR = (502, "챗봇 서버 응답에 실패했습니다")

    # 서버
    INTERNAL_ERROR = (500, "처리 중 오류가 발생했습니다")

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message


class ClosrException(Exception):
    def __init__(self, error_code: ErrorCode) -> None:
        self.error_code = error_code
        super().__init__(error_code.message)


async def closr_exception_handler(_: Request, exc: ClosrException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.error_code.status_code,
        content=ApiResponse.fail(
            code=exc.error_code.name,
            message=exc.error_code.message,
        ).model_dump(),
    )


async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=ApiResponse.fail(
            code=ErrorCode.INTERNAL_ERROR.name,
            message=ErrorCode.INTERNAL_ERROR.message,
        ).model_dump(),
    )
