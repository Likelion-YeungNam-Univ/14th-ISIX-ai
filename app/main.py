"""CLOSER AI 서버.

백엔드(Spring Boot)가 HTTP로 호출하는 별도 서버입니다.
Java 에서 Python 파이프라인을 직접 호출할 수 없어 분리했습니다.

    [Spring Boot]  ──HTTP──→  [FastAPI · AI]
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exceptions import (
    CloserException,
    closer_exception_handler,
    unhandled_exception_handler,
)
from app.core.response import ApiResponse
from app.routers import avatar, fitting

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델을 1회만 로드합니다.

    요청마다 SMPL-X(약 830MB)를 다시 읽으면 응답이 수 배 느려집니다.
    """
    logger.info("CLOSER AI 서버를 시작합니다")

    if not settings.smplx_model_path.exists():
        logger.warning(
            "SMPL-X 모델을 찾을 수 없습니다: %s\n"
            "  https://smpl-x.is.tue.mpg.de 에서 다운로드 후 배치하세요.\n"
            "  재배포 금지 라이선스이므로 저장소에 커밋하지 않습니다.",
            settings.smplx_model_path,
        )

    if not settings.measure_calibration_path.exists():
        logger.warning(
            "계측 보정 파일이 없습니다. 보정 미적용 상태로 동작합니다: %s",
            settings.measure_calibration_path,
        )

    # TODO: app.state.body = Body()  — SMPL-X 모델 로드
    yield
    logger.info("CLOSER AI 서버를 종료합니다")


app = FastAPI(
    title="CLOSER AI",
    description="사진 1장으로 3D 아바타를 생성하고 의류 사이즈를 추천합니다",
    version="0.1.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(CloserException, closer_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(avatar.router)
app.include_router(fitting.router)


@app.get("/health", tags=["health"])
async def health() -> ApiResponse:
    return ApiResponse.ok({"status": "ok"})
