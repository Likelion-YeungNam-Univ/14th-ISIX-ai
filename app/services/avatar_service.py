"""아바타 생성 서비스.

사진 1장 + 키 · 몸무게로 3D 아바타(GLB)와 12부위 치수를 만듭니다.

주의 — 사진 픽셀에서 직접 cm 를 계산하지 않습니다.
2D 사진은 몸의 두께를 알 수 없어 둘레를 구할 수 없습니다.
사진에서는 비율만 뽑고, 3D 메시를 만든 뒤 그 위에서 측정합니다.

이 모듈은 **동기 CPU 작업**입니다. 요청당 10~20초 동안 코어를 100% 씁니다.
FastAPI 의 async 함수 안에서 직접 부르면 이벤트 루프가 막히므로,
반드시 워커 스레드로 넘겨서 호출하세요 (라우터 참고).
"""

import json
import logging
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.exceptions import ClosrException, ErrorCode
from app.services import bodytype, storage
from app.services.pipeline import config as pcfg
from app.services.pipeline import step1_photo, step2_shape, step3_scale, step4_measure

logger = logging.getLogger(__name__)

# 파이프라인 동시 실행 제한.
# CPU 전용이라 여러 요청이 겹치면 서로 코어를 뺏어 전부 느려지고,
# 요청당 762MB 라 메모리도 못 버팁니다.
_SEMAPHORE = threading.Semaphore(settings.max_concurrent_pipeline)

# SMPL-X 모델은 로드에 수 초 + 291MB 가 듭니다. 프로세스당 1회만 만듭니다.
_body = None
_body_lock = threading.Lock()


def load_body():
    """SMPL-X 모델을 로드합니다. 서버 시작 시 1회 호출하세요."""
    global _body
    with _body_lock:
        if _body is None:
            logger.info("SMPL-X 모델 로드 중: %s", settings.smplx_model_path)
            _body = step2_shape.Body()
            logger.info("SMPL-X 모델 로드 완료 (gender=%s)", pcfg.GENDER)
    return _body


@dataclass
class AvatarOutput:
    glb_path: Path
    # 프론트가 그대로 3D 뷰어에 넣는 주소.
    # R2 가 설정돼 있으면 https 절대 주소, 아니면 AI 서버의 상대 경로다.
    glb_url: str
    body_bucket: str
    measurements: dict
    confidence: float
    # 체형 유형. 허리·가슴·엉덩이 중 하나라도 계측에 실패하면 None 이다.
    # body_bucket(격자 배정)과는 다른 값이다 — 격자는 사전 계산 GLB 를 찾기
    # 위한 내부 키이고, 이쪽은 사용자에게 보여주는 진단 결과다.
    body_type: Optional[str] = None
    body_type_label: Optional[str] = None
    body_type_message: Optional[str] = None
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------- 구간 배정

def assign_bucket(height_cm: float, chest_cm: float) -> str:
    """(키, 가슴둘레) -> 체형 12구간 코드.

    경계값을 코드에 박지 않고 body_grid.json 에서 읽습니다.
    백엔드·의류 파트도 같은 파일을 읽으므로, 한쪽만 하드코딩하면
    에러 없이 서로 다른 구간을 가리키게 됩니다.
    """
    grid = json.loads(settings.body_grid_path.read_text())
    a = grid["assignment"]
    h = sum(height_cm >= t for t in a["H_bounds_cm"])
    b = sum(chest_cm >= t for t in a["B_bounds_cm"])
    return f"H{h}B{b}"


# ---------------------------------------------------------------- 본체

def generate(photo_bytes: bytes, height_cm: int, weight_kg: int,
             avatar_id: Optional[str] = None) -> AvatarOutput:
    """사진 -> 아바타. 동기 함수입니다 (워커 스레드에서 호출할 것).

    단계
        1 사진 -> 관절 33개 + 실루엣 -> 키 대비 비율
        2 비율 -> SMPL-X beta (가중 최소제곱)
        3 입력 키로 메시 절대 크기 보정 -> GLB
        4 3D 메시 단면에서 12부위 계측
        5 body_grid.json 으로 체형 구간 배정
    """
    avatar_id = avatar_id or f"av_{uuid.uuid4().hex[:12]}"
    body = load_body()

    with _SEMAPHORE:
        # 원본 사진은 디스크에 남기지 않습니다. 처리 후 즉시 삭제됩니다.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp.write(photo_bytes)
            tmp.flush()

            # --- 1단계 ---
            try:
                seen = step1_photo.analyze(tmp.name, save_debug=False)
            except RuntimeError as e:
                logger.warning("[%s] 1단계 실패: %s", avatar_id, e)
                raise ClosrException(ErrorCode.NO_PERSON_DETECTED) from e

            # --- 2단계 ---
            est = step2_shape.fit(seen, height_cm, weight_kg,
                                  verbose=False, body=body, save_obj=False)

            # --- 3단계 ---
            mesh, joints, info, _ = step3_scale.build(est["beta"], height_cm, body=body)

            # --- 4단계 ---
            meas, _, extra, warns = step4_measure.measure(
                mesh, joints, height_cm, body)

    # --- 5단계: 구간 배정 ---
    chest = meas.get("chest_circ")
    bucket = assign_bucket(height_cm, chest) if chest else None

    # --- GLB 저장 ---
    settings.avatar_dir.mkdir(parents=True, exist_ok=True)
    glb_path = settings.avatar_dir / f"{avatar_id}.glb"
    mesh.export(str(glb_path))

    warnings = list(seen.get("warnings", [])) + list(warns)

    # --- R2 업로드 ---
    # 실패해도 아바타 생성 자체는 살린다. 10~20초 걸린 결과를 저장소 문제로
    # 통째로 버릴 이유가 없다. 대신 로컬 주소로 내려가는 사실을 경고에 남긴다.
    # 이걸 조용히 넘기면 운영에서 http 주소가 나가고, 프론트가 https 면
    # 브라우저가 막아서 "모델이 안 보인다" 로만 드러난다.
    glb_url = f"{settings.avatar_url_prefix}/{glb_path.name}"
    if storage.is_configured():
        try:
            glb_url = storage.upload_glb(glb_path, glb_path.name)
        except Exception as e:
            logger.error("[%s] R2 업로드 실패, 로컬 주소로 대체합니다: %s",
                         avatar_id, e)
            warnings.append(
                "GLB 를 저장소에 올리지 못해 임시 주소로 내려갑니다. "
                "https 페이지에서는 모델이 보이지 않을 수 있습니다.")

    if not extra.get("calibrated"):
        warnings.append("실측 보정 전 값입니다. 계통 오차가 남아 있습니다.")
    if bucket is None:
        warnings.append("가슴둘레를 측정하지 못해 체형 구간을 배정하지 못했습니다.")

    measured = {k: meas[k] for k in pcfg.MEASUREMENTS if meas.get(k) is not None}

    # 체형 유형 판정. 판정에 필요한 둘레가 없으면 None 이 오고, 그때는
    # 체형 정보 없이 아바타만 내려간다. 여기서 예외를 던지면 10~20초 걸린
    # 파이프라인 결과를 통째로 버리게 된다.
    # warnings 를 함께 넘긴다. 팔이 몸통에 붙은 사진이면 어깨 계측이 과대
    # 추정되므로 어깨 보조 문구를 내보내지 않는다.
    shape = bodytype.classify(measured, height_cm, warnings)

    logger.info("[%s] 완료 — 구간 %s, 체형 %s, confidence %.3f, 경고 %d건",
                avatar_id, bucket, (shape or {}).get("body_type"),
                est.get("confidence", 0), len(warnings))

    return AvatarOutput(
        glb_path=glb_path,
        glb_url=glb_url,
        body_bucket=bucket or "",
        measurements=measured,
        confidence=float(est.get("confidence", 0.0)),
        body_type=(shape or {}).get("body_type"),
        body_type_label=(shape or {}).get("body_type_label"),
        body_type_message=(shape or {}).get("body_type_message"),
        warnings=warnings,
    )
