# CLOSR AI — 개발 문서

README 는 프로젝트 소개용입니다. 개발에 필요한 내용은 여기 모읍니다.

**부위 이름 12개와 파일명 규칙은 백엔드·의류 파트가 그대로 사용하는 약속입니다.**
임의로 바꾸면 에러 없이 조용히 어긋납니다.

---

## 이 서버가 하는 일

```
전신 사진 + 키 · 몸무게
      ↓
관절 · 실루엣 추출 (MediaPipe)
      ↓
키 대비 비율 정규화
      ↓
SMPL-X 체형 파라미터 최적화 (scipy)
      ↓
3D 메시 생성 → 키로 스케일 보정
      ↓
12부위 치수 계측 (단면 볼록껍질)
      ↓
GLB + measurements
```

**사진 픽셀에서 직접 cm를 계산하지 않습니다.**
2D 사진은 몸의 두께를 알 수 없어 둘레를 구할 수 없습니다.
사진에서는 비율만 추출하고, 3D 메시를 만든 뒤 그 위에서 측정합니다.
이것이 SMPL-X를 거치는 이유입니다.

---

## 왜 별도 서버인가

백엔드가 Spring Boot(Java)라 Python 파이프라인을 직접 호출할 수 없습니다.
HTTP로 통신하는 별도 서버로 분리했습니다.

```
[React]  ──→  [Spring Boot]  ──HTTP──→  [FastAPI · AI]
```

---

## 폴더 구조

```
14th-ISIX-ai/
├─ app/
│  ├─ main.py              FastAPI 엔트리포인트
│  ├─ core/
│  │  ├─ config.py         환경변수 · 설정
│  │  ├─ response.py       공통 응답 래퍼
│  │  └─ exceptions.py     오류 코드 · 전역 핸들러
│  ├─ routers/
│  │  ├─ avatar.py         POST /api/avatar/generate, GET /api/avatar/{id}
│  │  └─ fitting.py        GET  /api/fitting/{avatar_id}
│  ├─ services/            파이프라인 로직 (step1~6)
│  ├─ models/              Pydantic 스키마
│  └─ utils/
├─ assets/                 커밋 제외
│  ├─ models/smplx/        SMPL-X 모델
│  ├─ grid/                body_grid.json
│  └─ draped/              사전 계산 GLB 216개
├─ requirements.txt
└─ constraints.txt         버전 고정
```

---

## API

### 아바타 생성 — 비동기(폴링)

사진 1장 처리에 **10~20초**가 걸립니다. 대부분 SMPL-X beta 최적화이고
CPU 전용이라 더 줄이기 어렵습니다. 그래서 작업을 등록하고 폴링합니다.

**1) 작업 등록**

```
POST /api/avatar/generate
Content-Type: multipart/form-data
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `photo` | file | 전신 사진 (JPEG/PNG, 최대 10MB) |
| `height` | int | 키 (cm), 130~220 |
| `weight` | int | 몸무게 (kg), 30~150 |

```json
202 Accepted
{
  "success": true,
  "data": {
    "avatar_id": "av_7f3a9c21b4d0",
    "status": "processing",
    "poll_after_ms": 3000
  },
  "error": null
}
```

**2) 상태 조회**

```
GET /api/avatar/{avatar_id}
```

`status` 가 `done` 이 될 때까지 `poll_after_ms` 간격으로 호출합니다.

```json
{
  "success": true,
  "data": {
    "avatar_id": "av_7f3a9c21b4d0",
    "status": "done",
    "result": {
      "glb_url": "https://cdn.closr.xxx/avatars/av_7f3a9c21b4d0.glb",
      "body_bucket": "H1B1",
      "measurements": { "shoulder_width": 40.1, "chest_circ": 87.2 },
      "confidence": 0.771,
      "body_type": "hourglass",
      "body_type_label": "모래시계형",
      "body_type_message": "가슴과 엉덩이가 비슷하고 허리가 뚜렷합니다. 허리선이 있는 옷이 잘 맞습니다.",
      "warnings": []
    },
    "error_message": null
  },
  "error": null
}
```

`status` 는 `processing` · `done` · `failed` 중 하나입니다.
`done` 일 때만 `result` 가 채워지고, `failed` 면 `error_message` 에 사유가 담깁니다.

#### 체형 유형

`body_type` 은 `hourglass` · `triangle` · `inverted_triangle` · `rectangle` · `round`
다섯 가지입니다. 가슴·허리·엉덩이 **둘레** 세 개로 판정하며, 하나라도 계측에
실패하면 세 필드가 함께 `null` 입니다. 프론트는 `null` 분기를 처리해야 합니다.

`body_bucket` 과 혼동하지 마세요. 그쪽은 사전 계산 GLB 를 찾기 위한 내부 격자
키(`H{0-2}B{0-3}`)이고, `body_type` 은 사용자에게 보여주는 진단 결과입니다.

어깨(`shoulder_width`)는 판정에 쓰지 않습니다. 너비라서 둘레와 같은 축에서
비교할 수 없고, 팔이 몸통에 붙은 사진에서 과대 추정됩니다. 키 대비 비율로
`body_type_message` 뒤에 보조 문구만 덧붙입니다.

### 가상 피팅

```
GET /api/fitting/{avatar_id}?garment_id=shirt&size=M
```

사전 계산된 드레이핑 결과를 조회합니다.

---

## 부위 이름 12개 — 고정

백엔드·프론트가 이 문자열을 그대로 사용합니다. 임의로 바꾸지 않습니다.

```
shoulder_width   chest_circ      waist_circ      hip_circ
neck_circ        arm_circ        thigh_circ      back_length
sleeve_length    inseam          total_length    front_width
```

---

## 파일명 규칙

파트 간 통신의 기준입니다.

```
아바타 메시      body_{bucket}.glb                body_H1B2.glb
착용 메시        {garment}_{size}__{bucket}.glb   shirt_M__H1B2.glb
여유량           {garment}_{size}__{bucket}.json  shirt_M__H1B2.json
체형 격자        body_grid.json
```

`bucket` = `H{0-2}B{0-3}` — 키 3단계 × 체격 4단계 = 12구간

**`body_grid.json`은 AI 파트가 정의하고 배포합니다.**
백엔드가 하드코딩하지 않고 이 파일을 읽어 사용합니다.
값이 어긋나면 엉뚱한 GLB를 서빙하게 되며 에러가 나지 않습니다.

---

## 주의사항

**패키지 버전을 올리지 않습니다.**
numpy 2.x는 mediapipe와 호환되지 않습니다.

**SMPL-X 모델을 커밋하지 않습니다.**
재배포 금지 라이선스이며, 저장소가 Public입니다.
한 번 커밋하면 히스토리에 영구히 남고 `git rm`으로는 지워지지 않습니다.

**원본 사진을 저장하지 않습니다.**
전신 사진은 민감정보입니다. 메모리에서 처리 후 즉시 폐기합니다.

**검증 이미지를 반드시 확인합니다.**
3D 작업은 코드가 정상 동작하면서 결과만 틀린 경우가 많습니다.
수치가 정상 범위여도 눈으로 확인한 뒤 다음 단계로 넘어갑니다.

---

