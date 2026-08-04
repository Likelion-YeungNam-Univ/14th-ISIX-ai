<div align="center">

<img src="https://raw.githubusercontent.com/Likelion-YeungNam-Univ/14th-ISIX-ai/main/docs/banner-closr-white.svg" width="100%" />

</div>

<br />

## 👥 팀원 소개

<div align="center">

<table>
  <tr>
    <td align="center" width="120"><a href="https://github.com/knayoung0"><img src="https://github.com/knayoung0.png" width="100"/><br/>구나영</a></td>
    <td align="center" width="120"><a href="https://github.com/copepb"><img src="https://github.com/copepb.png" width="100"/><br/>김민호</a></td>
    <td align="center" width="120"><a href="https://github.com/ryudayeong"><img src="https://github.com/ryudayeong.png" width="100"/><br/>류다영</a></td>
    <td align="center" width="120"><a href="https://github.com/hyeonseo-sung"><img src="https://github.com/hyeonseo-sung.png" width="100"/><br/>성현서</a></td>
    <td align="center" width="120"><a href="https://github.com/ckrhkdwls"><img src="https://github.com/ckrhkdwls.png" width="100"/><br/>차광진</a></td>
    <td align="center" width="120"><a href="https://github.com/user070917"><img src="https://github.com/user070917.png" width="100"/><br/>황연준</a></td>
  </tr>
</table>

</div>

## 🎯 프로젝트 소개

> **내 몸 위의 3D 가상 아틀리에**

한 장의 사진으로 3D 아바타를 생성하고, 물리 시뮬레이션 기반으로 의류 사이즈를 추천하는 가상 피팅 플랫폼입니다.

기존 가상 피팅은 대부분 2D 이미지 합성으로, 옷과 몸의 공간 관계를 계산하지 않습니다.
**CLOSER는 합성이 아니라 물리 연산을 합니다.**

<br />

## ✨ 핵심 기능

| | 기능 | 내용 |
|:---:|---|---|
| **①** | **신체 비율 판단** | 전신 사진 1장 + 키·몸무게로 3D 아바타 생성, 12개 부위 치수 자동 계측 |
| **②** | **체형 유형 진단** | 역삼각형·직사각형·모래시계형 등 체형 타입 판정 |
| **③** | **가상 피팅** | 아바타에 의류 착용 · 360° 회전 · 핏 히트맵 · 사이즈 추천 |
| **④** | **피팅 저장** | 아바타와 피팅 결과를 계정에 저장, 재방문 시 복원 |

<br />

## 🏗 AI 아키텍처

**[사전 계산 · GPU]** 표준 마네킹 → 패턴 18개 → 체형 12구간 드레이핑 → GLB 216개

**[실시간 · CPU]** 사진 → β → 3D 메시 → 치수 12개 → 최근접 GLB 조회

의류 시뮬레이션은 1벌당 30초~2분 소요 → 런타임 실행 불가.
의류 6종 × 사이즈 3 × 체형 12구간 = **216개 사전 계산**, 실시간엔 최근접 결과 조회.

<br />

## 🚀 로컬 실행

```bash
# 1. 레포 클론
git clone https://github.com/Likelion-YeungNam-Univ/14th-ISIX-ai.git
git clone https://github.com/Likelion-YeungNam-Univ/14th-ISIX-was.git
git clone https://github.com/Likelion-YeungNam-Univ/14th-ISIX-web.git

# 2. AI 서버 (Python 3.9+)
cd 14th-ISIX-ai
python -m venv venv && source venv/bin/activate
pip install -c constraints.txt -r requirements.txt   # -c 필수
cp .env.example .env
uvicorn app.main:app --reload --port 8000

# 3. BE 서버 (Java 17+, Gradle)
cd 14th-ISIX-was && ./gradlew bootRun

# 4. FE
cd 14th-ISIX-web && npm install && npm run dev
```

> **AI 서버 두 가지 주의**
> - `pip install` 에 `-c constraints.txt` 를 빼면 numpy가 2.x로 올라가 mediapipe가 동작하지 않습니다.
> - SMPL-X 모델은 재배포 금지 라이선스라 저장소에 없습니다.
>   [직접 다운로드](https://smpl-x.is.tue.mpg.de) 후 `assets/models/smplx/SMPLX_FEMALE.npz` 에 배치하세요.

<br />

---

## 🤖 이 저장소 — AI 서버 (FastAPI)

사진 1장으로 3D 아바타를 생성하고 의류 사이즈를 추천하는 AI 서버입니다.

멋쟁이사자처럼 해커톤 ISIX 2026

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

## 실행 방법

### 1. 가상환경

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. 의존성 설치

**반드시 `-c constraints.txt`를 지정합니다.**

```bash
pip install -c constraints.txt -r requirements.txt
```

`-c` 없이 설치하면 numpy가 2.x로 올라가면서 mediapipe가 동작하지 않습니다.

확인:

```bash
python -c "import mediapipe, cv2, numpy; print(numpy.__version__)"
# 1.26.4 가 나와야 정상입니다
```

### 3. 환경변수

```bash
cp .env.example .env
```

### 4. SMPL-X 모델 배치

저장소에 포함되지 않으므로 각자 다운로드합니다.

1. https://smpl-x.is.tue.mpg.de 계정 생성
2. 라이선스 동의 후 SMPL-X v1.1 (NPZ+PKL) 다운로드
3. `assets/models/smplx/SMPLX_FEMALE.npz` 에 배치

**재배포 금지 라이선스입니다.**
저장소가 Public이므로 커밋되면 즉시 위반입니다. `.gitignore`에 이미 등록되어 있습니다.

### 5. 서버 실행

```bash
uvicorn app.main:app --reload --port 8000
```

| | |
|---|---|
| API 문서 | http://localhost:8000/docs |
| 헬스체크 | http://localhost:8000/health |

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
      "glb_url": "https://cdn.closer.xxx/avatars/av_7f3a9c21b4d0.glb",
      "body_bucket": "H1B1",
      "measurements": { "shoulder_width": 40.1, "chest_circ": 87.2 },
      "confidence": 0.771,
      "warnings": []
    },
    "error_message": null
  },
  "error": null
}
```

`status` 는 `processing` · `done` · `failed` 중 하나입니다.
`done` 일 때만 `result` 가 채워지고, `failed` 면 `error_message` 에 사유가 담깁니다.

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

## 라이선스

이 저장소의 **코드는 MIT License** 입니다. `LICENSE` 참고.

단, 아래는 이 저장소의 라이선스가 적용되지 않습니다.

| 대상 | 라이선스 | 비고 |
|---|---|---|
| SMPL-X 모델 (`*.npz`, `*.pkl`) | **비상업 학술 라이선스** | 저장소에 포함하지 않음. [직접 다운로드](https://smpl-x.is.tue.mpg.de) 필요 |
| SMPL-X에서 생성한 메시 (`assets/grid/*.glb`, `assets/draped/*.glb`) | 상동 (파생물) | 저장소에 포함하지 않음 |

MIT는 **우리가 작성한 코드에만** 적용됩니다.
SMPL-X 모델과 그로부터 생성한 3D 메시는 재배포가 금지되어 있어,
`.gitignore`로 차단하고 각자 내려받아 배치하는 방식을 씁니다.

상업적 이용을 검토할 경우 SMPL-X 라이선스를 먼저 확인해야 합니다.

<br />

<div align="center">

<img src="https://raw.githubusercontent.com/Likelion-YeungNam-Univ/14th-ISIX-ai/main/docs/footer.svg" width="100%" />

</div>
