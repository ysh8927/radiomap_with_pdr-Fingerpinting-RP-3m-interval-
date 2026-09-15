import cv2
import numpy as np
import math
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
import json
import os

# ============================================================
# 도면 ROI -> 미터 스케일(격자) 좌표 추출 도구
#
# 기능
# - (1) 캘리브레이션: 치수선 2점 클릭
# - (2) 원점: 이미지 좌측 상단 (0, 0) 고정
# - (3) +X 방향: 이미지 오른쪽 방향 고정
# - (4) ROI 폴리곤: 꼭짓점 클릭(오목 형태 OK), Enter로 완료
# - (5) ROI 내부 GRID_M 간격의 (X,Y)[m] 격자점 생성 후 CSV 저장
# - (6) 캘리브레이션 값 JSON 저장
# - (7) CSV 저장 직전 Y축 부호 반전(Y = -Y) 옵션 적용
#
# 조작
# - 휠: 줌(확대/축소)
# - 우클릭 드래그: 팬(이동)
# - 좌클릭: 점 추가
# - Ctrl+Z 또는 z: 마지막 점 취소(Undo)
# - Enter: 단계 완료
# - ESC: 종료
# ============================================================


# ============================================================
# 설정
# ============================================================
IMAGE_PATH = "data/공학관3층_동측_도면.png"      # 도면 이미지
KNOWN_LENGTH_M = 3.500                          # 치수선 실제 길이 [m]
GRID_M = 1.0                                    # 격자 간격 [m]

OUT_DIR = "temp_data"
OUT_CSV = os.path.join(OUT_DIR, "rp_pos.csv")
OUT_CALIB_JSON = os.path.join(OUT_DIR, "calibration.json")
OUT_ROI_PREVIEW = os.path.join(OUT_DIR, "roi_polygon_preview.png")
OUT_GRID_PREVIEW = os.path.join(OUT_DIR, "roi_grid_points_preview.png")

os.makedirs(OUT_DIR, exist_ok=True)

# 줌 설정
SCALE_STEP = 0.10
MIN_SCALE = 0.2
MAX_SCALE = 10.0

# 한글 폰트 설정(Windows)
FONT_PATH_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\malgunsl.ttf",
    r"C:\Windows\Fonts\NanumGothic.ttf",
]
FONT_SIZE = 20

# Y축 부호 반전 옵션
# True:
#   원본 이미지 좌표계에서는 y+가 아래쪽
#   저장되는 rp_pos.csv에서는 y_m = -y_m 적용
FLIP_Y_BEFORE_SAVE = True


# ============================================================
# 한글 경로 안전 로드
# ============================================================
def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


img = imread_unicode(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {IMAGE_PATH}")


# ============================================================
# PIL 폰트 로드
# ============================================================
def load_korean_font():
    for p in FONT_PATH_CANDIDATES:
        try:
            return ImageFont.truetype(p, FONT_SIZE)
        except:
            pass
    return None


KOR_FONT = load_korean_font()


def draw_text_pil(bgr_img, x, y, text, font=None, color=(30, 30, 30)):
    """
    bgr_img: OpenCV BGR 이미지
    x, y: 좌상단
    text: 한글 OK
    color: BGR
    """
    if font is None:
        cv2.putText(
            bgr_img,
            text,
            (x, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )
        return bgr_img

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_im)
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(color[2], color[1], color[0])
    )

    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


# ============================================================
# 줌/팬/클릭/Undo 가능한 뷰어
# ============================================================
def collect_points_zoom_pan(
    window_name,
    base_img,
    instruction_lines,
    required_min_points,
    draw_closed_poly=False
):
    H, W = base_img.shape[:2]

    scale = 1.0
    pts = []

    offset_x, offset_y = 0, 0

    dragging = False
    drag_start = (0, 0)
    offset_start = (0, 0)

    win_w, win_h = 1400, 900

    def clamp_offsets():
        nonlocal offset_x, offset_y

        view_w = int(win_w / scale)
        view_h = int(win_h / scale)

        view_w = max(50, min(view_w, W))
        view_h = max(50, min(view_h, H))

        offset_x = int(np.clip(offset_x, 0, max(0, W - view_w)))
        offset_y = int(np.clip(offset_y, 0, max(0, H - view_h)))

        return view_w, view_h

    def get_disp():
        view_w, view_h = clamp_offsets()

        crop = base_img[
            offset_y:offset_y + view_h,
            offset_x:offset_x + view_w
        ].copy()

        disp = cv2.resize(
            crop,
            (win_w, win_h),
            interpolation=cv2.INTER_LINEAR
        )

        return disp, view_w, view_h

    def img_to_disp(px, py, view_w, view_h):
        sx = win_w / view_w
        sy = win_h / view_h

        dx = int(round((px - offset_x) * sx))
        dy = int(round((py - offset_y) * sy))

        return dx, dy

    def disp_to_img(x, y, view_w, view_h):
        sx = win_w / view_w
        sy = win_h / view_h

        px = int(round(offset_x + (x / sx)))
        py = int(round(offset_y + (y / sy)))

        return px, py

    def redraw():
        disp, view_w, view_h = get_disp()

        # 점 그리기
        for i, (x, y) in enumerate(pts, start=1):
            cx, cy = img_to_disp(x, y, view_w, view_h)

            if 0 <= cx < win_w and 0 <= cy < win_h:
                cv2.circle(disp, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    disp,
                    str(i),
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        # 선 그리기
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                p1 = img_to_disp(*pts[i], view_w, view_h)
                p2 = img_to_disp(*pts[i + 1], view_w, view_h)
                cv2.line(disp, p1, p2, (0, 255, 0), 2)

        # 닫힌 폴리곤 그리기
        if draw_closed_poly and len(pts) >= 3:
            p1 = img_to_disp(*pts[-1], view_w, view_h)
            p2 = img_to_disp(*pts[0], view_w, view_h)
            cv2.line(disp, p1, p2, (0, 255, 0), 2)

        # 안내문 박스
        overlay = disp.copy()
        pad = 12
        line_h = 28
        box_h = pad * 2 + line_h * len(instruction_lines)
        box_w = min(980, win_w - 20)

        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + box_w, 10 + box_h),
            (255, 255, 255),
            -1
        )

        disp = cv2.addWeighted(overlay, 0.65, disp, 0.35, 0)

        y0 = 10 + pad

        for line in instruction_lines:
            disp = draw_text_pil(
                disp,
                10 + pad,
                y0,
                line,
                font=KOR_FONT,
                color=(30, 30, 30)
            )
            y0 += line_h

        status = f"scale={scale:.2f}, offset=({offset_x},{offset_y}), points={len(pts)}"

        cv2.putText(
            disp,
            status,
            (12, win_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 30),
            2
        )

        cv2.imshow(window_name, disp)

    def on_mouse(event, x, y, flags, param):
        nonlocal scale
        nonlocal offset_x, offset_y
        nonlocal dragging, drag_start, offset_start

        # 마우스 휠 줌
        if event == cv2.EVENT_MOUSEWHEEL:
            view_w, view_h = clamp_offsets()
            img_before = disp_to_img(x, y, view_w, view_h)

            if flags > 0:
                scale = min(MAX_SCALE, scale + SCALE_STEP)
            else:
                scale = max(MIN_SCALE, scale - SCALE_STEP)

            view_w2, view_h2 = clamp_offsets()

            sx2 = win_w / view_w2
            sy2 = win_h / view_h2

            offset_x = int(round(img_before[0] - (x / sx2)))
            offset_y = int(round(img_before[1] - (y / sy2)))

            clamp_offsets()
            redraw()

        # 우클릭 팬 시작
        elif event == cv2.EVENT_RBUTTONDOWN:
            dragging = True
            drag_start = (x, y)
            offset_start = (offset_x, offset_y)

        # 팬 이동
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = x - drag_start[0]
            dy = y - drag_start[1]

            view_w, view_h = clamp_offsets()

            sx = win_w / view_w
            sy = win_h / view_h

            offset_x = int(round(offset_start[0] - dx / sx))
            offset_y = int(round(offset_start[1] - dy / sy))

            clamp_offsets()
            redraw()

        # 팬 종료
        elif event == cv2.EVENT_RBUTTONUP:
            dragging = False

        # 좌클릭 점 추가
        elif event == cv2.EVENT_LBUTTONDOWN:
            view_w, view_h = clamp_offsets()

            px, py = disp_to_img(x, y, view_w, view_h)

            px = int(np.clip(px, 0, W - 1))
            py = int(np.clip(py, 0, H - 1))

            pts.append((px, py))

            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, win_w, win_h)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except:
        pass

    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        # Enter
        if key == 13:
            if len(pts) < required_min_points:
                print(f"[경고] 최소 {required_min_points}개 점이 필요합니다. 현재: {len(pts)}")
            else:
                break

        # Undo: Ctrl+Z 또는 z/Z
        if key in [26, ord("z"), ord("Z")]:
            if pts:
                pts.pop()
                redraw()

        # ESC
        if key == 27:
            cv2.destroyWindow(window_name)
            raise SystemExit("사용자에 의해 종료됨(ESC).")

    cv2.destroyWindow(window_name)

    return pts


# ============================================================
# STEP 1) 캘리브레이션
# - 원점: 이미지 좌측 상단 (0, 0) 고정
# - +X 방향: 이미지 오른쪽 방향 고정
# - 사용자는 치수선 양 끝 2점만 클릭
# ============================================================
print("\n[STEP 1/2] 캘리브레이션 시작")

calib_instructions = [
    "휠: 확대/축소 | 우클릭드래그: 이동(Pan) | 좌클릭: 점 | Ctrl+Z(또는 z): 취소 | Enter: 완료 | ESC: 종료",
    "캘리브레이션 2점 찍기:",
    "1-2) 치수선 양 끝(실제 길이 = KNOWN_LENGTH_M)",
    "※ 원점은 이미지 좌측 상단 (0,0)으로 고정",
    "※ +X 방향은 이미지 오른쪽 방향으로 고정"
]

calib_pts = collect_points_zoom_pan(
    "CALIB (zoom+pan)",
    img,
    calib_instructions,
    2,
    False
)

print("[CALIB] 찍힌 점(픽셀):", calib_pts)

p1 = np.array(calib_pts[0], dtype=float)
p2 = np.array(calib_pts[1], dtype=float)

# 원점은 이미지 좌측 상단으로 고정
origin_px = np.array([0.0, 0.0], dtype=float)

d_px = np.linalg.norm(p2 - p1)
m_per_px = KNOWN_LENGTH_M / d_px
px_per_m = 1.0 / m_per_px

print(f"[스케일] d_px={d_px:.3f}px")
print(f"[스케일] m_per_px={m_per_px:.8f} m/px")
print(f"[스케일] px_per_m={px_per_m:.8f} px/m")

if not (1e-6 < m_per_px < 1.0):
    print("[경고] m_per_px 값이 비정상적으로 보입니다. 치수선 두 점을 정확히 찍었는지 확인하세요!")

# 회전 보정 없음
# 이미지 좌표계 그대로 사용:
# X+ : 오른쪽
# Y+ : 아래쪽
R = np.eye(2, dtype=float)


def px_to_meter(pt_xy_px):
    """
    pixel(x,y) -> meter(X,Y)

    기준:
    - 원점: 이미지 좌측 상단
    - X+ : 오른쪽
    - Y+ : 아래쪽
    """
    p = np.array(pt_xy_px, dtype=float)
    dp = p - origin_px

    X_m = dp[0] * m_per_px
    Y_m = dp[1] * m_per_px

    return float(X_m), float(Y_m)


def meter_to_px(pt_xy_m):
    """
    meter(X,Y) -> pixel(x,y)

    기준:
    - 원점: 이미지 좌측 상단
    - X+ : 오른쪽
    - Y+ : 아래쪽
    """
    xy_m = np.array(pt_xy_m, dtype=float)

    x_px = xy_m[0] / m_per_px
    y_px = xy_m[1] / m_per_px

    return float(x_px), float(y_px)


# ============================================================
# STEP 2) ROI 폴리곤
# ============================================================
print("\n[STEP 2/2] ROI 폴리곤 지정 시작")

roi_instructions = [
    "휠: 확대/축소 | 우클릭드래그: 이동(Pan) | 좌클릭: 꼭짓점 | Ctrl+Z(또는 z): 취소 | Enter: 완료 | ESC: 종료",
    "ROI 폴리곤 꼭짓점을 여러 번 찍고 Enter로 완료하세요 (최소 3점)"
]

roi_pts = collect_points_zoom_pan(
    "ROI (zoom+pan)",
    img,
    roi_instructions,
    3,
    True
)

print("[ROI] 찍힌 점(픽셀):", roi_pts)

roi_poly_px = np.array(roi_pts, dtype=np.int32)


# ============================================================
# ROI 미리보기 저장
# ============================================================
preview = img.copy()

cv2.polylines(
    preview,
    [roi_poly_px],
    True,
    (0, 255, 0),
    3
)

cv2.imwrite(OUT_ROI_PREVIEW, preview)

print(f"[저장] {OUT_ROI_PREVIEW}")


# ============================================================
# ROI 내부 격자 생성
# ============================================================
roi_poly_m = np.array(
    [px_to_meter(pt) for pt in roi_pts],
    dtype=float
)

Xs = roi_poly_m[:, 0]
Ys = roi_poly_m[:, 1]

x_min, x_max = Xs.min(), Xs.max()
y_min, y_max = Ys.min(), Ys.max()

x_vals = np.arange(
    np.floor(x_min / GRID_M) * GRID_M,
    np.ceil(x_max / GRID_M) * GRID_M + 1e-9,
    GRID_M
)

y_vals = np.arange(
    np.floor(y_min / GRID_M) * GRID_M,
    np.ceil(y_max / GRID_M) * GRID_M + 1e-9,
    GRID_M
)

grid_points = []

for X in x_vals:
    for Y in y_vals:
        x_px, y_px = meter_to_px((X, Y))

        inside = cv2.pointPolygonTest(
            roi_poly_px,
            (x_px, y_px),
            False
        )

        if inside >= 0:
            grid_points.append((X, Y))

grid_points = np.array(grid_points, dtype=float)

if grid_points.size == 0:
    print("[경고] ROI 내부에 격자점이 0개입니다.")
    print(" - ROI 폴리곤을 더 크게 찍었는지 확인")
    print(" - 캘리브레이션 스케일이 맞는지 확인")
    print(" - GRID_M 값이 너무 큰지 확인")
    raise RuntimeError("ROI 내부 격자점이 0개라서 CSV를 생성할 수 없습니다.")

print(f"[결과] ROI 내부 {GRID_M:.1f}m 격자 점 개수: {len(grid_points)}")


# ============================================================
# DataFrame 생성
# - x_m, y_m: 저장용 미터 좌표
# - pixel_x, pixel_y: 도면 이미지 기준 픽셀 좌표
# ============================================================
rows = []

for X, Y in grid_points:
    pixel_x, pixel_y = meter_to_px((X, Y))

    row = {
        "x_m": float(X),
        "y_m": float(Y),
        "pixel_x": float(pixel_x),
        "pixel_y": float(pixel_y),
    }

    rows.append(row)

df = pd.DataFrame(rows)

# CSV 저장 직전에 Y축 뒤집기
# 단, pixel_y는 이미지 좌표이므로 뒤집지 않음
if FLIP_Y_BEFORE_SAVE:
    df["y_m"] = -df["y_m"]
    print("[INFO] Y축 부호를 반전하여 저장합니다. (y_m = -y_m)")

df = df[
    [
        "x_m",
        "y_m",
        "pixel_x",
        "pixel_y",
    ]
]


# ============================================================
# calibration.json 저장
# ============================================================

# ROI meter 좌표도 저장용 좌표계 기준으로 하나 더 만들어둠
roi_poly_m_saved = roi_poly_m.copy()

if FLIP_Y_BEFORE_SAVE:
    roi_poly_m_saved[:, 1] = -roi_poly_m_saved[:, 1]

grid_points_saved = grid_points.copy()

if FLIP_Y_BEFORE_SAVE:
    grid_points_saved[:, 1] = -grid_points_saved[:, 1]

calibration_data = {
    "image": {
        "path": IMAGE_PATH,
        "width_px": int(img.shape[1]),
        "height_px": int(img.shape[0])
    },

    "calibration": {
        "known_length_m": float(KNOWN_LENGTH_M),
        "calibration_points_px": {
            "p1": {
                "x": float(p1[0]),
                "y": float(p1[1])
            },
            "p2": {
                "x": float(p2[0]),
                "y": float(p2[1])
            }
        },
        "calibration_length_px": float(d_px),
        "m_per_px": float(m_per_px),
        "px_per_m": float(px_per_m)
    },

    "coordinate_system": {
        "origin_px": {
            "x": float(origin_px[0]),
            "y": float(origin_px[1])
        },
        "origin_description": "image_top_left",
        "x_positive_direction": "right",
        "y_positive_direction_raw": "down",
        "flip_y_before_save": bool(FLIP_Y_BEFORE_SAVE),
        "saved_y_m_rule": "saved_y_m = -raw_y_m" if FLIP_Y_BEFORE_SAVE else "saved_y_m = raw_y_m",
        "rotation_matrix": R.tolist()
    },

    "grid": {
        "grid_m": float(GRID_M),
        "num_grid_points": int(len(grid_points)),
        "raw_bounds_m": {
            "x_min": float(grid_points[:, 0].min()),
            "x_max": float(grid_points[:, 0].max()),
            "y_min": float(grid_points[:, 1].min()),
            "y_max": float(grid_points[:, 1].max())
        },
        "saved_bounds_m": {
            "x_min": float(grid_points_saved[:, 0].min()),
            "x_max": float(grid_points_saved[:, 0].max()),
            "y_min": float(grid_points_saved[:, 1].min()),
            "y_max": float(grid_points_saved[:, 1].max())
        }
    },

    "roi": {
        "roi_points_px": [
            {
                "x": float(pt[0]),
                "y": float(pt[1])
            }
            for pt in roi_pts
        ],
        "roi_points_m_raw": [
            {
                "x_m": float(pt[0]),
                "y_m": float(pt[1])
            }
            for pt in roi_poly_m
        ],
        "roi_points_m_saved": [
            {
                "x_m": float(pt[0]),
                "y_m": float(pt[1])
            }
            for pt in roi_poly_m_saved
        ]
    },

    "output_files": {
        "rp_pos_csv": OUT_CSV,
        "calibration_json": OUT_CALIB_JSON,
        "roi_polygon_preview": OUT_ROI_PREVIEW,
        "roi_grid_points_preview": OUT_GRID_PREVIEW
    }
}

with open(OUT_CALIB_JSON, "w", encoding="utf-8") as f:
    json.dump(
        calibration_data,
        f,
        ensure_ascii=False,
        indent=4
    )

print(f"[저장] {OUT_CALIB_JSON}")


# ============================================================
# CSV 저장
# ============================================================
df.to_csv(
    OUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(f"[저장] {OUT_CSV}")
print(df.head())


# ============================================================
# 결과 시각화 저장
# ============================================================
vis = img.copy()

for X, Y in grid_points:
    x_px, y_px = meter_to_px((X, Y))

    cv2.circle(
        vis,
        (int(round(x_px)), int(round(y_px))),
        2,
        (255, 0, 0),
        -1
    )

cv2.polylines(
    vis,
    [roi_poly_px],
    True,
    (0, 255, 0),
    2
)

cv2.imwrite(OUT_GRID_PREVIEW, vis)

print(f"[저장] {OUT_GRID_PREVIEW}")


# ============================================================
# 저장 결과 요약
# ============================================================
print("\n========== 저장 완료 ==========")
print(f"CSV              : {OUT_CSV}")
print(f"Calibration JSON : {OUT_CALIB_JSON}")
print(f"ROI Preview      : {OUT_ROI_PREVIEW}")
print(f"Grid Preview     : {OUT_GRID_PREVIEW}")
print("================================")