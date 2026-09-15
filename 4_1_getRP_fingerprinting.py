# 도면을 불러와 Spot 위치를 획득하는 코드 

import cv2
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 사용자 설정
# ============================================================

# 1. 도면 이미지 파일 이름
FLOOR_MAP_IMAGE_NAME = "data/공학관3층_동측_도면.png"

# 2. 도면에서 알고 있는 실제 길이 [m]
KNOWN_LENGTH_M = 3.500

# 3. 저장 파일 이름
SAVE_NAME = "rp_pos"

# 4. 클릭한 점 2개 사이에 생성할 포인트 간격 [m]
INTERPOLATION_INTERVAL_M = 1.0

# 5. 끝점과 마지막 RP 사이 최소 거리 [m]
# 마지막 1m 보간점과 끝점이 이 값보다 가까우면 끝점은 저장하지 않음
MIN_END_POINT_DISTANCE_M = 0.5

# 결과 파일
OUTPUT_CSV_PATH = f"temp_data/{SAVE_NAME}.csv"
OUTPUT_PREVIEW_PATH = f"temp_data/{SAVE_NAME}_preview.png"
OUTPUT_CALIBRATION_JSON_PATH = f"temp_data/{SAVE_NAME}_calibration.json"

# 화면 크기
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900

# True이면 이미지 위쪽을 +Y 방향으로 사용
# True  : 이미지 아래쪽으로 갈수록 y_m 감소
# False : 이미지 아래쪽으로 갈수록 y_m 증가
ENU_Y_AXIS = True


# ============================================================
# Preview 표시 설정
# ============================================================

RP_POINT_RADIUS = 10
RP_POINT_OUTLINE_RADIUS = 14
RP_POINT_TEXT_SCALE = 0.7
RP_POINT_TEXT_THICKNESS = 2

CLICKED_POINT_RADIUS = 8


# ============================================================
# 유니코드 경로 이미지 로드 / 저장
# ============================================================

def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(path: str, img):
    ext = Path(path).suffix
    result, encoded = cv2.imencode(ext, img)

    if result:
        encoded.tofile(path)
    else:
        raise RuntimeError(f"이미지 저장 실패: {path}")


# ============================================================
# 줌 / 팬 / 클릭 수집 함수
# ============================================================

def collect_points_zoom_pan(
    window_name,
    base_img,
    instruction_lines,
    required_points=None,
    min_points=1,
    win_w=1400,
    win_h=900,
):
    """
    required_points:
        None이면 Enter 누를 때까지 점을 계속 찍음
        숫자이면 해당 개수만큼 찍고 Enter 종료

    조작:
        좌클릭: 점 추가
        z: 마지막 점 취소
        마우스 휠: 줌
        우클릭 드래그: 팬
        Enter: 완료
        ESC: 종료
    """

    H, W = base_img.shape[:2]

    scale = 1.0
    min_scale = 0.2
    max_scale = 10.0
    scale_step = 0.1

    pts = []

    offset_x = 0
    offset_y = 0

    dragging = False
    drag_start = (0, 0)
    offset_start = (0, 0)

    def clamp_offsets():
        nonlocal offset_x, offset_y

        view_w = int(win_w / scale)
        view_h = int(win_h / scale)

        view_w = max(50, min(view_w, W))
        view_h = max(50, min(view_h, H))

        offset_x = int(np.clip(offset_x, 0, max(0, W - view_w)))
        offset_y = int(np.clip(offset_y, 0, max(0, H - view_h)))

        return view_w, view_h

    def get_display_image():
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

        x = int(round((px - offset_x) * sx))
        y = int(round((py - offset_y) * sy))

        return x, y

    def disp_to_img(x, y, view_w, view_h):
        sx = win_w / view_w
        sy = win_h / view_h

        px = int(round(offset_x + x / sx))
        py = int(round(offset_y + y / sy))

        return px, py

    def redraw():
        disp, view_w, view_h = get_display_image()

        # 찍은 점 표시
        for i, (px, py) in enumerate(pts, start=1):
            dx, dy = img_to_disp(px, py, view_w, view_h)

            if 0 <= dx < win_w and 0 <= dy < win_h:
                cv2.circle(disp, (dx, dy), 6, (0, 0, 255), -1)
                cv2.putText(
                    disp,
                    str(i),
                    (dx + 10, dy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        # 찍은 점을 2개씩 pair로만 선 연결
        if len(pts) >= 2:
            usable_len = len(pts)

            if usable_len % 2 != 0:
                usable_len -= 1

            for i in range(0, usable_len, 2):
                p1 = img_to_disp(*pts[i], view_w, view_h)
                p2 = img_to_disp(*pts[i + 1], view_w, view_h)
                cv2.line(disp, p1, p2, (0, 255, 0), 2)

        # 안내문 박스
        overlay = disp.copy()
        box_h = 30 + 30 * len(instruction_lines)
        cv2.rectangle(
            overlay,
            (10, 10),
            (1350, 10 + box_h),
            (255, 255, 255),
            -1,
        )
        disp = cv2.addWeighted(overlay, 0.65, disp, 0.35, 0)

        y = 40
        for line in instruction_lines:
            cv2.putText(
                disp,
                line,
                (25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 30, 30),
                2,
            )
            y += 30

        status = f"scale={scale:.2f}, offset=({offset_x},{offset_y}), points={len(pts)}"
        cv2.putText(
            disp,
            status,
            (15, win_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 30),
            2,
        )

        cv2.imshow(window_name, disp)

    def on_mouse(event, x, y, flags, param):
        nonlocal scale, offset_x, offset_y
        nonlocal dragging, drag_start, offset_start

        # 마우스 휠 줌
        if event == cv2.EVENT_MOUSEWHEEL:
            view_w, view_h = clamp_offsets()
            img_before = disp_to_img(x, y, view_w, view_h)

            if flags > 0:
                scale = min(max_scale, scale + scale_step)
            else:
                scale = max(min_scale, scale - scale_step)

            view_w2, view_h2 = clamp_offsets()
            sx2 = win_w / view_w2
            sy2 = win_h / view_h2

            offset_x = int(round(img_before[0] - x / sx2))
            offset_y = int(round(img_before[1] - y / sy2))

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
            if required_points is not None and len(pts) >= required_points:
                print(f"[안내] 이미 {required_points}개 점을 찍었습니다. Enter를 누르세요.")
                return

            view_w, view_h = clamp_offsets()
            px, py = disp_to_img(x, y, view_w, view_h)

            px = int(np.clip(px, 0, W - 1))
            py = int(np.clip(py, 0, H - 1))

            pts.append((px, py))
            print(f"[클릭] point {len(pts)}: pixel=({px}, {py})")

            if len(pts) % 2 == 0:
                print(f"[PAIR 완성] point {len(pts) - 1} -> point {len(pts)} 구간 생성 예정")
            else:
                print(f"[PAIR 시작] point {len(pts)} 선택됨. 다음 점을 찍으면 한 구간이 됩니다.")

            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, win_w, win_h)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        # Enter
        if key == 13:
            if required_points is not None:
                if len(pts) == required_points:
                    break
                else:
                    print(f"[경고] {required_points}개 점이 필요합니다. 현재 {len(pts)}개")
            else:
                if len(pts) >= min_points:
                    break
                else:
                    print(f"[경고] 최소 {min_points}개 점이 필요합니다. 현재 {len(pts)}개")

        # z 또는 Ctrl+Z
        elif key in [ord("z"), ord("Z"), 26]:
            if pts:
                removed = pts.pop()
                print(f"[UNDO] 제거된 점: {removed}")
                redraw()

        # ESC
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise SystemExit("사용자 종료")

    cv2.destroyWindow(window_name)

    return np.asarray(pts, dtype=float)


# ============================================================
# 좌표 변환 함수
# ============================================================

def make_calibration_info_fixed_origin(
    image_path,
    image_shape,
    known_length_m,
    known_length_pts_px,
    enu_y_axis=True,
):
    """
    원점과 X축 방향을 고정한 calibration 정보 생성.

    고정 규칙:
        원점: 이미지 좌측 상단 pixel (0, 0)
        +X 방향: 이미지 오른쪽
        theta_rad: 0
    """

    p1 = np.asarray(known_length_pts_px[0], dtype=float)
    p2 = np.asarray(known_length_pts_px[1], dtype=float)

    pixel_dist = np.linalg.norm(p2 - p1)

    if pixel_dist <= 1e-9:
        raise ValueError("알고 있는 길이의 두 점이 너무 가깝습니다.")

    m_per_px = known_length_m / pixel_dist

    H, W = image_shape[:2]

    origin_px = np.array([0.0, 0.0], dtype=float)
    x_axis_point_px = np.array([min(200.0, W - 1.0), 0.0], dtype=float)

    theta_rad = 0.0

    info = {
        "image_path": image_path,
        "known_length_m": float(known_length_m),
        "known_length_pts_px": known_length_pts_px.tolist(),
        "pixel_dist_for_known_length": float(pixel_dist),
        "m_per_px": float(m_per_px),
        "px_per_m": float(1.0 / m_per_px),
        "origin_px": origin_px.tolist(),
        "x_axis_point_px": x_axis_point_px.tolist(),
        "theta_rad": float(theta_rad),
        "theta_deg": float(math.degrees(theta_rad)),
        "enu_y_axis": bool(enu_y_axis),
        "origin_rule": "fixed_top_left",
        "x_axis_rule": "fixed_image_right",
    }

    return info


def px_to_meter(pt_px, calibration_info):
    """
    이미지 픽셀 좌표 -> 미터 좌표

    기준:
        원점: 이미지 좌측 상단 (0, 0)
        +X 방향: 이미지 오른쪽
        ENU_Y_AXIS=True이면 이미지 위쪽이 +Y
            따라서 이미지 아래쪽으로 갈수록 y_m은 음수
    """

    p = np.asarray(pt_px, dtype=float)
    origin_px = np.asarray(calibration_info["origin_px"], dtype=float)
    m_per_px = float(calibration_info["m_per_px"])
    enu_y_axis = bool(calibration_info["enu_y_axis"])

    dp = p - origin_px

    x_m = dp[0] * m_per_px
    y_m = dp[1] * m_per_px

    if enu_y_axis:
        y_m = -y_m

    return np.array([x_m, y_m], dtype=float)


def meter_to_px(pt_m, calibration_info):
    """
    미터 좌표 -> 이미지 픽셀 좌표

    기준:
        원점: 이미지 좌측 상단 (0, 0)
        +X 방향: 이미지 오른쪽
    """

    x_m, y_m = float(pt_m[0]), float(pt_m[1])

    m_per_px = float(calibration_info["m_per_px"])
    origin_px = np.asarray(calibration_info["origin_px"], dtype=float)
    enu_y_axis = bool(calibration_info["enu_y_axis"])

    if enu_y_axis:
        y_m = -y_m

    px = origin_px[0] + x_m / m_per_px
    py = origin_px[1] + y_m / m_per_px

    return np.array([px, py], dtype=float)


# ============================================================
# 2개씩 Pair 단위 보간 함수
# ============================================================

def generate_interpolated_points_by_pairs(
    points_m,
    interval_m=1.0,
    min_end_point_distance_m=0.5
):
    """
    사용자가 클릭한 점을 2개씩 한 쌍으로 보고,
    각 쌍 사이만 interval_m 간격으로 보간한다.

    개선 규칙:
        1. 시작점은 항상 추가
        2. 시작점으로부터 interval_m 간격으로 내부 점 생성
        3. 끝점은 마지막 생성 점과의 거리가 min_end_point_distance_m 이상일 때만 추가
        4. 끝점이 마지막 생성 점과 너무 가까우면 끝점은 저장하지 않음
    """

    points_m = np.asarray(points_m, dtype=float)

    if len(points_m) < 2:
        return (
            points_m.copy(),
            np.zeros(len(points_m), dtype=int),
            np.array(["single"] * len(points_m), dtype=object),
        )

    if len(points_m) % 2 != 0:
        print("[경고] 클릭한 점 개수가 홀수입니다.")
        print("[경고] 마지막 점은 짝이 없으므로 보간에서 제외됩니다.")
        points_m = points_m[:-1]

    interpolated_points = []
    pair_ids = []
    point_types = []

    pair_id = 0

    for i in range(0, len(points_m), 2):
        start = points_m[i]
        end = points_m[i + 1]

        vec = end - start
        dist = np.linalg.norm(vec)

        if dist <= 1e-9:
            print(f"[경고] Pair {pair_id}: 시작점과 끝점이 너무 가까워 제외합니다.")
            continue

        direction = vec / dist

        interpolated_points.append(start)
        pair_ids.append(pair_id)
        point_types.append("start")

        last_point = start.copy()

        current_dist = interval_m

        while current_dist < dist:
            new_point = start + direction * current_dist

            interpolated_points.append(new_point)
            pair_ids.append(pair_id)
            point_types.append("interpolated")

            last_point = new_point.copy()
            current_dist += interval_m

        distance_to_end = np.linalg.norm(end - last_point)

        if distance_to_end >= min_end_point_distance_m:
            interpolated_points.append(end)
            pair_ids.append(pair_id)
            point_types.append("end")
            end_status = "끝점 추가"
        else:
            end_status = "끝점 제외"

        print(
            f"[PAIR {pair_id}] "
            f"거리={dist:.3f} m, "
            f"마지막점-끝점 거리={distance_to_end:.3f} m, "
            f"{end_status}, "
            f"생성 포인트={pair_ids.count(pair_id)}개"
        )

        pair_id += 1

    return (
        np.asarray(interpolated_points, dtype=float),
        np.asarray(pair_ids, dtype=int),
        np.asarray(point_types, dtype=object),
    )


# ============================================================
# 저장 함수
# ============================================================

def save_points_csv(points_px, points_m, pair_ids, point_types, output_csv_path):
    df = pd.DataFrame({
        "index": np.arange(len(points_m), dtype=int),
        "pair_id": pair_ids,
        "point_type": point_types,
        "x_m": points_m[:, 0],
        "y_m": points_m[:, 1],
        "pixel_x": points_px[:, 0],
        "pixel_y": points_px[:, 1],
    })

    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"[저장] CSV: {output_csv_path}")

    return df


def save_calibration_json(calibration_info, output_json_path):
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(calibration_info, f, ensure_ascii=False, indent=4)

    print(f"[저장] calibration JSON: {output_json_path}")


def draw_preview(
    img,
    calibration_info,
    points_px,
    points_m,
    clicked_points_px=None,
    pair_ids=None,
    output_preview_path=None,
):
    preview = img.copy()

    H, W = img.shape[:2]

    # 알고 있는 길이 표시
    known_pts = np.asarray(calibration_info["known_length_pts_px"], dtype=float)
    kp1 = tuple(np.round(known_pts[0]).astype(int))
    kp2 = tuple(np.round(known_pts[1]).astype(int))

    cv2.line(preview, kp1, kp2, (255, 0, 255), 3)
    cv2.circle(preview, kp1, 6, (255, 0, 255), -1)
    cv2.circle(preview, kp2, 6, (255, 0, 255), -1)
    cv2.putText(
        preview,
        f"Known length: {calibration_info['known_length_m']:.2f} m",
        kp1,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )

    # 고정 원점, X축 표시
    origin_px = (0, 0)
    x_axis_px = (min(200, W - 1), 0)

    cv2.circle(preview, origin_px, 10, (0, 0, 255), -1)
    cv2.putText(
        preview,
        "Origin (0,0)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    cv2.arrowedLine(
        preview,
        origin_px,
        x_axis_px,
        (0, 0, 255),
        3,
        tipLength=0.15,
    )
    cv2.putText(
        preview,
        "+X",
        (x_axis_px[0] + 10, x_axis_px[1] + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    # Y 방향 표시
    if calibration_info["enu_y_axis"]:
        y_text = "+Y is upward"
    else:
        y_text = "+Y is downward"

    cv2.putText(
        preview,
        y_text,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    # Pair별 보간 선 표시
    if pair_ids is not None and len(points_px) >= 2:
        unique_pair_ids = np.unique(pair_ids)

        for pid in unique_pair_ids:
            pair_points = points_px[pair_ids == pid]

            if len(pair_points) < 2:
                continue

            for i in range(len(pair_points) - 1):
                p1 = tuple(np.round(pair_points[i]).astype(int))
                p2 = tuple(np.round(pair_points[i + 1]).astype(int))
                cv2.line(preview, p1, p2, (0, 255, 0), 2)

    # 보간된 전체 좌표 점 표시
    for i, (px, py) in enumerate(points_px, start=1):
        p = (int(round(px)), int(round(py)))

        cv2.circle(preview, p, RP_POINT_OUTLINE_RADIUS, (255, 255, 255), -1)
        cv2.circle(preview, p, RP_POINT_RADIUS, (255, 0, 0), -1)
        cv2.circle(preview, p, RP_POINT_RADIUS, (0, 0, 0), 2)

        if i == 1 or i == len(points_px) or i % 5 == 0:
            cv2.putText(
                preview,
                str(i),
                (p[0] + RP_POINT_RADIUS + 4, p[1] - RP_POINT_RADIUS - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                RP_POINT_TEXT_SCALE,
                (255, 0, 0),
                RP_POINT_TEXT_THICKNESS,
            )

    # 사용자가 실제로 클릭한 기준점 표시
    if clicked_points_px is not None:
        for i, (px, py) in enumerate(clicked_points_px, start=1):
            p = (int(round(px)), int(round(py)))

            cv2.circle(preview, p, CLICKED_POINT_RADIUS, (0, 255, 255), -1)
            cv2.putText(
                preview,
                f"C{i}",
                (p[0] + 10, p[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        usable_len = len(clicked_points_px)

        if usable_len % 2 != 0:
            usable_len -= 1

        for i in range(0, usable_len, 2):
            p1 = tuple(np.round(clicked_points_px[i]).astype(int))
            p2 = tuple(np.round(clicked_points_px[i + 1]).astype(int))
            cv2.line(preview, p1, p2, (0, 180, 255), 3)

    if output_preview_path is None:
        raise ValueError("output_preview_path가 지정되지 않았습니다.")

    imwrite_unicode(output_preview_path, preview)
    print(f"[저장] preview image: {output_preview_path}")


# ============================================================
# 메인 실행
# ============================================================

def main():
    # ------------------------------------------------------------
    # 1. 도면 불러오기
    # ------------------------------------------------------------
    img = imread_unicode(FLOOR_MAP_IMAGE_NAME)

    if img is None:
        raise FileNotFoundError(f"도면 이미지를 불러올 수 없습니다: {FLOOR_MAP_IMAGE_NAME}")

    print("========== 도면 좌표 수집 시작 ==========")
    print(f"[도면] {FLOOR_MAP_IMAGE_NAME}")
    print(f"[알고 있는 길이] {KNOWN_LENGTH_M} m")
    print(f"[보간 간격] {INTERPOLATION_INTERVAL_M} m")
    print(f"[끝점 최소 거리] {MIN_END_POINT_DISTANCE_M} m")
    print(f"[저장 이름] {SAVE_NAME}")
    print("[원점] 이미지 좌측 상단 pixel (0, 0)")
    print("[X+ 방향] 이미지 오른쪽")
    print("========================================")

    # ------------------------------------------------------------
    # 2. 알고 있는 길이 지정
    # ------------------------------------------------------------
    known_length_pts_px = collect_points_zoom_pan(
        window_name="Step 1 - Known Length",
        base_img=img.copy(),
        instruction_lines=[
            "Step 1: Click two endpoints of the known length.",
            "Origin is fixed to image top-left (0,0).",
            "+X direction is fixed to image right.",
            "Left click: point | z: undo | Mouse wheel: zoom | Right drag: pan",
            "After selecting 2 points, press Enter.",
        ],
        required_points=2,
        win_w=WINDOW_WIDTH,
        win_h=WINDOW_HEIGHT,
    )

    # ------------------------------------------------------------
    # 3. Calibration 정보 생성
    # 원점과 X축 방향은 자동 고정
    # ------------------------------------------------------------
    calibration_info = make_calibration_info_fixed_origin(
        image_path=FLOOR_MAP_IMAGE_NAME,
        image_shape=img.shape,
        known_length_m=KNOWN_LENGTH_M,
        known_length_pts_px=known_length_pts_px,
        enu_y_axis=ENU_Y_AXIS,
    )

    print()
    print("========== Calibration 정보 ==========")
    print(f"m_per_px: {calibration_info['m_per_px']:.8f} m/px")
    print(f"px_per_m: {calibration_info['px_per_m']:.3f} px/m")
    print(f"theta_deg: {calibration_info['theta_deg']:.3f} deg")
    print(f"origin_px: {calibration_info['origin_px']}")
    print(f"x_axis_rule: {calibration_info['x_axis_rule']}")
    print(f"enu_y_axis: {calibration_info['enu_y_axis']}")
    print("======================================")
    print()

    # save_calibration_json(
    #     calibration_info=calibration_info,
    #     output_json_path=OUTPUT_CALIBRATION_JSON_PATH,
    # )

    # ------------------------------------------------------------
    # 4. 경로 기준선 Pair 찍기
    # ------------------------------------------------------------
    clicked_points_px = collect_points_zoom_pan(
        window_name="Step 2 - Collect Route Line Pairs",
        base_img=img.copy(),
        instruction_lines=[
            "Step 2: Click points as pairs.",
            "Pair rule: 1-2, 3-4, 5-6 will be interpolated.",
            "No interpolation is made between 2-3, 4-5, ...",
            "Origin: image top-left (0,0), +X: right.",
            "Left click: point | z: undo | Mouse wheel: zoom | Right drag: pan",
            "Press Enter when finished.",
        ],
        required_points=None,
        min_points=2,
        win_w=WINDOW_WIDTH,
        win_h=WINDOW_HEIGHT,
    )

    if len(clicked_points_px) % 2 != 0:
        print()
        print("[경고] 클릭한 점 개수가 홀수입니다.")
        print("[경고] 마지막 점은 pair가 없으므로 제외됩니다.")
        print(f"[제외되는 점] point {len(clicked_points_px)}: {clicked_points_px[-1]}")
        print()

    # ------------------------------------------------------------
    # 클릭한 픽셀 좌표 -> 미터 좌표 변환
    # ------------------------------------------------------------
    clicked_points_m = np.array([
        px_to_meter(pt, calibration_info)
        for pt in clicked_points_px
    ], dtype=float)

    # ------------------------------------------------------------
    # 클릭한 점을 2개씩 pair로 묶어서 각 pair 사이만 1m 간격 보간
    # ------------------------------------------------------------
    points_m, pair_ids, point_types = generate_interpolated_points_by_pairs(
        points_m=clicked_points_m,
        interval_m=INTERPOLATION_INTERVAL_M,
        min_end_point_distance_m=MIN_END_POINT_DISTANCE_M,
    )

    # ------------------------------------------------------------
    # 보간된 미터 좌표 -> 픽셀 좌표 변환
    # ------------------------------------------------------------
    points_px = np.array([
        meter_to_px(pt, calibration_info)
        for pt in points_m
    ], dtype=float)

    print()
    print("========== 보간 결과 ==========")
    print(f"사용자가 클릭한 기준점 개수: {len(clicked_points_m)}")
    print(f"사용 가능한 pair 개수: {len(np.unique(pair_ids)) if len(pair_ids) > 0 else 0}")
    print(f"최종 저장되는 포인트 개수: {len(points_m)}")
    print(f"보간 간격: {INTERPOLATION_INTERVAL_M} m")
    print(f"끝점 최소 거리: {MIN_END_POINT_DISTANCE_M} m")
    print("================================")
    print()

    # ------------------------------------------------------------
    # CSV 저장
    # ------------------------------------------------------------
    df = save_points_csv(
        points_px=points_px,
        points_m=points_m,
        pair_ids=pair_ids,
        point_types=point_types,
        output_csv_path=OUTPUT_CSV_PATH,
    )

    # ------------------------------------------------------------
    # Preview 이미지 저장
    # ------------------------------------------------------------
    draw_preview(
        img=img,
        calibration_info=calibration_info,
        points_px=points_px,
        points_m=points_m,
        clicked_points_px=clicked_points_px,
        pair_ids=pair_ids,
        output_preview_path=OUTPUT_PREVIEW_PATH,
    )

    print()
    print("========== 좌표 저장 완료 ==========")
    print(df)
    print("===================================")


if __name__ == "__main__":
    main()