# ============================================================
# 보정된 PDR좌표 + RSSI 측정값을 RP 위치에 매핑한 뒤,
# PDR로 채워지지 않은 빈 RP만 선형 보간 + 상하좌우 1m 이웃 평균으로 채우는 코드
# ============================================================

import pandas as pd
import numpy as np
import os
import re

# scipy가 있으면 선형 보간 사용
try:
    from scipy.interpolate import LinearNDInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# 사용자 설정
# ============================================================
BASE_DIR = os.getcwd()

MASTER_POS_PATH = os.path.join(BASE_DIR, "temp_data", "rp_pos.csv")
PDR_WIFI_PATHS = os.path.join(BASE_DIR, "temp_data", "pdr_WiFi_map_matched.csv")
SAVE_PATH = os.path.join(BASE_DIR, "radio_map.csv")

# PDR map-matched 좌표 컬럼
PDR_X_COL = "x"
PDR_Y_COL = "y"

# master 위치 컬럼
MASTER_X_COL = "x_m"
MASTER_Y_COL = "y_m"

# master 픽셀 좌표 컬럼
MASTER_PIXEL_X_COL = "pixel_x"
MASTER_PIXEL_Y_COL = "pixel_y"

# RP 근처라고 판단할 반경 meter
SEARCH_RADIUS_M = 0.5

# RSSI가 없을 때 사용할 값
MISSING_RSSI = -100

# 상하좌우 격자 간격 (2026-09: RP 간격 1m -> 3m 변경에 맞춰 조정)
GRID_STEP_M = 3.0

# 좌표가 소수점 오차를 가질 수 있으므로 허용 오차
GRID_TOL_M = 0.15

# 저장 옵션
SAVE_METER_COORDS = True      # x, y meter 좌표도 저장할지 여부
SAVE_DEBUG_COLUMNS = True     # is_pdr_filled, 보간 개수 등 디버그 컬럼 저장 여부


# ============================================================
# RSSI 컬럼 자동 추출
# ============================================================
def get_rssi_columns(df):
    """
    Wi-Fi RSSI 컬럼 자동 추출.
    지원 형태:
        1. MAC 주소만 있는 경우: 00:40:5A:AF:98:BA
        2. MAC/SSID 형태: 00:40:5A:AF:98:BA/SSID
    """
    exclude_cols = {
        "time",
        "elapsed_time_sec",

        "original_x",
        "original_y",
        "x",
        "y",
        "heading",
        "is_turn",

        "wifi_id",

        "pdr_x_raw",
        "pdr_y_raw",
        "pdr_x_start_aligned",
        "pdr_y_start_aligned",

        "map_error_x",
        "map_error_y",
        "matched_x",
        "matched_y",
        "x_reconstructed",
        "y_reconstructed",

        "is_manual_turn",
        "is_anchor",
    }

    mac_pattern = re.compile(
        r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(/.*)?$"
    )

    rssi_cols = []

    for col in df.columns:
        col_str = str(col).strip()

        if col_str in exclude_cols:
            continue

        if mac_pattern.match(col_str):
            rssi_cols.append(col)

    return rssi_cols


# ============================================================
# RSSI 평균 계산 함수
# -100은 제외하고 평균
# 모두 -100 또는 NaN이면 -100 반환
# ============================================================
def mean_rssi_excluding_missing(values, missing_value=-100):
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    valid = values[
        (~np.isnan(values)) &
        (values != missing_value)
    ]

    if len(valid) == 0:
        return missing_value

    return float(np.mean(valid))


# ============================================================
# 특정 RP 기준 상하좌우 1m 이웃 RP index 찾기
# ============================================================
def get_cardinal_neighbor_indices(
    df,
    target_x,
    target_y,
    x_col="x",
    y_col="y",
    step=1.0,
    tol=0.15
):
    """
    현재 RP 기준 상/하/좌/우 1m 위치에 있는 RP를 찾음.
    좌표 오차를 고려해 tol 이내의 가장 가까운 RP를 사용.
    """
    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)

    target_positions = [
        (target_x + step, target_y),  # 오른쪽
        (target_x - step, target_y),  # 왼쪽
        (target_x, target_y + step),  # 위쪽 또는 아래쪽
        (target_x, target_y - step),  # 아래쪽 또는 위쪽
    ]

    neighbor_indices = []

    for nx, ny in target_positions:
        dist = np.sqrt((xs - nx) ** 2 + (ys - ny) ** 2)

        if len(dist) == 0:
            continue

        min_idx = int(np.argmin(dist))

        if dist[min_idx] <= tol:
            neighbor_indices.append(min_idx)

    return list(set(neighbor_indices))


# ============================================================
# 데이터 로드
# ============================================================
pdr_wifi_df = pd.read_csv(PDR_WIFI_PATHS)
master_df = pd.read_csv(MASTER_POS_PATH)

required_master_cols = [
    MASTER_X_COL,
    MASTER_Y_COL,
    MASTER_PIXEL_X_COL,
    MASTER_PIXEL_Y_COL,
]

for col in required_master_cols:
    if col not in master_df.columns:
        raise ValueError(f"master 위치 파일에 {col} 컬럼이 없습니다.")

for col in [PDR_X_COL, PDR_Y_COL]:
    if col not in pdr_wifi_df.columns:
        raise ValueError(f"PDR Wi-Fi 파일에 {col} 컬럼이 없습니다.")

print("master RP 개수:", len(master_df))


# ============================================================
# 전체 AP 컬럼 정리
# ============================================================
rssi_cols = get_rssi_columns(pdr_wifi_df)

if len(rssi_cols) == 0:
    raise ValueError("RSSI AP 컬럼을 찾지 못했습니다. MAC 주소 컬럼명을 확인하세요.")

print("AP 개수:", len(rssi_cols))


# ============================================================
# RSSI 컬럼 숫자 변환 및 NaN 처리
# ============================================================
pdr_wifi_df[rssi_cols] = pdr_wifi_df[rssi_cols].apply(
    pd.to_numeric,
    errors="coerce"
)

pdr_wifi_df[rssi_cols] = pdr_wifi_df[rssi_cols].fillna(MISSING_RSSI)


# ============================================================
# 1단계: PDR 경로와 RP 매칭해서 실제 RSSI 값 채우기
# ============================================================
radio_map_rows = []

pdr_x = pdr_wifi_df[PDR_X_COL].to_numpy(dtype=float)
pdr_y = pdr_wifi_df[PDR_Y_COL].to_numpy(dtype=float)

for _, rp in master_df.iterrows():
    rp_x = float(rp[MASTER_X_COL])
    rp_y = float(rp[MASTER_Y_COL])

    rp_pixel_x = float(rp[MASTER_PIXEL_X_COL])
    rp_pixel_y = float(rp[MASTER_PIXEL_Y_COL])

    # --------------------------------------------------------
    # RP 위치와 PDR Wi-Fi 수집 위치 사이 거리 계산
    # --------------------------------------------------------
    dist = np.sqrt(
        (pdr_x - rp_x) ** 2 +
        (pdr_y - rp_y) ** 2
    )

    near_mask = dist <= SEARCH_RADIUS_M
    near_df = pdr_wifi_df.loc[near_mask]

    is_pdr_filled = len(near_df) > 0

    # --------------------------------------------------------
    # RP 기본 정보 저장
    # --------------------------------------------------------
    row = {
        "x": rp_x,
        "y": rp_y,
        "pixel_x": rp_pixel_x,
        "pixel_y": rp_pixel_y,

        # 디버그용
        "is_pdr_filled": is_pdr_filled,
        "near_sample_count": len(near_df),
        "linear_filled_ap_count": 0,
        "neighbor_filled_ap_count": 0,
    }

    # --------------------------------------------------------
    # 근처 PDR Wi-Fi 데이터가 없으면 일단 모든 AP를 -100
    # --------------------------------------------------------
    if len(near_df) == 0:
        for ap in rssi_cols:
            row[ap] = MISSING_RSSI

    # --------------------------------------------------------
    # 근처 PDR Wi-Fi 데이터가 있으면 AP별 평균
    # 단, -100은 평균에서 제외
    # --------------------------------------------------------
    else:
        for ap in rssi_cols:
            row[ap] = mean_rssi_excluding_missing(
                near_df[ap],
                missing_value=MISSING_RSSI
            )

    radio_map_rows.append(row)


radio_map_df = pd.DataFrame(radio_map_rows)

print()
print("PDR로 직접 채워진 RP 개수:", radio_map_df["is_pdr_filled"].sum())
print("PDR로 채워지지 않은 빈 RP 개수:", (~radio_map_df["is_pdr_filled"]).sum())


# ============================================================
# 2단계: PDR로 채워지지 않은 RP만 선형 보간
# ============================================================
if SCIPY_AVAILABLE:
    print()
    print("선형 보간 수행 중...")

    target_mask_base = ~radio_map_df["is_pdr_filled"]

    target_points = radio_map_df.loc[target_mask_base, ["x", "y"]].to_numpy(dtype=float)
    target_indices = radio_map_df.index[target_mask_base].to_numpy()

    for ap in rssi_cols:
        # ----------------------------------------------------
        # 보간 기준점은 실제 PDR로 RSSI가 들어간 RP 중,
        # 해당 AP 값이 -100이 아닌 RP만 사용
        # ----------------------------------------------------
        donor_mask = (
            radio_map_df["is_pdr_filled"] &
            (radio_map_df[ap] != MISSING_RSSI) &
            (~radio_map_df[ap].isna())
        )

        donor_df = radio_map_df.loc[donor_mask, ["x", "y", ap]].copy()

        # 선형 보간은 최소 3개 이상의 기준점 필요
        if len(donor_df) < 3:
            continue

        donor_points = donor_df[["x", "y"]].to_numpy(dtype=float)
        donor_values = donor_df[ap].to_numpy(dtype=float)

        try:
            interpolator = LinearNDInterpolator(
                donor_points,
                donor_values,
                fill_value=np.nan
            )

            interp_values = interpolator(target_points)

        except Exception:
            # 점들이 일직선 등으로 배치된 경우 Qhull 에러가 날 수 있음
            continue

        valid_interp_mask = ~np.isnan(interp_values)

        valid_target_indices = target_indices[valid_interp_mask]
        valid_values = interp_values[valid_interp_mask]

        # ----------------------------------------------------
        # PDR로 채워진 RP는 제외.
        # 빈 RP에 대해서만 선형 보간 값 저장.
        # ----------------------------------------------------
        for idx, value in zip(valid_target_indices, valid_values):
            if radio_map_df.at[idx, "is_pdr_filled"]:
                continue

            if radio_map_df.at[idx, ap] == MISSING_RSSI:
                radio_map_df.at[idx, ap] = float(value)
                radio_map_df.at[idx, "linear_filled_ap_count"] += 1

else:
    print()
    print("[주의] scipy가 없어 선형 보간은 건너뜁니다.")
    print("설치 명령 예: pip install scipy")


# ============================================================
# 3단계: 선형 보간 후에도 -100인 빈 RP는 상하좌우 1m 이웃 평균으로 채우기
# ============================================================
print()
print("상하좌우 1m 이웃 평균 보정 수행 중...")

empty_target_indices = radio_map_df.index[~radio_map_df["is_pdr_filled"]].to_numpy()

for idx in empty_target_indices:
    target_x = float(radio_map_df.at[idx, "x"])
    target_y = float(radio_map_df.at[idx, "y"])

    neighbor_indices = get_cardinal_neighbor_indices(
        radio_map_df,
        target_x,
        target_y,
        x_col="x",
        y_col="y",
        step=GRID_STEP_M,
        tol=GRID_TOL_M
    )

    if len(neighbor_indices) == 0:
        continue

    for ap in rssi_cols:
        # 이미 선형 보간으로 채워졌으면 건드리지 않음
        if radio_map_df.at[idx, ap] != MISSING_RSSI:
            continue

        neighbor_values = radio_map_df.loc[neighbor_indices, ap]

        neighbor_mean = mean_rssi_excluding_missing(
            neighbor_values,
            missing_value=MISSING_RSSI
        )

        if neighbor_mean != MISSING_RSSI:
            radio_map_df.at[idx, ap] = neighbor_mean
            radio_map_df.at[idx, "neighbor_filled_ap_count"] += 1


# ============================================================
# 최종 결측 상태 확인
# ============================================================
ap_values = radio_map_df[rssi_cols]

all_missing_rows = (ap_values == MISSING_RSSI).all(axis=1)
partial_missing_rows = (ap_values == MISSING_RSSI).any(axis=1) & (~all_missing_rows)

print()
print("최종 Radio Map 정보")
print("전체 RP 개수:", len(radio_map_df))
print("PDR 직접 채움 RP 개수:", radio_map_df["is_pdr_filled"].sum())
print("보간 대상 RP 개수:", (~radio_map_df["is_pdr_filled"]).sum())
print("선형 보간으로 채워진 AP 값 개수:", int(radio_map_df["linear_filled_ap_count"].sum()))
print("상하좌우 평균으로 채워진 AP 값 개수:", int(radio_map_df["neighbor_filled_ap_count"].sum()))
print("모든 AP가 여전히 -100인 RP 개수:", int(all_missing_rows.sum()))
print("일부 AP만 -100인 RP 개수:", int(partial_missing_rows.sum()))


# ============================================================
# 컬럼 순서 정리
# ============================================================
front_cols = []

if SAVE_METER_COORDS:
    front_cols += ["x", "y"]

front_cols += [
    "pixel_x",
    "pixel_y",
]

if SAVE_DEBUG_COLUMNS:
    front_cols += [
        "is_pdr_filled",
        "near_sample_count",
        "linear_filled_ap_count",
        "neighbor_filled_ap_count",
    ]

other_cols = [
    col for col in radio_map_df.columns
    if col not in front_cols
]

# AP 컬럼만 뒤쪽으로 정렬
non_ap_other_cols = [
    col for col in other_cols
    if col not in rssi_cols
]

radio_map_df = radio_map_df[
    front_cols +
    non_ap_other_cols +
    rssi_cols
]


# ============================================================
# 저장
# ============================================================
radio_map_df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

print()
print("저장 완료:", SAVE_PATH)
print("출력 shape:", radio_map_df.shape)
print()
print(radio_map_df.head())