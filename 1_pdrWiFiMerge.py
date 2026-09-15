import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.signal import butter, filtfilt, find_peaks

BASE_DIR = os.getcwd()

IMU_DATA_PATH = os.path.join(BASE_DIR, "data", "imu.csv")
WIFI_2_4GHZ_DATA_PATH = os.path.join(BASE_DIR, "data", "rssi_2ghz.csv")
WIFI_5GHZ_DATA_PATH = os.path.join(BASE_DIR, "data", "rssi_5ghz.csv")

OUTPUT_DATA_PATH = os.path.join(BASE_DIR, "temp_data")
OUTPUT_DATA_NAME = "pdr_WiFi.csv"

STEP_LENGTH = 0.7
FS = 50


def load_sensor(file_path):
    df = pd.read_csv(file_path)

    df.columns = [
        "time", "board_time",
        "ax", "ay", "az",
        "gx", "gy", "gz",
        "rotation_hint"
    ]

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    start_dt = df["time"].iloc[0]
    df["Elapsed Time"] = (df["time"] - start_dt).dt.total_seconds()

    return df


def pdr(df, fs=50, step_length=0.7, height=1.1, turn_angle_deg=60):
    acc_norm = np.sqrt(
        df["ax"] ** 2 +
        df["ay"] ** 2 +
        df["az"] ** 2
    )

    nyquist = 0.5 * fs
    normal_cutoff = 3.0 / nyquist

    b, a = butter(4, normal_cutoff, btype="low", analog=False)
    acc_norm_smoothed = filtfilt(b, a, acc_norm)

    peaks, _ = find_peaks(
        acc_norm_smoothed,
        height=height,
        distance=int(fs * 0.4)
    )

    ax = df["ax"].values
    ay = df["ay"].values
    az = df["az"].values

    pitch = np.arctan2(ay, az)
    roll = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

    gx = df["gx"].values
    gy = df["gy"].values
    gz = df["gz"].values

    gyro_corr = np.zeros((len(df), 3))

    for i in range(len(df)):
        cos_p = np.cos(pitch[i])
        sin_p = np.sin(pitch[i])
        cos_r = np.cos(roll[i])
        sin_r = np.sin(roll[i])

        R_inv_pitch = np.array([
            [1,     0,      0],
            [0, cos_p, -sin_p],
            [0, sin_p,  cos_p]
        ])

        R_inv_roll = np.array([
            [ cos_r, 0, sin_r],
            [     0, 1,     0],
            [-sin_r, 0, cos_r]
        ])

        R_tilt = R_inv_roll @ R_inv_pitch

        gyro_vector = np.array([gx[i], gy[i], gz[i]])
        gyro_corr[i] = R_tilt @ gyro_vector

    gz_corr = gyro_corr[:, 2]

    # 기존 코드의 FS 대신 함수 인자인 fs 사용
    heading = np.cumsum(gz_corr) / fs
    rad_heading = np.radians(heading)

    x_pos = [0.0]
    y_pos = [0.0]

    step_times = [df["time"].iloc[0]]
    step_sample_idx = [0]

    turn_idx = [0]
    prev_heading = None

    for step_count, step_idx in enumerate(peaks, start=1):
        psi = rad_heading[step_idx]

        if prev_heading is not None:
            heading_diff = psi - prev_heading
            heading_diff = np.arctan2(
                np.sin(heading_diff),
                np.cos(heading_diff)
            )

            if np.abs(heading_diff) > np.radians(turn_angle_deg):
                turn_idx.append(step_count - 1)

        prev_heading = psi

        delta_x = step_length * np.cos(psi)
        delta_y = step_length * np.sin(psi)

        x_pos.append(x_pos[-1] + delta_x)
        y_pos.append(y_pos[-1] + delta_y)

        step_times.append(df["time"].iloc[step_idx])
        step_sample_idx.append(step_idx)

    last_idx = len(x_pos) - 1

    if last_idx not in turn_idx:
        turn_idx.append(last_idx)

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)
    turn_idx = np.array(sorted(set(turn_idx)), dtype=int)

    # WiFi와 시간 매칭하기 위한 PDR DataFrame
    pdr_df = pd.DataFrame({
        "time": step_times,
        "step_no": np.arange(len(x_pos)),
        "sample_idx": step_sample_idx,
        "x": x_pos,
        "y": y_pos
    })

    pdr_df["is_turn"] = pdr_df["step_no"].isin(turn_idx).astype(int)

    return x_pos, y_pos, turn_idx, pdr_df


def load_wifi_rssi(file_path):
    """
    WiFi RSSI CSV 로드

    입력 포맷:
        timestamp, MAC1, MAC2, MAC3, ...

    출력:
        time, MAC1, MAC2, MAC3, ...
    """
    df_wifi = pd.read_csv(file_path)

    # 첫 번째 컬럼을 시간 컬럼으로 사용
    df_wifi = df_wifi.rename(columns={df_wifi.columns[0]: "time"})

    df_wifi["time"] = pd.to_datetime(df_wifi["time"], errors="coerce")
    df_wifi = df_wifi.dropna(subset=["time"])
    df_wifi = df_wifi.sort_values("time").reset_index(drop=True)

    return df_wifi


def average_wifi_in_time_window(
    pdr_times,
    wifi_df,
    window_sec=0.5,
    fill_value=-100
):
    """
    각 PDR 시간 기준 ±window_sec 안에 들어온 WiFi RSSI를 AP별 평균낸다.
    평균 계산 시 NaN, -100은 제외한다.

    Parameters
    ----------
    pdr_times : Series
        PDR 좌표 시간
    wifi_df : DataFrame
        time, MAC1, MAC2, ... 형태의 WiFi DataFrame
    window_sec : float
        PDR 시간 기준 앞뒤 시간 범위
    fill_value : int or float
        미관측 RSSI 값

    Returns
    -------
    wifi_avg_df : DataFrame
        PDR row 개수와 동일한 WiFi 평균 RSSI DataFrame
    """

    mac_cols = [col for col in wifi_df.columns if col != "time"]

    wifi_times = wifi_df["time"]
    wifi_rssi = wifi_df[mac_cols].copy()

    # -100은 미관측으로 보고 평균에서 제외
    wifi_rssi = wifi_rssi.replace(fill_value, np.nan)

    result_rows = []

    window = pd.Timedelta(seconds=window_sec)

    for t in pdr_times:
        start_t = t - window
        end_t = t + window

        mask = (wifi_times >= start_t) & (wifi_times <= end_t)

        if mask.sum() == 0:
            avg_row = pd.Series(index=mac_cols, dtype=float)
        else:
            avg_row = wifi_rssi.loc[mask, mac_cols].mean(axis=0, skipna=True)

        result_rows.append(avg_row)

    wifi_avg_df = pd.DataFrame(result_rows)

    # 평균 이후에도 없는 AP는 -100으로 채움
    wifi_avg_df = wifi_avg_df.fillna(fill_value)

    return wifi_avg_df


def merge_pdr_with_wifi_window_mean(
    pdr_df,
    wifi_2g_path,
    wifi_5g_path,
    output_dir,
    output_name="pdr_WiFi.csv",
    window_sec=0.5,
    fill_value=-100
):
    """
    PDR 좌표 기준 ±window_sec 안에 들어온 2.4GHz, 5GHz WiFi RSSI를
    AP별 평균내어 결합한다.

    출력 포맷:
        time, x, y, is_turn, MAC1, MAC2, ...
    """

    pdr_base = pdr_df[["time", "x", "y", "is_turn"]].copy()
    pdr_base = pdr_base.sort_values("time").reset_index(drop=True)

    wifi_2g = load_wifi_rssi(wifi_2g_path)
    wifi_5g = load_wifi_rssi(wifi_5g_path)

    # 2.4GHz와 5GHz에 같은 MAC 컬럼명이 있으면 충돌 방지
    duplicated_macs = (set(wifi_2g.columns) & set(wifi_5g.columns)) - {"time"}

    if duplicated_macs:
        wifi_2g = wifi_2g.rename(
            columns={mac: f"{mac}_2g" for mac in duplicated_macs}
        )
        wifi_5g = wifi_5g.rename(
            columns={mac: f"{mac}_5g" for mac in duplicated_macs}
        )

    wifi_2g_avg = average_wifi_in_time_window(
        pdr_times=pdr_base["time"],
        wifi_df=wifi_2g,
        window_sec=window_sec,
        fill_value=fill_value
    )

    wifi_5g_avg = average_wifi_in_time_window(
        pdr_times=pdr_base["time"],
        wifi_df=wifi_5g,
        window_sec=window_sec,
        fill_value=fill_value
    )

    merged = pd.concat(
        [
            pdr_base.reset_index(drop=True),
            wifi_2g_avg.reset_index(drop=True),
            wifi_5g_avg.reset_index(drop=True)
        ],
        axis=1
    )

    # time 포맷 정리
    merged["time"] = (
        merged["time"]
        .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        .str[:-3]
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_name)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    rssi_cols = [
        col for col in merged.columns
        if col not in ["time", "x", "y", "is_turn"]
    ]

    print("저장 완료:", output_path)
    print("출력 shape:", merged.shape)
    print("PDR-WiFi row 개수:", len(merged))
    print("WiFi MAC 컬럼 개수:", len(rssi_cols))
    print(f"WiFi 평균 시간 범위: PDR time 기준 ±{window_sec}초")

    return merged


df = load_sensor(IMU_DATA_PATH)

x, y, idx, pdr_df = pdr(
    df,
    fs=FS,
    step_length=STEP_LENGTH,
    height=1.1
)

df_pdr_wifi = merge_pdr_with_wifi_window_mean(
    pdr_df=pdr_df,
    wifi_2g_path=WIFI_2_4GHZ_DATA_PATH,
    wifi_5g_path=WIFI_5GHZ_DATA_PATH,
    output_dir=OUTPUT_DATA_PATH,
    output_name=OUTPUT_DATA_NAME,
    window_sec=0.5,
    fill_value=-100
)

print(df_pdr_wifi.head())