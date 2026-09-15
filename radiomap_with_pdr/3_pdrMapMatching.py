import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


BASE_DIR = os.getcwd()

PDR_WIFI_PATH = os.path.join(BASE_DIR, "temp_data", "pdr_WiFi.csv")
NODE_PATH = os.path.join(BASE_DIR, "temp_data", "node.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "temp_data")
OUTPUT_NAME = "pdr_WiFi_map_matched.csv"


def load_pdr_wifi(file_path):
    """
    pdr_WiFi.csv 로드

    필요 컬럼:
        time, x, y, is_turn, MAC...
    """
    df = pd.read_csv(file_path)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    required_cols = ["x", "y", "is_turn"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"pdr_WiFi 파일에 '{col}' 컬럼이 없습니다.")

    return df


def load_turn_points(file_path):
    """
    node.csv 로드

    필요 컬럼:
        x_m, y_m

    반환:
        turn_points = [[x1, y1], [x2, y2], ...]
    """
    df_node = pd.read_csv(file_path)

    if "index" in df_node.columns:
        df_node = df_node.sort_values("index").reset_index(drop=True)

    required_cols = ["x_m", "y_m"]
    for col in required_cols:
        if col not in df_node.columns:
            raise ValueError(f"node 파일에 '{col}' 컬럼이 없습니다.")

    turn_points = df_node[["x_m", "y_m"]].to_numpy(dtype=float)

    return turn_points, df_node


def map_matching_by_turn_points(
    x_pos,
    y_pos,
    turn_points,
    turn_idx
):
    """
    turn point 사이 구간을 실제 node 선분 위로 projection 한다.
    
    기존 방식:
        원본 PDR 좌표 + 선형 오차 보정
        -> 원본 곡률이 남음
        
    변경 방식:
        각 구간의 PDR 누적 이동거리 비율을 계산한 뒤,
        실제 turn point 선분 위 좌표로 변환
        -> 복도/직선 구간이 직선으로 보정됨
    """

    x_pos = np.asarray(x_pos, dtype=float)
    y_pos = np.asarray(y_pos, dtype=float)
    turn_points = np.asarray(turn_points, dtype=float)
    turn_idx = np.asarray(turn_idx, dtype=int)

    if len(turn_points) != len(turn_idx):
        raise ValueError(
            f"기준점 개수와 PDR turn index 개수가 다릅니다.\n"
            f"node 기준점 개수: {len(turn_points)}\n"
            f"PDR is_turn 개수: {len(turn_idx)}"
        )

    matched_x = np.full(len(x_pos), np.nan)
    matched_y = np.full(len(y_pos), np.nan)

    # ============================================================
    # 1. turn point 구간별 projection
    # ============================================================
    for i in range(len(turn_idx) - 1):
        start_idx = turn_idx[i]
        end_idx = turn_idx[i + 1]

        if end_idx <= start_idx:
            continue

        # PDR 구간 좌표
        seg_x = x_pos[start_idx:end_idx + 1]
        seg_y = y_pos[start_idx:end_idx + 1]

        # 실제 node 시작점 / 끝점
        true_start = turn_points[i]
        true_end = turn_points[i + 1]

        # PDR 구간 내 누적 이동거리 계산
        dx = np.diff(seg_x)
        dy = np.diff(seg_y)
        step_dist = np.sqrt(dx ** 2 + dy ** 2)

        cum_dist = np.insert(np.cumsum(step_dist), 0, 0.0)
        total_dist = cum_dist[-1]

        ratio = cum_dist / total_dist

        # 실제 node 선분 위로 projection
        matched_seg = true_start + ratio[:, None] * (true_end - true_start)

        matched_x[start_idx:end_idx + 1] = matched_seg[:, 0]
        matched_y[start_idx:end_idx + 1] = matched_seg[:, 1]

    # ============================================================
    # 2. 첫 turn 이전 구간 처리
    # ============================================================
    first_idx = turn_idx[0]

    if first_idx > 0:
        matched_x[:first_idx] = turn_points[0, 0]
        matched_y[:first_idx] = turn_points[0, 1]

    # ============================================================
    # 3. 마지막 turn 이후 구간 처리
    # ============================================================
    last_idx = turn_idx[-1]

    if last_idx < len(x_pos) - 1:
        matched_x[last_idx:] = turn_points[-1, 0]
        matched_y[last_idx:] = turn_points[-1, 1]

    return matched_x, matched_y


def map_match_pdr_wifi_file(
    pdr_wifi_path,
    node_path,
    output_dir,
    output_name="pdr_WiFi_map_matched.csv",
    save_csv=True,
    show_plot=True
):
    """
    pdr_WiFi.csv와 node.csv를 입력받아
    PDR 좌표를 맵매칭 보정하고 CSV로 저장한다.

    출력 컬럼:
        time,
        original_x, original_y,
        x, y,
        is_turn,
        MAC...
    
    여기서 x, y는 보정된 좌표이다.
    """

    df_pdr = load_pdr_wifi(pdr_wifi_path)
    turn_points, df_node = load_turn_points(node_path)

    x_pos = df_pdr["x"].to_numpy(dtype=float)
    y_pos = df_pdr["y"].to_numpy(dtype=float)

    # is_turn == 1인 row index를 turn_idx로 사용
    turn_idx = df_pdr.index[df_pdr["is_turn"] == 1].to_numpy(dtype=int)

    print("========== 입력 정보 ==========")
    print("PDR-WiFi row 개수:", len(df_pdr))
    print("PDR turn index 개수:", len(turn_idx))
    print("node 기준점 개수:", len(turn_points))
    print("turn_idx:", turn_idx)
    print()

    matched_x, matched_y = map_matching_by_turn_points(
        x_pos=x_pos,
        y_pos=y_pos,
        turn_points=turn_points,
        turn_idx=turn_idx
    )

    df_result = df_pdr.copy()

    # 원본 좌표 보존
    df_result["original_x"] = df_result["x"]
    df_result["original_y"] = df_result["y"]

    # x, y를 보정 좌표로 교체
    df_result["x"] = matched_x
    df_result["y"] = matched_y

    # 컬럼 순서 정리
    base_cols = ["time", "original_x", "original_y", "x", "y", "is_turn"]
    other_cols = [col for col in df_result.columns if col not in base_cols]
    df_result = df_result[base_cols + other_cols]

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        df_result.to_csv(output_path, index=False, encoding="utf-8-sig")

        print("저장 완료:", output_path)
        print("출력 shape:", df_result.shape)

    if show_plot:
        plt.figure(figsize=(8, 6))

        plt.plot(
            df_pdr["x"],
            df_pdr["y"],
            marker="o",
            markersize=3,
            linewidth=1,
            alpha=0.5,
            label="Original PDR"
        )

        plt.plot(
            matched_x,
            matched_y,
            marker="o",
            markersize=3,
            linewidth=1,
            label="Map-Matched PDR"
        )

        plt.scatter(
            turn_points[:, 0],
            turn_points[:, 1],
            marker="*",
            s=200,
            label="Turn Points"
        )

        for i, (tx, ty) in enumerate(turn_points):
            plt.text(tx, ty, f"  TP{i}", fontsize=10)

        plt.title("PDR Map Matching Result")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

    return df_result


# ============================================================
# 실행
# ============================================================

df_map_matched = map_match_pdr_wifi_file(
    pdr_wifi_path=PDR_WIFI_PATH,
    node_path=NODE_PATH,
    output_dir=OUTPUT_DIR,
    output_name=OUTPUT_NAME,
    save_csv=True,
    show_plot=True
)

print(df_map_matched.head())