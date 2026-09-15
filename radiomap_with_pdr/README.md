# radiomap_with_pdr — 핑거프린팅 RP 3m 간격 (AI융합연구원 과제용)

PDR(보행자 추측 항법) 기반으로 수집한 Wi-Fi RSSI 데이터를 이용해, 도면 위에
**3m 간격 기준점(RP)** 라디오맵을 생성하는 파이프라인입니다.

> **취지**: 성능 최적화가 아니라 지도교수님이 지정한 원본 알고리즘을 그대로
> 재현하는 것이 목표입니다. 핑거프린팅(KNN, K=3)용 RP는 `4_1`로 3m 간격
> 보간해서 생성하며, `4_2`(ROI 격자 방식, SC/Sequence-Chaining 용)는 이
> 파이프라인과 무관하므로 건드리지 않았습니다.

## 폴더 구조

```
radiomap_with_pdr/
├── data/                              # 원본 raw 데이터
│   ├── 공학관_3층.png                  # 도면 이미지 (캘리브레이션/RP 클릭용)
│   ├── imu.csv                        # IMU 원본 (Arduino Nano 33 BLE Rev2, 50Hz)
│   ├── rssi_2ghz.csv                  # 2.4GHz Wi-Fi RSSI 원본 (ESP32-S3 ×3)
│   └── rssi_5ghz.csv                  # 5GHz Wi-Fi RSSI 원본 (모니터 모드, DFS 채널 포함)
├── 0_pdr_디버깅용.py                   # PDR 궤적 디버깅용 스크립트
├── 0_testPDR.ipynb                    # PDR 알고리즘 테스트 노트북
├── 1_pdrWiFiMerge.py                  # IMU + WiFi RSSI 병합, PDR 궤적 계산
├── 2_getTruePath.py                   # 도면 위에서 실제 이동 경로(정답) 클릭 수집
├── 3_pdrMapMatching.py                # PDR 궤적을 도면 코너에 맞춰 보정(map-matching)
├── 4_1_getRP_fingerprinting.py        # ★ 핑거프린팅용 RP 생성 (3m 간격, pair 보간 방식)
├── 4_2_getRP_SC.py                    # SC(Sequence Chaining)용 RP 생성 — 이 워크플로우와 무관, 미사용
├── 5_makeRadioMap.py                  # RP + map-matched WiFi 데이터 → 최종 라디오맵(radio_map.csv)
├── 6_SC.ipynb                         # SC 관련 노트북
├── 7_라디오맵2D시각화.py                # 라디오맵 2D 시각화
├── 7_라디오맵3D시각화.py                # 라디오맵 3D 시각화
└── README.md
```

`temp_data/`(중간 산출물)와 `radio_map.csv`(최종 결과)는 스크립트를 실행하면
자동 생성되므로 저장소에는 포함하지 않았습니다.

## 실행 순서

핑거프린팅(3m RP) 기준 실행 순서는 다음과 같습니다.

1. `python 1_pdrWiFiMerge.py` — `data/imu.csv`, `rssi_2ghz.csv`, `rssi_5ghz.csv`를 병합해
   `temp_data/pdr_WiFi.csv` 생성
2. `python 2_getTruePath.py` — 도면 위에서 실제 경로(코너 등) 클릭 → `temp_data/node.csv` 생성
3. `python 3_pdrMapMatching.py` — PDR 궤적을 코너에 맞춰 보정 →
   `temp_data/pdr_WiFi_map_matched.csv` 생성
4. `python 4_1_getRP_fingerprinting.py` — 도면 위에서 캘리브레이션 2점 + 경로 pair를
   클릭하면 **3m 간격**으로 보간된 RP를 `temp_data/rp_pos.csv`에 저장
5. `python 5_makeRadioMap.py` — `rp_pos.csv` + `pdr_WiFi_map_matched.csv`를 조합해
   `radio_map.csv` 생성 (최종 라디오맵)

`4_2_getRP_SC.py`는 이 워크플로우에서 사용하지 않습니다.

## 요구 사항

```
opencv-python
numpy
pandas
pillow
matplotlib
scipy
```

`4_1`, `4_2`, `2_getTruePath.py`는 OpenCV 창(GUI)에서 마우스 클릭으로 좌표를
입력받으므로, 원격 서버가 아닌 **화면이 있는 로컬 환경**에서 실행해야 합니다.

## 주요 설정값

| 스크립트 | 상수 | 값 | 의미 |
|---|---|---|---|
| `4_1_getRP_fingerprinting.py` | `INTERPOLATION_INTERVAL_M` | `3.0` | RP 간격 (m) |
| `5_makeRadioMap.py` | `GRID_STEP_M` | `3.0` | 이웃 격자 탐색 간격 (RP 간격과 일치) |
| `5_makeRadioMap.py` | `SEARCH_RADIUS_M` | `0.5` | RP 근접 판정 반경 (원본 값 유지, RP 간격에 비례해 키우지 않음) |
