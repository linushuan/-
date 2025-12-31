#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STL 空氣品質數據後處理工具 (Step 2)
功能：
1. 自動讀取 output_results_v3/anomaly_csvs 中的距平檔案。
2. 執行 STL 分解移除日夜週期 (Period=24)。
3. 嚴格的斷點處理：
   - 缺測 <= 2小時：自動補值 (線性插值)。
   - 缺測 > 2小時：視為斷點 (保留 NaN)，並生成獨立報告。
4. 輸出結果至 output_results_v3/stl_processed_data 與 reports。
"""

import os
import glob
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import warnings

# 忽略 Statsmodels 在某些極端數據下的警告
warnings.filterwarnings("ignore")

# ===============================================================
#   路徑設定 (根據您提供的程式碼邏輯)
# ===============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
base_output_root = os.path.join(current_dir, "output_results_v3_nodb/")

# 輸入：上一階段產出的距平 CSV
INPUT_DIR = os.path.join(base_output_root, "anomaly_csvs")

# 輸出：新的資料夾
OUTPUT_DATA_DIR = os.path.join(base_output_root, "stl_processed_data")
OUTPUT_REPORT_DIR = os.path.join(base_output_root, "stl_reports")

# 建立輸出目錄
for d in [OUTPUT_DATA_DIR, OUTPUT_REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ===============================================================
#   參數設定
# ===============================================================

TOLERANCE_HOURS = 2       # 容忍缺測時數 (超過此數值則視為斷點)
STL_PERIOD = 24           # 週期 (小時資料為 24)
STL_SEASONAL = 13         # 季節平滑參數 (通常為奇數，13 是常用預設值)

# ===============================================================
#   核心處理函式
# ===============================================================

def process_series_stl(series, site, item, filename):
    """
    對單一序列進行：缺測檢查 -> 插值 -> STL分解 -> 斷點還原
    """
    gap_reports = []

    # 1. 偵測 NaN 分佈
    is_nan = series.isna()
    if is_nan.all():
        return series, gap_reports

    # 找出連續 NaN 的區塊
    # 利用 ne() 與 cumsum() 快速分組
    nan_groups = is_nan.ne(is_nan.shift()).cumsum()
    nan_blocks = series[is_nan].groupby(nan_groups)

    for _, block in nan_blocks:
        gap_len = len(block)
        if gap_len > TOLERANCE_HOURS:
            # 記錄超過容忍值的缺測
            start_t = block.index[0]
            end_t = block.index[-1]
            gap_reports.append({
                'file': filename,
                'site': site,
                'item': item,
                'type': '長時間缺測 (STL中斷)',
                'duration_hours': gap_len,
                'start_time': start_t.strftime('%Y-%m-%d %H:%M'),
                'end_time': end_t.strftime('%Y-%m-%d %H:%M')
            })

    # 2. 插值補值
    # limit=TOLERANCE_HOURS: 只補小洞，大洞留著 NaN
    series_interp = series.interpolate(method='linear', limit=TOLERANCE_HOURS)

    # 3. STL 準備
    # STL 不接受 NaN，對於剩下的大洞 (超過2小時的)，我們暫時填 0 (假設距平為0是平均態)
    # 以便算出整體的 Trend。算完後必須把這些洞挖回來。
    mask_large_gaps = series_interp.isna()
    series_for_stl = series_interp.fillna(0)

    # 4. 執行 STL
    # 資料長度至少要是週期的兩倍才能算
    if len(series_for_stl) > STL_PERIOD * 2:
        try:
            stl = STL(series_for_stl, period=STL_PERIOD, seasonal=STL_SEASONAL)
            res = stl.fit()

            # 核心公式：移除季節性 = 原始值 - 季節性成分
            # 這樣保留了 Trend (趨勢) + Residual (突發異常)
            deseasonalized = series_for_stl - res.seasonal

            # 5. 還原大斷點 (避免圖表誤導)
            final_series = deseasonalized.mask(mask_large_gaps)
        except Exception:
            # 如果數學運算失敗 (極少見，如數據變異數為0)，退回使用插值後的原數據
            final_series = series_interp
    else:
        final_series = series_interp

    return final_series, gap_reports

def process_file(file_path):
    file_name = os.path.basename(file_path)

    try:
        # 讀取上一階段的 CSV
        df = pd.read_csv(file_path)

        # 欄位檢查
        if 'anomaly' not in df.columns and 'value' in df.columns:
            target_col = 'value'
        elif 'anomaly' in df.columns:
            target_col = 'anomaly'
        else:
            print(f"⚠️ 跳過 {file_name}: 找不到 anomaly 或 value 欄位")
            return

        # --- 關鍵修正：支援混合格式時間 ---
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')

        # 移除時間解析失敗的行 (避免後面報錯)
        df = df.dropna(subset=['datetime'])

    except Exception as e:
        print(f"❌ 讀取錯誤 {file_name}: {e}")
        return

    # 建立該檔案的完整時間軸 (用於對齊)
    min_dt = df['datetime'].min()
    max_dt = df['datetime'].max()
    full_range = pd.date_range(start=min_dt, end=max_dt, freq='H')

    processed_rows = []
    all_reports = []

    # 針對每個 [測站, 測項] 分組處理
    grouped = df.groupby(['site', 'item'])

    for (site, item), group in grouped:
        # 重建索引以確保時間連續 (產生必要的 NaNs)
        group = group.set_index('datetime')
        # 消除重複索引 (避免 reindex 報錯)
        group = group[~group.index.duplicated(keep='first')]

        series = group[target_col].reindex(full_range)

        # 執行 STL 與斷點分析
        stl_series, reports = process_series_stl(series, site, item, file_name)

        all_reports.extend(reports)

        # 整理結果
        df_res = pd.DataFrame({
            'datetime': stl_series.index,
            'site': site,
            'item': item,
            'anomaly_stl': stl_series.values
        })
        processed_rows.append(df_res)

    # 輸出結果
    if processed_rows:
        final_df = pd.concat(processed_rows)
        final_df = final_df.dropna(subset=['anomaly_stl'])

        save_path = os.path.join(OUTPUT_DATA_DIR, f"stl_{file_name}")
        final_df.to_csv(save_path, index=False)

    # 輸出報告
    if all_reports:
        report_df = pd.DataFrame(all_reports)
        report_save_path = os.path.join(OUTPUT_REPORT_DIR, f"gap_report_{file_name}")
        report_df.to_csv(report_save_path, index=False, encoding='utf-8-sig')

# ===============================================================
#   主程式執行區
# ===============================================================

if __name__ == "__main__":
    # 檢查輸入目錄是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到輸入目錄: {INPUT_DIR}")
        print("請確認您是否已經執行過第一階段的程式，並產出了 anomaly_csvs 資料夾。")
        exit()

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

    if not files:
        print(f"⚠️  在 {INPUT_DIR} 中沒有找到 CSV 檔案。")
        exit()

    print(f"🚀 開始 STL 後處理 (去除日夜差異)...")
    print(f"📂 讀取來源: {INPUT_DIR}")
    print(f"📂 數據輸出: {OUTPUT_DATA_DIR}")
    print(f"📂 報告輸出: {OUTPUT_REPORT_DIR}")
    print("-" * 50)

    for f in tqdm(files, unit="file"):
        process_file(f)

    print("\n✅ STL 處理完成！")
    print(f"請至 {OUTPUT_REPORT_DIR} 查看斷點報告。")
