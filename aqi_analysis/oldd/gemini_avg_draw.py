#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高效能距平值計算與繪圖工具 (最終版)
特色：
 1. 獨立報告：每個檔案缺失情形獨立輸出 CSV。
 2. 斷線繪圖：缺失資料處自動斷開不連線。
 3. 區域平均：部分缺失仍計算平均，全缺失則為空。
 4. 進度條：恢復詳細顯示模式 (顯示任務名稱與檔案單位)。
"""

import os
import glob
import time
import gc
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import fontManager, FontProperties
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ===============================================================
#   CONFIG 設定區
# ===============================================================

N_WORKERS = 4  # 維持穩定設定

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, "data")
output_root = os.path.join(current_dir, "output_results")
anomaly_dir = os.path.join(output_root, "anomaly_csvs")
img_dir = os.path.join(output_root, "anomaly_pictures")
report_dir = os.path.join(output_root, "reports")

for d in [base_dir, anomaly_dir, img_dir, report_dir]:
    os.makedirs(d, exist_ok=True)

# 區域定義
areas = {
    "北": ["三重", "中壢", "中山", "冬山", "古亭", "土城", "基隆", "士林",
           "大園", "宜蘭", "平鎮", "新店", "新竹", "新莊", "松山", "板橋",
           "林口", "桃園", "永和", "汐止", "淡水", "湖口", "竹東", "菜寮",
           "萬華", "萬里", "觀音", "陽明", "龍潭"],
    "中": ["三義", "二林", "南投", "大里", "彰化", "忠明", "沙鹿",
           "線西", "苗栗", "西屯", "豐原", "頭份"],
    "南": ["仁武", "前金", "前鎮", "善化", "嘉義", "大寮", "安南",
           "小港", "屏東", "崙背", "左營", "復興", "恆春", "斗六",
           "新港", "新營", "朴子", "林園", "楠梓", "橋頭", "潮州",
           "美濃", "臺南", "臺西", "鳳山"],
    "東": ["臺東", "花蓮"]
}

site_to_region = {}
for region, sites in areas.items():
    for site in sites:
        site_to_region[site] = region

items_info = {
    "AMB_TEMP": {"name": "環境溫度", "unit": "°C"},
    "CO": {"name": "一氧化碳", "unit": "ppm"},
    "NO": {"name": "一氧化氮", "unit": "ppb"},
    "NO2": {"name": "二氧化氮", "unit": "ppb"},
    "NOx": {"name": "氮氧化物", "unit": "ppb"},
    "O3": {"name": "臭氧", "unit": "ppb"},
    "PM10": {"name": "PM10", "unit": "μg/m³"},
    "PM2.5": {"name": "PM2.5", "unit": "μg/m³"},
    "RAINFALL": {"name": "降雨量", "unit": "mm"},
    "RH": {"name": "相對濕度", "unit": "%"},
    "SO2": {"name": "二氧化硫", "unit": "ppb"},
    "WD_HR": {"name": "平均風向", "unit": "degrees"},
    "WIND_DIREC": {"name": "風向", "unit": "degrees"},
    "WIND_SPEED": {"name": "風速", "unit": "m/s"},
    "WS_HR": {"name": "平均風速", "unit": "m/s"}
}

def set_chinese_font():
    p = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
    try:
        fontManager.addfont(p)
        prop = FontProperties(fname=p)
        if prop.get_name():
            mpl.rcParams['font.family'] = 'sans-serif'
            mpl.rcParams['font.sans-serif'] = [prop.get_name()]
    except Exception:
        pass
set_chinese_font()

# ===============================================================
#   STEP 1: 預處理歷史平均值 (Global Shared)
# ===============================================================

global_avg_lookup = None

def load_and_transform_averages():
    print("🔄 載入歷史平均值...")
    all_avg_list = []
    for item in items_info.keys():
        avg_file = os.path.join(base_dir, f"{item.lower()}_hourly_avg_fast.csv")
        if not os.path.exists(avg_file): continue
        try:
            df = pd.read_csv(avg_file)
            if '測站' in df.columns: df = df.rename(columns={'測站': 'site'})
            if 'site' not in df.columns: continue

            df_melted = df.melt(id_vars=['site'], var_name='day_hour', value_name='avg_value')
            temp_split = df_melted['day_hour'].str.split('_', expand=True)

            df_melted['day_of_year'] = temp_split[0].astype('int16')
            df_melted['hour'] = temp_split[1].astype('int8')
            df_melted['item'] = item
            df_melted['avg_value'] = df_melted['avg_value'].astype('float32')

            df_melted = df_melted.drop(columns=['day_hour']).dropna(subset=['avg_value'])
            all_avg_list.append(df_melted)
            del df, temp_split
        except Exception:
            pass

    if not all_avg_list: return None
    print("⚡ 合併索引中...")
    df_lookup = pd.concat(all_avg_list, ignore_index=True)
    df_lookup['site'] = df_lookup['site'].astype(str)
    df_lookup['item'] = df_lookup['item'].astype(str)

    del all_avg_list
    gc.collect()
    return df_lookup

def init_worker(shared_df):
    global global_avg_lookup
    global_avg_lookup = shared_df

# ===============================================================
#   STEP 2: 核心處理
# ===============================================================

def process_single_file(file_path):
    global global_avg_lookup
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    missing_report = []

    try:
        # 1. 讀取
        df = pd.read_csv(file_path, usecols=['datetime', 'site', 'item', 'value'])
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        if df.empty: return None

        df['site'] = df['site'].astype(str)
        df['item'] = df['item'].astype(str)
        df['value'] = pd.to_numeric(df['value'], errors='coerce').astype('float32')

        # --- 2. 偵測原始資料缺失 (Type A) ---
        min_time, max_time = df['datetime'].min(), df['datetime'].max()
        full_time_range = pd.date_range(start=min_time, end=max_time, freq='h')
        expected_len = len(full_time_range)

        # 使用低記憶體 Loop
        for (site, item), group in df.groupby(['site', 'item']):
            if len(group) < expected_len:
                missing_report.append({
                    'file': file_name,
                    'site': site,
                    'item': item,
                    'missing_type': '原始資料缺失',
                    'count': expected_len - len(group),
                    'note': '該時段無數據'
                })

        # --- 3. 準備計算 ---
        df['day_of_year'] = df['datetime'].dt.dayofyear.astype('int16')
        df['hour'] = df['datetime'].dt.hour.astype('int8')

        # --- 4. 偵測歷史平均缺失 (Type B) & 計算 ---
        merged = pd.merge(
            df,
            global_avg_lookup,
            on=['item', 'site', 'day_of_year', 'hour'],
            how='left',
            indicator=True
        )

        # 記錄歷史平均缺失
        missing_avg = merged[merged['_merge'] == 'left_only']
        if not missing_avg.empty:
            avg_summary = missing_avg.groupby(['site', 'item']).size().reset_index(name='count')
            for _, row in avg_summary.iterrows():
                missing_report.append({
                    'file': file_name,
                    'site': row['site'],
                    'item': row['item'],
                    'missing_type': '歷史平均缺失',
                    'count': row['count'],
                    'note': '無歷史平均值'
                })
        del missing_avg

        # --- 5. 輸出 Report (獨立檔案) ---
        if missing_report:
            df_rep = pd.DataFrame(missing_report)
            rep_path = os.path.join(report_dir, f"report_{file_name}.csv")
            df_rep.to_csv(rep_path, index=False, encoding='utf-8-sig')

        # --- 6. 計算距平 (僅計算資料完整的) ---
        final_data = merged[merged['_merge'] == 'both'].copy()
        del merged

        final_data['anomaly'] = final_data['value'] - final_data['avg_value']

        # dropna 確保沒有計算出 NaN 的結果
        final_data = final_data.dropna(subset=['anomaly'])

        if final_data.empty: return None

        # 輸出 Raw
        df_out = final_data[['datetime', 'site', 'item', 'anomaly']].sort_values(['datetime', 'item', 'site'])
        raw_path = os.path.join(anomaly_dir, f"anomaly_{file_name}.csv")
        df_out.to_csv(raw_path, index=False, encoding='utf-8-sig')

        # 輸出 Region Avg
        final_data['region'] = final_data['site'].map(site_to_region)
        final_data = final_data.dropna(subset=['region'])

        if final_data.empty:
            del final_data, df_out
            return (raw_path, None)

        # 區域平均計算 (SkipNA=True, 部分缺失算有的)
        region_avg = final_data.groupby(['datetime', 'item', 'region'])['anomaly'].mean().reset_index()
        region_avg['site'] = "AVG_" + region_avg['region']
        region_avg = region_avg.drop(columns=['region'])

        reg_path = os.path.join(anomaly_dir, f"region_avg_{file_name}.csv")
        region_avg.to_csv(reg_path, index=False, encoding='utf-8-sig')

        del final_data, region_avg, df_out
        gc.collect()

        return (raw_path, reg_path)

    except Exception as e:
        with open(os.path.join(report_dir, f"ERROR_{file_name}.txt"), "w") as f:
            f.write(str(e))
        return None

# ===============================================================
#   STEP 3: 繪圖邏輯 (斷線處理)
# ===============================================================

def plot_file_result(raw_csv, region_csv):
    if not region_csv or not os.path.exists(region_csv): return 0
    try:
        df_raw = pd.read_csv(raw_csv, usecols=['datetime', 'site', 'item', 'anomaly'])
        df_region = pd.read_csv(region_csv)

        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_region['datetime'] = pd.to_datetime(df_region['datetime'])

        file_name = os.path.basename(raw_csv).replace("anomaly_", "").replace(".csv", "")
        save_dir = os.path.join(img_dir, file_name)
        os.makedirs(save_dir, exist_ok=True)

        # 取得該檔案的全域時間範圍 (用於 Reindex 斷線)
        min_t = min(df_raw['datetime'].min(), df_region['datetime'].min())
        max_t = max(df_raw['datetime'].max(), df_region['datetime'].max())
        full_time_idx = pd.date_range(start=min_t, end=max_t, freq='h')

        plot_count = 0
        items = df_region['item'].unique()

        for item in items:
            item_info = items_info.get(item, {'name': item, 'unit': ''})

            raw_data = df_raw[df_raw['item'] == item]
            reg_data = df_region[df_region['item'] == item]

            if reg_data.empty: continue

            fig, ax = plt.subplots(figsize=(15, 8))

            # --- 畫背景測站 (灰線，斷線處理) ---
            for site, group in raw_data.groupby('site'):
                group_reindexed = group.set_index('datetime').reindex(full_time_idx)
                ax.plot(group_reindexed.index, group_reindexed['anomaly'],
                        color='gray', alpha=0.15, linewidth=1)

            # --- 畫區域平均 (彩線，斷線處理) ---
            region_colors = {'AVG_北': 'blue', 'AVG_中': 'green', 'AVG_南': 'red', 'AVG_東': 'orange'}
            for site, group in reg_data.groupby('site'):
                color = region_colors.get(site, 'black')
                group_reindexed = group.set_index('datetime').reindex(full_time_idx)

                ax.plot(group_reindexed.index, group_reindexed['anomaly'],
                        color=color, linewidth=2.5,
                        label=site.replace('AVG_', '')+"部")

            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_title(f"{file_name} - {item_info['name']} ({item})", fontsize=16)
            ax.set_ylabel(f"距平值 ({item_info['unit']})")
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            ax.legend(loc='upper right')

            out_path = os.path.join(save_dir, f"ANOMALY_{item}.png")
            plt.savefig(out_path, dpi=100)
            plt.close(fig)
            plt.clf()
            plot_count += 1

        del df_raw, df_region
        gc.collect()
        return plot_count
    except Exception as e:
        print(f"Plot Error: {e}")
        return 0

# ===============================================================
#   主流程
# ===============================================================

def main():
    print("🚀 啟動工具 (獨立報告 & 斷線繪圖版)")

    df_avg_lookup = load_and_transform_averages()
    if df_avg_lookup is None: return

    files = sorted(glob.glob(os.path.join(base_dir, "hourly_*.csv")))
    processed_results = []

    # [復原進度條風格] 加入 desc 和 unit
    print("⚡ 計算與生成報告中...")
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=init_worker, initargs=(df_avg_lookup,)) as executor:
        future_to_file = {executor.submit(process_single_file, f): f for f in files}

        for future in tqdm(as_completed(future_to_file), total=len(files), desc="計算與生成報告", unit="file"):
            result = future.result()
            if result: processed_results.append(result)
            gc.collect()

    del df_avg_lookup
    gc.collect()

    print("\n🎨 繪圖中 (自動斷開缺失部分)...")
    total_plots = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_plot = {executor.submit(plot_file_result, raw, region): raw for raw, region in processed_results}

        # [復原進度條風格] 加入 desc 和 unit
        for future in tqdm(as_completed(future_to_plot), total=len(processed_results), desc="繪製圖表", unit="file"):
            total_plots += future.result()
            gc.collect()

    print(f"\n✅ 完成！共產出 {total_plots} 張圖表。")

if __name__ == '__main__':
    main()
