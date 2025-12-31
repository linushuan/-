#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高效能環境數據距平分析工具 (v3_NoDB - 無資料庫版)
改進內容：
 1. 移除所有資料庫 (SQLite) 寫入操作，純粹進行檔案與圖表輸出。
 2. 保留所有 v3 的繪圖與報告增強功能。
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

N_WORKERS = 6
# DB_PATH 已移除

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, "data")
output_root = os.path.join(current_dir, "output_results_v3_nodb") # 改名區隔
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

# 容錯範圍設定
items_info = {
    "AMB_TEMP": {"name": "環境溫度", "unit": "°C", "min": -15, "max": 55},
    "CO": {"name": "一氧化碳", "unit": "ppm", "min": 0, "max": 60},
    "NO": {"name": "一氧化氮", "unit": "ppb", "min": 0, "max": 600},
    "NO2": {"name": "二氧化氮", "unit": "ppb", "min": 0, "max": 600},
    "NOx": {"name": "氮氧化物", "unit": "ppb", "min": 0, "max": 1200},
    "O3": {"name": "臭氧", "unit": "ppb", "min": 0, "max": 600},
    "PM10": {"name": "PM10", "unit": "μg/m³", "min": 0, "max": 1200},
    "PM2.5": {"name": "PM2.5", "unit": "μg/m³", "min": 0, "max": 600},
    "RAINFALL": {"name": "降雨量", "unit": "mm", "min": 0, "max": 3000},
    "RH": {"name": "相對濕度", "unit": "%", "min": 0, "max": 100},
    "SO2": {"name": "二氧化硫", "unit": "ppb", "min": 0, "max": 300},
    "WD_HR": {"name": "平均風向", "unit": "degrees", "min": 0, "max": 360},
    "WIND_DIREC": {"name": "風向", "unit": "degrees", "min": 0, "max": 360},
    "WIND_SPEED": {"name": "風速", "unit": "m/s", "min": 0, "max": 120},
    "WS_HR": {"name": "平均風速", "unit": "m/s", "min": 0, "max": 120}
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
#   STEP 1: 預處理歷史平均值 (無 DB 寫入)
# ===============================================================

global_avg_lookup = None

def load_and_transform_averages():
    print("🔄 載入歷史平均值 (記憶體模式)...")
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
            df_melted['avg_value'] = pd.to_numeric(df_melted['avg_value'], errors='coerce').astype('float32')

            df_melted = df_melted.dropna(subset=['avg_value'])
            all_avg_list.append(df_melted[['item', 'site', 'day_of_year', 'hour', 'avg_value']])

        except Exception as e:
            print(f"Error loading avg {item}: {e}")

    if not all_avg_list: return None

    print("⚡ 合併記憶體中的歷史索引...")
    df_lookup = pd.concat(all_avg_list, ignore_index=True)
    df_lookup.set_index(['item', 'site', 'day_of_year', 'hour'], inplace=True)
    df_lookup.sort_index(inplace=True)
    return df_lookup

def init_worker(shared_df):
    global global_avg_lookup
    global_avg_lookup = shared_df

# ===============================================================
#   STEP 2 & 3: 核心處理與繪圖
# ===============================================================

def process_and_plot(file_path):
    global global_avg_lookup
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    report_list = []
    major_events = []

    try:
        # 1. 讀取原始資料
        df = pd.read_csv(file_path, dtype={'value': 'object'})
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        if df['datetime'].dropna().empty: return None

        min_dt = df['datetime'].min()
        max_dt = df['datetime'].max()
        full_range = pd.date_range(min_dt, max_dt, freq='h')

        # 2. 嚴格檢查 CSV 空值
        null_mask = df['value'].isna()
        if null_mask.any():
            null_rows = df[null_mask]
            null_summary = null_rows.groupby(['site', 'item']).size().reset_index(name='count')
            for _, row in null_summary.iterrows():
                report_list.append({
                    'file': file_name, 'site': row['site'], 'item': row['item'],
                    'type': '原始資料空值', 'detail': f"有 {row['count']} 筆記錄值為空白"
                })

        # 3. 數值轉換與非預期文字檢查
        df['numeric_value'] = pd.to_numeric(df['value'], errors='coerce')

        invalid_text_mask = df['numeric_value'].isna() & df['value'].notna()
        if invalid_text_mask.any():
            bad_rows = df[invalid_text_mask]
            for _, row in bad_rows.iterrows():
                report_list.append({
                    'file': file_name, 'site': row['site'], 'item': row['item'],
                    'type': '非預期文字', 'detail': f"Value: {row['value']}"
                })

        df = df.dropna(subset=['datetime', 'numeric_value'])
        df['value'] = df['numeric_value'].astype('float32')
        df = df.drop(columns=['numeric_value'])

        # 4. 數值邏輯檢查
        valid_df_list = []
        for item, group in df.groupby('item'):
            info = items_info.get(item)
            if info:
                out_of_bound = (group['value'] < info['min']) | (group['value'] > info['max'])
                if out_of_bound.any():
                    errs = group[out_of_bound]
                    for _, row in errs.head(5).iterrows():
                        report_list.append({
                            'file': file_name, 'site': row['site'], 'item': row['item'],
                            'type': '數值越界', 'detail': f"Value: {row['value']} (Limit: {info['min']}~{info['max']})"
                        })
                group = group[~out_of_bound]
            valid_df_list.append(group)

        if not valid_df_list: return None
        df_clean = pd.concat(valid_df_list)

        # 5. 缺失資料偵測 (Missing Timestamps)
        expected_len = len(full_range)
        for (site, item), group in df_clean.groupby(['site', 'item']):
            if len(group) < expected_len:
                existing_dts = set(group['datetime'])
                missing_count = expected_len - len(existing_dts)

                report_list.append({
                    'file': file_name, 'site': site, 'item': item,
                    'type': '時段資料遺失', 'detail': f"缺失 {missing_count} 小時"
                })

        # 6. 重大事件偵測 (全網無資料)
        existing_times = df_clean['datetime'].unique()
        missing_times = set(full_range) - set(existing_times)
        for t in missing_times:
            major_events.append({
                'file': file_name, 'datetime': t, 'event': '重大事件：全網斷訊'
            })

        # 7. 計算距平
        df_clean['day_of_year'] = df_clean['datetime'].dt.dayofyear.astype('int16')
        df_clean['hour'] = df_clean['datetime'].dt.hour.astype('int8')
        df_clean.set_index(['item', 'site', 'day_of_year', 'hour'], inplace=True)

        merged = df_clean.join(global_avg_lookup, how='inner')
        merged['anomaly'] = merged['value'] - merged['avg_value']
        merged = merged.reset_index()

        if merged.empty: return None

        # 8. 區域平均計算
        merged['region'] = merged['site'].map(site_to_region)
        valid_regions = merged.dropna(subset=['region'])

        region_avg = valid_regions.groupby(['datetime', 'item', 'region'])['anomaly'].mean().reset_index()
        region_avg['site'] = "AVG_" + region_avg['region']

        # 9. 繪圖與 CSV 輸出
        save_dir = os.path.join(img_dir, file_name)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        # 輸出 Raw Anomaly CSV (本地存檔保留)
        csv_out_path = os.path.join(anomaly_dir, f"anomaly_{file_name}.csv")
        merged[['datetime', 'site', 'item', 'anomaly']].to_csv(csv_out_path, index=False)

        plot_count = 0
        items = region_avg['item'].unique()

        for item in items:
            reg_data = region_avg[region_avg['item'] == item]
            if reg_data.empty: continue

            # Pivot: datetime x region
            pivot_reg = reg_data.pivot(index='datetime', columns='site', values='anomaly')
            pivot_reg = pivot_reg.reindex(full_range)

            # --- 繪圖 ---
            fig, ax = plt.subplots(figsize=(15, 8))
            region_colors = {'AVG_北': 'blue', 'AVG_中': 'green', 'AVG_南': 'red', 'AVG_東': 'orange'}

            for col in pivot_reg.columns:
                series = pivot_reg[col]
                if not series.dropna().empty:
                    color = region_colors.get(col, 'black')
                    ax.plot(pivot_reg.index, series, color=color, linewidth=2.5, label=col.replace('AVG_', '')+"部")

            item_dict = items_info.get(item, {'name': item, 'unit': ''})

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))

            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_title(f"{file_name} - {item_dict['name']} 區域平均距平", fontsize=16)
            ax.set_ylabel(f"距平 ({item_dict['unit']})")
            ax.legend(loc='upper right')
            plt.xticks(rotation=0)

            plt.savefig(os.path.join(save_dir, f"{item}.png"), dpi=100)
            plt.close(fig)
            plot_count += 1

        # 10. 輸出報告
        if report_list:
            pd.DataFrame(report_list).to_csv(os.path.join(report_dir, f"report_{file_name}.csv"), index=False, encoding='utf-8-sig')
        if major_events:
            pd.DataFrame(major_events).to_csv(os.path.join(report_dir, f"major_event_{file_name}.csv"), index=False, encoding='utf-8-sig')

        del df, df_clean, merged, pivot_reg
        gc.collect()

        return plot_count

    except Exception as e:
        with open(os.path.join(report_dir, f"CRITICAL_ERROR_{file_name}.txt"), "w") as f:
            f.write(str(e))
        return 0

# ===============================================================
#   主流程
# ===============================================================

def main():
    print("🚀 啟動工具 (v3_NoDB - 無資料庫純淨版)")

    # 不再需要 Manager 和 DB Writer Process
    df_avg_lookup = load_and_transform_averages()
    if df_avg_lookup is None:
        print("❌ 無歷史資料，結束程序。")
        return

    files = sorted(glob.glob(os.path.join(base_dir, "hourly_*.csv")))

    total_plots = 0
    start_time = time.time()

    # Init 只需要傳遞 DataFrame，不需要 Queue
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=init_worker, initargs=(df_avg_lookup,)) as executor:
        future_to_file = {executor.submit(process_and_plot, f): f for f in files}

        for future in tqdm(as_completed(future_to_file), total=len(files), desc="處理進度", unit="file"):
            try:
                res = future.result()
                if res: total_plots += res
            except Exception as e:
                print(f"Worker Error: {e}")
            gc.collect()

    del df_avg_lookup
    gc.collect()

    print(f"\n✅ 完成！耗時: {time.time() - start_time:.2f}秒, 產出 {total_plots} 張圖表。")

if __name__ == '__main__':
    main()
