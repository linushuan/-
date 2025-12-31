#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
長年平均值週報繪圖工具 (修正版)
修正內容：解決 Pandas FutureWarning (unit='H' -> unit='h')
"""

import os
import glob
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

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, "data")
output_root = os.path.join(current_dir, "output_longterm_weekly")

# 輸出結構： output_longterm_weekly / [測項名稱] / Week_XX.png
if not os.path.exists(output_root):
    os.makedirs(output_root)

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
#   核心處理邏輯
# ===============================================================

def process_item_weekly(item):
    """
    處理單一測項：讀取 -> 切分週 -> 繪圖
    """
    item_lower = item.lower()
    csv_path = os.path.join(base_dir, f"{item_lower}_hourly_avg_fast.csv")

    if not os.path.exists(csv_path):
        return 0

    try:
        # 1. 讀取資料
        df = pd.read_csv(csv_path)
        if '測站' in df.columns: df = df.rename(columns={'測站': 'site'})

        # 轉換 day_hour (格式如 "1_0" 到 "365_23")
        df_melted = df.melt(id_vars=['site'], var_name='day_hour', value_name='avg_value')
        temp_split = df_melted['day_hour'].str.split('_', expand=True)
        df_melted['day_of_year'] = temp_split[0].astype(int)
        df_melted['hour'] = temp_split[1].astype(int)
        df_melted['avg_value'] = pd.to_numeric(df_melted['avg_value'], errors='coerce')

        # 移除無數值的資料
        df_melted = df_melted.dropna(subset=['avg_value'])

        # 加入區域資訊
        df_melted['region'] = df_melted['site'].map(site_to_region)
        df_melted = df_melted.dropna(subset=['region']) # 過濾掉非四大區的測站

        # 建立該測項的輸出資料夾
        item_out_dir = os.path.join(output_root, item)
        os.makedirs(item_out_dir, exist_ok=True)

        info = items_info.get(item, {"name": item, "unit": ""})
        plot_count = 0

        # 2. 迴圈處理每一週 (1~53週)
        for week_num in range(1, 54):
            start_day = (week_num - 1) * 7 + 1
            end_day = week_num * 7

            # 處理最後一週邊界
            if start_day > 365: break
            if end_day > 365: end_day = 365

            # 篩選該週資料
            mask = (df_melted['day_of_year'] >= start_day) & (df_melted['day_of_year'] <= end_day)
            week_data = df_melted[mask].copy()

            if week_data.empty: continue

            # 3. 建立虛擬時間軸 (Dummy Datetime)
            origin_date = pd.Timestamp("2023-01-01")

            # [修正點] 將 unit='H' 改為 unit='h'
            week_data['plot_time'] = origin_date + pd.to_timedelta(week_data['day_of_year'] - 1, unit='D') + pd.to_timedelta(week_data['hour'], unit='h')

            # 計算區域平均
            region_avg = week_data.groupby(['plot_time', 'region'])['avg_value'].mean().reset_index()

            # 4. 繪圖
            fig, ax = plt.subplots(figsize=(15, 8))

            # 背景：個別測站 (灰色)
            for site, group in week_data.groupby('site'):
                group = group.sort_values('plot_time')
                ax.plot(group['plot_time'], group['avg_value'], color='gray', alpha=0.15, linewidth=1)

            # 前景：區域平均 (彩色)
            region_colors = {'北': 'blue', '中': 'green', '南': 'red', '東': 'orange'}
            for region in ['北', '中', '南', '東']:
                reg_group = region_avg[region_avg['region'] == region].sort_values('plot_time')
                if not reg_group.empty:
                    ax.plot(reg_group['plot_time'], reg_group['avg_value'],
                            color=region_colors.get(region, 'black'),
                            linewidth=2.5,
                            label=f"{region}部平均")

            # 設定標題與標籤
            week_str = f"第 {week_num:02d} 週 (Day {start_day} - {end_day})"
            ax.set_title(f"長年平均分布 - {info['name']} ({item}) - {week_str}", fontsize=16)
            ax.set_ylabel(f"{info['name']} ({info['unit']})")

            # X 軸格式化 (顯示 月/日 時:分)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            ax.set_xlim(week_data['plot_time'].min(), week_data['plot_time'].max())

            ax.legend(loc='upper right')
            ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.3)

            # 存檔
            save_path = os.path.join(item_out_dir, f"Week_{week_num:02d}_{item}.png")
            plt.savefig(save_path, dpi=100)
            plt.close(fig)
            plot_count += 1

        return plot_count

    except Exception as e:
        print(f"Error processing {item}: {e}")
        return 0

# ===============================================================
#   主程式
# ===============================================================

def main():
    print("🚀 啟動長年平均週報繪圖工具 (修正版)...")
    print(f"📂 讀取資料來源: {base_dir}")
    print(f"📂 輸出圖片路徑: {output_root}")

    items = list(items_info.keys())
    total_plots = 0

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_item = {executor.submit(process_item_weekly, item): item for item in items}

        for future in tqdm(as_completed(future_to_item), total=len(items), desc="繪製各測項週報", unit="item"):
            total_plots += future.result()

    print(f"\n✅ 全部完成！共產出 {total_plots} 張圖表。")
    print(f"   請查看資料夾: {output_root}")

if __name__ == '__main__':
    main()
