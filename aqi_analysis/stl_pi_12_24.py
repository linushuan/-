#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: STL 數據繪圖工具
功能：
1. 讀取 output_results_v3/stl_processed_data 中的資料。
2. 計算「區域平均」(北中南東)。
3. 繪製時序圖 (Matplotlib)，遇缺測自動斷線。
"""

import os
import glob
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')  # 不顯示視窗，直接存檔
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import fontManager, FontProperties
from tqdm import tqdm

# ===============================================================
#   設定區
# ===============================================================

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
base_output_root = os.path.join(current_dir, "output_results_v3_nodb/")

INPUT_DIR = os.path.join(base_output_root, "stl_processed_data_51")
OUTPUT_IMG_DIR = os.path.join(base_output_root, "stl_plots_51")

# 區域定義 (用於歸類測站)
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

# 反向查表 (測站 -> 區域)
site_to_region = {}
for region, sites in areas.items():
    for site in sites:
        site_to_region[site] = region

# 測項顯示資訊 (單位與名稱)
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
    "WD_HR": {"name": "平均風向", "unit": "deg"},
    "WIND_SPEED": {"name": "風速", "unit": "m/s"},
    "WS_HR": {"name": "平均風速", "unit": "m/s"}
}

# ===============================================================
#   工具函式
# ===============================================================

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

def plot_file(file_path):
    file_name = os.path.basename(file_path)
    # 移除 "stl_" 前綴和 ".csv" 後綴，取得原始檔名標識
    clean_name = file_name.replace("stl_", "").replace(".csv", "")

    # 建立該檔案的圖片輸出目錄
    save_dir = os.path.join(OUTPUT_IMG_DIR, clean_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. 讀取資料
    try:
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
    except Exception as e:
        print(f"❌ 讀取失敗 {file_name}: {e}")
        return

    # 2. 標記區域
    df['region'] = df['site'].map(site_to_region)
    # 移除不在四大區域內的測站 (如果有)
    df = df.dropna(subset=['region'])

    # 3. 計算區域平均
    # GroupBy: 時間、項目、區域 -> 取 anomaly_stl 的平均
    region_avg = df.groupby(['datetime', 'item', 'region'])['anomaly_stl'].mean().reset_index()

    # 4. 針對每個測項畫圖
    items = region_avg['item'].unique()

    # 顏色定義
    colors = {'北': 'blue', '中': 'green', '南': 'red', '東': 'orange'}
    region_order = ['北', '中', '南', '東']

    for item in items:
        data_item = region_avg[region_avg['item'] == item]
        if data_item.empty: continue

        # Pivot 轉置: Index=時間, Columns=區域, Values=數值
        # 這樣 Matplotlib 才能畫多條線
        pivot_df = data_item.pivot(index='datetime', columns='region', values='anomaly_stl')

        # 確保時間軸完整 (這樣 Matplotlib 才能正確處理斷點)
        # 這裡不需要 reindex 插補 NaN，因為 pivot 後原本沒資料的地方自然就是 NaN

        # --- 開始繪圖 ---
        fig, ax = plt.subplots(figsize=(15, 8))

        has_data = False
        for reg in region_order:
            if reg in pivot_df.columns:
                series = pivot_df[reg]
                # 檢查是否全空
                if not series.dropna().empty:
                    has_data = True
                    # Matplotlib 遇到 NaN 會自動斷開線條
                    ax.plot(series.index, series,
                            color=colors.get(reg, 'black'),
                            label=f"{reg}部",
                            linewidth=2,
                            alpha=0.8)

        if not has_data:
            plt.close(fig)
            continue

        # 設定標題與標籤
        info = items_info.get(item, {'name': item, 'unit': ''})
        title_str = f"{clean_name} - {info['name']} 區域平均 (STL去除日夜變化)"

        ax.set_title(title_str, fontsize=18, fontweight='bold', pad=15)
        ax.set_ylabel(f"距平值 ({info['unit']})", fontsize=14)

        # X 軸格式化 (自動日期)
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter('%m/%d\n%H:%M')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=12)

        # 輔助線 (0線)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

        # 圖例
        ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

        # 格線
        ax.grid(True, which='both', linestyle=':', alpha=0.4)

        # 存檔
        out_path = os.path.join(save_dir, f"{item}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close(fig)

# ===============================================================
#   主程式
# ===============================================================

if __name__ == "__main__":
    set_chinese_font()

    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到輸入目錄: {INPUT_DIR}")
        exit()

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

    if not files:
        print("⚠️ 無資料可繪圖")
        exit()

    print(f"🚀 開始繪圖 (共 {len(files)} 個檔案)...")
    print(f"📂 圖片輸出至: {OUTPUT_IMG_DIR}")

    for f in tqdm(files, unit="file"):
        plot_file(f)

    print("\n✅ 繪圖完成！")
