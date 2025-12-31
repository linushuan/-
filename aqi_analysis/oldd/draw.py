import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys
import platform
from datetime import datetime
import matplotlib.dates as mdates
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.font_manager import fontManager
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

# === 本地端環境設定 ===

# 設定 matplotlib 風格
plt.style.use('seaborn-v0_8-whitegrid') # 使用較美觀的樣式
mpl.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

# 設定非互動模式 (批次繪圖時不需跳出視窗)
plt.ioff()
mpl.use('Agg') 

# === 自動設定中文字體 ===
def set_chinese_font():
    """
    根據作業系統自動尋找可用的中文字體，針對 Linux 提供較完整的候選字體清單。
    若未發現常見中文字體，會提示使用者安裝建議。
    """
    system_name = platform.system()
    # 候選字體清單（依優先度）
    if system_name == "Windows":
        font_candidates = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
    elif system_name == "Darwin":
        font_candidates = ["PingFang TC", "Heiti TC", "Arial Unicode MS"]
    else:  # Linux
        # 常見於各發行版或可透過套件安裝的字體
        font_candidates = [
            "Noto Sans CJK TC",   # Noto CJK 繁體（若安裝，優先）
            "Noto Sans CJK",      # 有時以此名稱出現
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "AR PL KaitiM GB",    # 部分系統可用
            "AR PL UKai CN",
            "Droid Sans Fallback",
            "DejaVu Sans"         # 通用字型（對 CJK 支援有限，但有時可用）
        ]

    available_fonts = {f.name for f in fontManager.ttflist}

    for f in font_candidates:
        if f in available_fonts:
            mpl.rcParams['font.family'] = f
            mpl.rcParams['font.sans-serif'] = [f]
            print(f"✅ 已設定中文字體: {f}")
            return

    # 若沒有找到，嘗試以常見路徑尋找字體檔（較進階）
    common_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKtc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for p in common_paths:
        if os.path.exists(p):
            try:
                from matplotlib import font_manager
                prop = font_manager.FontProperties(fname=p)
                mpl.rcParams['font.family'] = prop.get_name()
                print(f"✅ 已以檔案路徑設定字體: {p} (font name: {prop.get_name()})")
                return
            except Exception:
                continue

    # 若仍無法設定，給使用者安裝建議
    print("⚠️ 未偵測到適合的中文字體，圖表中文字可能會顯示為方塊。")
    print("   建議安裝常見中文字體，例如 (Debian/Ubuntu):")
    print("     sudo apt update && sudo apt install fonts-noto-cjk fonts-wqy-microhei")
    print("   或手動安裝「Microsoft JhengHei」並將字體檔放到 ~/.local/share/fonts 或 /usr/share/fonts")

# 呼叫字體設定
set_chinese_font()

# === Config (請依據你的資料夾結構修改這裡) ===
# 取得目前腳本所在的路徑
current_dir = os.path.dirname(os.path.abspath(__file__))

# 假設 csv 檔案放在腳本旁邊的 'data' 資料夾內
# 結構:
#   - plot_aqi.py
#   - data/
#       - hourly_2015....csv
base_dir = os.path.join(current_dir, "data") 

# 輸出圖片的路徑
output_dir = os.path.join(current_dir, "output_pictures")
os.makedirs(output_dir, exist_ok=True)

print(f"📂 資料來源目錄: {base_dir}")
print(f"📂 圖片輸出目錄: {output_dir}")

# === 測項資訊 ===
items_info = {
    "AMB_TEMP": {"name": "環境溫度", "unit": "°C", "color": "#FF6B6B"},
    "CO": {"name": "一氧化碳", "unit": "ppm", "color": "#4ECDC4"},
    "NO": {"name": "一氧化氮", "unit": "ppb", "color": "#45B7D1"},
    "NO2": {"name": "二氧化氮", "unit": "ppb", "color": "#96CEB4"},
    "NOx": {"name": "氮氧化物", "unit": "ppb", "color": "#FFEAA7"},
    "O3": {"name": "臭氧", "unit": "ppb", "color": "#DFE6E9"},
    "PM10": {"name": "懸浮微粒", "unit": "μg/m³", "color": "#A29BFE"},
    "PM2.5": {"name": "細懸浮微粒", "unit": "μg/m³", "color": "#FD79A8"},
    "RAINFALL": {"name": "降雨量", "unit": "mm", "color": "#74B9FF"},
    "RH": {"name": "相對濕度", "unit": "%", "color": "#81ECEC"},
    "SO2": {"name": "二氧化硫", "unit": "ppb", "color": "#FAB1A0"},
    "WD_HR": {"name": "風向", "unit": "degrees", "color": "#00B894"},
    "WIND_DIREC": {"name": "風向", "unit": "degrees", "color": "#00CEC9"},
    "WIND_SPEED": {"name": "風速", "unit": "m/s", "color": "#0984E3"},
    "WS_HR": {"name": "風速", "unit": "m/s", "color": "#6C5CE7"}
}


def plot_item_timeseries_single_station(data_tuple):
    """
    繪製單一站點、單一測項的時間序列圖
    (改為接受 tuple 以支援平行處理)
    """
    try:
        item_df, item, site, show_markers, figsize, alpha, output_path = data_tuple

        if item_df.empty:
            return None

        # 建立圖表
        fig, ax = plt.subplots(figsize=figsize)

        # 取得測項資訊
        info = items_info.get(item, {"name": item, "unit": "", "color": "#95A5A6"})

        # 繪製線條
        if show_markers:
            ax.plot(item_df['datetime'], item_df['value'],
                   marker='o', markersize=4, linestyle='-',
                   linewidth=2, alpha=alpha, color=info['color'])
        else:
            ax.plot(item_df['datetime'], item_df['value'],
                   linestyle='-', linewidth=2, alpha=alpha,
                   color=info['color'])

        # 設定標題和標籤
        # 轉換時間格式確保可讀性
        start_time = item_df['datetime'].min().strftime('%Y-%m-%d')
        end_time = item_df['datetime'].max().strftime('%Y-%m-%d')
        time_range = f"{start_time} ~ {end_time}"
        
        ax.set_title(f"{site} - {info['name']} ({item})\n{time_range}",
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('時間', fontsize=12)
        ax.set_ylabel(f"{info['name']} ({info['unit']})", fontsize=12)

        # 格式化 x 軸日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # 旋轉日期標籤以免重疊
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        # 網格線
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        # 儲存
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path

    except Exception as e:
        print(f"❌ Error plotting {site} - {item}: {str(e)}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def prepare_plot_data(df, file_name, show_markers, sites_to_plot):
    """
    準備所有繪圖所需的資料
    """
    # 過濾站點
    if sites_to_plot:
        df = df[df['site'].isin(sites_to_plot)]

    # 確保 datetime 格式正確（一次性處理）
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime')

    # 取得所有站點和測項
    all_sites = df['site'].unique()
    all_items = df['item'].unique()

    # 建立檔案專屬資料夾
    file_output_dir = os.path.join(output_dir, file_name)
    os.makedirs(file_output_dir, exist_ok=True)

    plot_tasks = []

    # 為每個站點+測項組合準備資料
    for site in all_sites:
        for item in all_items:
            # 過濾資料
            item_df = df[(df['item'] == item) & (df['site'] == site)].copy()

            if not item_df.empty:
                output_path = os.path.join(file_output_dir, f"{site}_{item}.png")

                # 建立 tuple
                plot_task = (
                    item_df[['datetime', 'value']],  # 只保留需要的欄位
                    item,
                    site,
                    show_markers,
                    (16, 6),
                    0.7,
                    output_path
                )
                plot_tasks.append(plot_task)

    return plot_tasks, len(all_sites), len(all_items)


def plot_all_items_from_file(file_path, show_markers=False, sites_to_plot=None,
                             save_plots=True, n_workers=None):
    """
    從單一 CSV 檔案繪製所有測項（已改進進度條為 as_completed 模式）
    """
    print(f"\n📊 處理檔案: {os.path.basename(file_path)}")

    # 讀取資料
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ 讀取錯誤 {file_path}: {e}")
        return

    required_cols = ['datetime', 'item', 'value', 'site']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ {file_path} 缺少必要欄位")
        print(f"   目前欄位: {df.columns.tolist()}")
        return

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print("🔧 正在準備資料...")
    plot_tasks, n_sites, n_items = prepare_plot_data(df, file_name, show_markers, sites_to_plot)

    total_plots = len(plot_tasks)
    print(f"📋 發現 {n_sites} 個站點 和 {n_items} 個測項")
    print(f"📈 預計產生 {total_plots} 張圖表")

    if total_plots == 0:
        print("⚠️ 無資料可繪製")
        return

    if n_workers is None:
        n_workers = os.cpu_count() or 4

    print(f"⚡ 使用 {n_workers} 個核心進行平行處理")

    # submit 所有任務，並用 as_completed + tqdm 逐一更新進度
    successful_plots = 0
    futures = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for task in plot_tasks:
            futures.append(executor.submit(plot_item_timeseries_single_station, task))

        # 使用 tqdm 監控 as_completed
        for future in tqdm(as_completed(futures), total=total_plots, desc="繪圖進度", unit="img"):
            try:
                result = future.result()
                if result is not None:
                    successful_plots += 1
            except Exception as e:
                # 個別 task 的錯誤已在 plot_item_timeseries_single_station 印出，這裡為保險再印一次
                print(f"❌ 執行 task 時遇到例外: {e}")

    print(f"\n✅ 成功產生 {successful_plots}/{total_plots} 張圖表")
    if successful_plots < total_plots:
        print(f"⚠️ {total_plots - successful_plots} 張圖表失敗")


def batch_plot_all_hourly_files(show_markers=False, sites_to_plot=None,
                                save_plots=True, n_workers=None):
    """
    批次處理所有 hourly CSV 檔案
    """
    # 找出所有 hourly 檔案
    # 使用 os.path.join 確保跨平台相容性
    pattern = os.path.join(base_dir, "hourly_201*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"❌ 在 {base_dir} 找不到符合 'hourly_201*.csv' 的檔案")
        print("   請確認您的 csv 檔案已放入正確的 data 資料夾中。")
        return

    print(f"🗂️ 發現 {len(files)} 個檔案")

    # 批次處理
    for file_path in files:
        plot_all_items_from_file(
            file_path,
            show_markers=show_markers,
            sites_to_plot=sites_to_plot,
            save_plots=save_plots,
            n_workers=n_workers
        )

    print(f"\n🎉 所有任務完成！圖片已儲存至: {output_dir}")


# ========================================
# === 主程式進入點 ===
# ========================================
if __name__ == "__main__":
    
    # === 使用範例設定 ===
    
    # 1. 是否顯示數據點 (True/False)
    SHOW_MARKERS = False 
    
    # 2. 指定要畫的站點 (None 表示全部畫，如果要指定則用列表)
    # 範例: SITES_TO_PLOT = ["臺南", "高雄", "臺北"]
    SITES_TO_PLOT = None 
    
    # 3. 指定使用的 CPU 核心數 (None 表示自動偵測)
    N_WORKERS = None 

    print("=== AQI 批次繪圖工具 (本地版) ===")
    
    # 執行批次處理
    batch_plot_all_hourly_files(
        show_markers=SHOW_MARKERS,
        sites_to_plot=SITES_TO_PLOT,
        save_plots=True,
        n_workers=N_WORKERS
    )
    
    # 讓視窗在執行完後暫停 (方便 Windows 使用者看結果)
    if platform.system() == "Windows":
        os.system("pause")