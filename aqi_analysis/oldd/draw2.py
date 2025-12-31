#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import platform
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import pandas as pd
import numpy as np

# matplotlib 相關（在主程式與子程序都會明確設定字體）
import matplotlib as mpl
mpl.use("Agg")  # 確保非互動環境
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

# 進度條
from tqdm import tqdm

# ======================
# 全域設定
# ======================
mpl.rcParams['axes.unicode_minus'] = False  # 負號顯示問題

# 請視情況調整（通常不用改）
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_pictures2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 測項資訊
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

# =============
# 字體處理工具（在主程序與 worker 都會呼叫）
# =============
def setup_chinese_font(preferred: list = None):
    """
    嘗試設定中文字體。返回設定成功的字體名稱或 None。
    preferred: 可傳入偏好的字體名稱列表（系統內的 family 名稱），若為 None 會使用預設清單。
    """
    # 預設候選（以 Linux 常見字體為主）
    if preferred is None:
        if platform.system() == "Windows":
            candidates = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
        elif platform.system() == "Darwin":
            candidates = ["PingFang TC", "Heiti TC", "Arial Unicode MS"]
        else:
            candidates = [
                "Noto Sans CJK TC", "NotoSansCJKtc", "Noto Sans CJK",
                "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                "AR PL UKai CN", "Droid Sans Fallback", "DejaVu Sans"
            ]
    else:
        candidates = preferred

    available = {f.name for f in font_manager.fontManager.ttflist}

    for name in candidates:
        if name in available:
            mpl.rcParams['font.family'] = name
            mpl.rcParams['font.sans-serif'] = [name]
            print(f"✅ 已設定中文字體 family: {name}")
            return name

    # 若 family 名稱沒有在 fontManager 中，嘗試用常見路徑直接加入字體檔（較具容錯）
    common_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKtc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        os.path.expanduser("~/.local/share/fonts/NotoSansCJKtc-Regular.otf")
    ]
    for p in common_paths:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)  # 新版 matplotlib 支援
                # 重新取得可用 font 名稱
                new_avail = {f.name for f in font_manager.fontManager.ttflist}
                # 取 newly added 的 name（用最後加入的檔案去尋找）
                for f in font_manager.fontManager.ttflist[::-1]:
                    if os.path.exists(getattr(f, "fname", "") or "") and getattr(f, "fname", "").startswith(p[:20]):
                        fam = f.name
                        mpl.rcParams['font.family'] = fam
                        mpl.rcParams['font.sans-serif'] = [fam]
                        print(f"✅ 以檔案加入並設定字體: {p} -> {fam}")
                        return fam
                # 若沒找到 family，仍嘗試以路徑加入並回傳成功
                print(f"✅ 已將字體檔加入 FontManager: {p}（若無顯示請清除 matplotlib 快取後重啟）")
                return p
            except Exception as e:
                print(f"⚠️ 嘗試加入字體檔失敗: {p} -> {e}")
                continue

    # 無法自動設定
    print("⚠️ 未偵測到合適中文字體。建議安裝：fonts-noto-cjk 或 fonts-wqy-microhei。")
    if platform.system() != "Windows":
        print("  Debian/Ubuntu 範例安裝指令：")
        print("    sudo apt update && sudo apt install fonts-noto-cjk fonts-wqy-microhei")
        print("  若安裝後仍無效，請清除 matplotlib 快取後重新執行：")
        print("    rm -rf ~/.cache/matplotlib")
    return None

# 立即在主進程嘗試設定字體（子程序也會再執行一次）
setup_chinese_font()

# ============================
# 用於建立單張圖的 worker（會在子程序中執行）
# ============================
def worker_plot_single(args):
    """
    在子程序中執行：建立圖表並儲存。
    args 為 tuple: (datetimes_list, values_list, item, site, show_markers, figsize, alpha, output_path)
    注意：只傳遞最小必要資料以便 process 傳輸 (避免 DataFrame 序列化問題)。
    """
    try:
        # 每個 worker 都要重新設定 matplotlib 與中文字體（確保子程序也能正確顯示中文）
        mpl.use("Agg")
        mpl.rcParams['axes.unicode_minus'] = False
        setup_chinese_font()

        datetimes, values, item, site, show_markers, figsize, alpha, output_path = args

        if len(datetimes) == 0:
            return None

        # 建立 figure
        fig, ax = plt.subplots(figsize=figsize)

        info = items_info.get(item, {"name": item, "unit": "", "color": "#95A5A6"})

        # 繪圖（datetimes 已是 datetime 物件的 list）
        if show_markers:
            ax.plot(datetimes, values, marker='o', markersize=4, linestyle='-',
                    linewidth=1.5, alpha=alpha, color=info['color'])
        else:
            ax.plot(datetimes, values, linestyle='-', linewidth=1.5, alpha=alpha, color=info['color'])

        # 標題與標籤
        start_time = min(datetimes).strftime('%Y-%m-%d')
        end_time = max(datetimes).strftime('%Y-%m-%d')
        time_range = f"{start_time} ~ {end_time}"
        ax.set_title(f"{site} - {info['name']} ({item})\n{time_range}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('時間', fontsize=10)
        ax.set_ylabel(f"{info['name']} ({info['unit']})", fontsize=10)

        # 日期格式化
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        ax.grid(True, alpha=0.25, linestyle='--')
        plt.tight_layout()

        # 儲存檔案
        plt.savefig(output_path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        return output_path

    except Exception as e:
        # 在子程序裡印出完整例外資訊
        import traceback
        traceback.print_exc()
        return None

# ============================
# 資料準備（更快的實作）
# ============================
def prepare_plot_tasks_fast(df: pd.DataFrame, file_name: str, show_markers: bool, sites_to_plot):
    """
    使用 groupby 來快速準備 (site, item) 的任務，避免大量 DataFrame 複製。
    回傳 plot_tasks (list of args) 以及 n_sites, n_items。
    """
    # 過濾站點（若有指定）
    if sites_to_plot:
        df = df[df['site'].isin(sites_to_plot)]

    # 確保 datetime 欄位是 datetime（外部讀檔時用 parse_dates 較快）
    if df['datetime'].dtype == object or not np.issubdtype(df['datetime'].dtype, np.datetime64):
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # dropna, sort（一次性）
    df = df.dropna(subset=['datetime', 'value'])
    df = df.sort_values('datetime')

    # 取得 unique counts
    all_sites = df['site'].unique()
    all_items = df['item'].unique()

    # 建立檔案專屬資料夾
    file_output_dir = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_dir, exist_ok=True)

    plot_tasks = []
    # 利用 groupby 一次iter
    grouped = df.groupby(['site', 'item'])
    for (site, item), group in grouped:
        # 避免複製整個 DataFrame，取需要的欄位並轉成 list（可序列化）
        datetimes = group['datetime'].tolist()
        values = group['value'].tolist()

        if len(datetimes) == 0:
            continue

        # 儲存路徑 (用 safe 檔名)
        safe_site = "".join(c if c.isalnum() or c in (' ', '-', '_') else "_" for c in str(site))
        safe_item = "".join(c if c.isalnum() or c in (' ', '-', '_') else "_" for c in str(item))
        output_path = os.path.join(file_output_dir, f"{safe_site}_{safe_item}.png")

        args = (datetimes, values, item, site, show_markers, (16, 6), 0.7, output_path)
        plot_tasks.append(args)

    return plot_tasks, len(all_sites), len(all_items)

# ============================
# 針對單個檔案的主流程（使用 ProcessPoolExecutor）
# ============================
def plot_all_items_from_file(file_path, show_markers=False, sites_to_plot=None, n_workers=None):
    print(f"\n📊 處理檔案: {os.path.basename(file_path)}")

    # 讀 CSV：使用 parse_dates 可加速 datetime 解析
    try:
        df = pd.read_csv(file_path, parse_dates=['datetime'], infer_datetime_format=True)
    except Exception as e:
        print(f"❌ 讀取錯誤 {file_path}: {e}")
        return

    # 檢查必要欄位
    required_cols = ['datetime', 'item', 'value', 'site']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ {file_path} 缺少必要欄位")
        print(f"   目前欄位: {df.columns.tolist()}")
        return

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print("🔧 正在準備資料...")
    plot_tasks, n_sites, n_items = prepare_plot_tasks_fast(df, file_name, show_markers, sites_to_plot)
    total_plots = len(plot_tasks)
    print(f"📋 發現 {n_sites} 個站點 和 {n_items} 個測項")
    print(f"📈 預計產生 {total_plots} 張圖表")

    if total_plots == 0:
        print("⚠️ 無資料可繪製")
        return

    if n_workers is None:
        n_workers = os.cpu_count() or 4
    else:
        try:
            n_workers = int(n_workers)
            if n_workers <= 0:
                n_workers = os.cpu_count() or 4
        except Exception:
            n_workers = os.cpu_count() or 4

    print(f"⚡ 使用 {n_workers} 個 process 進行平行處理")

    successful = 0
    futures = []
    # 使用 ProcessPoolExecutor 以真正利用多核心（matplotlib 在子 process 建圖）
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for args in plot_tasks:
            futures.append(ex.submit(worker_plot_single, args))

        # 以 as_completed 搭配 tqdm 可靠地更新進度
        for fut in tqdm(as_completed(futures), total=total_plots, desc="繪圖進度", unit="img"):
            try:
                r = fut.result()
                if r is not None:
                    successful += 1
            except Exception as e:
                # 印出錯誤但不中斷其他任務
                print(f"❌ 子程序產生錯誤: {e}")

    print(f"\n✅ 成功產生 {successful}/{total_plots} 張圖表")
    if successful < total_plots:
        print(f"⚠️ {total_plots - successful} 張圖表失敗")

# ============================
# 批次處理所有檔案
# ============================
def batch_plot_all_hourly_files(show_markers=False, sites_to_plot=None, n_workers=None):
    pattern = os.path.join(BASE_DIR, "hourly_201*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"❌ 在 {BASE_DIR} 找不到符合 'hourly_201*.csv' 的檔案")
        return
    print(f"🗂️ 發現 {len(files)} 個檔案")

    for fp in files:
        plot_all_items_from_file(fp, show_markers=show_markers, sites_to_plot=sites_to_plot, n_workers=n_workers)

    print(f"\n🎉 所有任務完成！圖片已儲存至: {OUTPUT_DIR}")

# ============================
# Main
# ============================
if __name__ == "__main__":
    # 範例設定
    SHOW_MARKERS = False
    SITES_TO_PLOT = None  # 或 ["臺南", "高雄"]
    N_WORKERS = None      # 指定數字或 None

    print("=== AQI 批次繪圖工具（改良版） ===")
    batch_plot_all_hourly_files(show_markers=SHOW_MARKERS, sites_to_plot=SITES_TO_PLOT, n_workers=N_WORKERS)
