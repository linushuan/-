#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改寫版：AQI 批次繪圖工具
主要改進：
 - 在程式最前強制使用 Agg backend（必須在 import pyplot 前設定）
 - 更強健的中文字體載入（嘗試字名、再以檔案路徑 addfont）
 - 使用 ProcessPoolExecutor 實作多核心（真正使用多核心）
 - prepare_plot_data 使用 groupby 減少複製並以 numpy arrays 傳遞給子行程
 - 繪圖進度條改為 as_completed + tqdm（即時且正確）
"""

import os
import sys
import glob
import platform
from datetime import datetime
import multiprocessing

# ---------- matplotlib backend 必須在 import pyplot 前設定 ----------
import matplotlib as mpl
mpl.use('Agg')       # 非互動模式，適合批次產生圖片
mpl.rcParams['axes.unicode_minus'] = False

# 現在再 import pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import fontManager, FontProperties
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---------- Config ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, "data")
output_dir = os.path.join(current_dir, "output_pictures3")
os.makedirs(output_dir, exist_ok=True)

# 測項資訊
items_info = {
    "AMB_TEMP": {"name": "環境溫度", "unit": "°C", "color": "#FF6B6B"},
    "CO": {"name": "一氧化碳", "unit": "ppm", "color": "#4ECDC4"},
    "NO": {"name": "一氧化氮", "unit": "ppb", "color": "#45B7D1"},
    "NO2": {"name": "二氧化氮", "unit": "ppb", "color": "#96CEB4"},
    "NOx": {"name": "氮氧化物", "unit": "ppb", "color": "#FFEAA7"},
    "O3": {"name": "臭氧", "unit": "ppb", "color": "#DBA3EA"},
    "PM10": {"name": "懸浮微粒", "unit": "μg/m³", "color": "#A08DFF"},
    "PM2.5": {"name": "細懸浮微粒", "unit": "μg/m³", "color": "#FD79A8"},
    "RAINFALL": {"name": "降雨量", "unit": "mm", "color": "#74B9FF"},
    "RH": {"name": "相對濕度", "unit": "%", "color": "#81ECEC"},
    "SO2": {"name": "二氧化硫", "unit": "ppb", "color": "#FAB1A0"},
    "WD_HR": {"name": "風向", "unit": "degrees", "color": "#00B894"},
    "WIND_DIREC": {"name": "風向", "unit": "degrees", "color": "#00CEC9"},
    "WIND_SPEED": {"name": "風速", "unit": "m/s", "color": "#0984E3"},
    "WS_HR": {"name": "平均風速", "unit": "m/s", "color": "#6C5CE7"}
}

# ---------- 字體設定（強化版） ----------
def set_chinese_font(preferred=None):
    """
    強健的中文字體設定：
      - 優先用 preferred（若在系統字表）
      - 否則以候選名稱尋找
      - 若找不到名稱，嘗試以常見路徑 addfont()
      - 最後設定 rcParams['font.family']='sans-serif' 並提供候選清單
    回傳實際使用的字型 name (若有)，否則回 None
    """
    import matplotlib as mpl
    available = {f.name for f in fontManager.ttflist}

    # 若使用者指定且存在
    if preferred and preferred in available:
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ["Noto Sans CJK TC"]
        print("✅ 使用指定字體:", preferred)
        return preferred

    # 候選名稱（以支援繁體中文的字型為優先）
    candidates = [
        "Noto Sans CJK TC", "Noto Sans CJK", "Noto Sans CJK TC Regular"
    ]

    # 1) 以字名找
    for name in candidates:
        if name in available:
            mpl.rcParams['font.family'] = 'sans-serif'
            # 把找到的字型放到第一，然後把常見備援放後面
            mpl.rcParams['font.sans-serif'] = [name] + [c for c in candidates if c != name]
            print(f"✅ 已設定中文字體 (name): {name}")
            return name

    # 2) 嘗試常見檔案路徑 addfont
    common_paths = [
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
    ]
    found_name = None
    for p in common_paths:
        if os.path.exists(p):
            try:
                fontManager.addfont(p)   # 將字檔加入 matplotlib font manager
                # 重新取得名稱（可能新增了新字型）
                new_set = {f.name for f in fontManager.ttflist}
                added = list(new_set - available)
                if added:
                    found_name = added[0]
                else:
                    # fallback: 用 FontProperties 嘗試取得名稱
                    prop = FontProperties(fname=p)
                    found_name = prop.get_name()
                if found_name:
                    import matplotlib as mpl
                    mpl.rcParams['font.family'] = 'sans-serif'
                    mpl.rcParams['font.sans-serif'] = [found_name] + candidates
                    print(f"✅ 以字檔載入並設定: {p} -> {found_name}")
                    return found_name
            except Exception as e:
                print(f"⚠ 無法載入字檔 {p}: {e}")

    # 3) 都沒找到：設定一個含候選的清單（DejaVu 放後備）
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = candidates + ["DejaVu Sans"]
    print("⚠ 未找到系統內合適 CJK 字型，已設定候選清單（但可能仍缺字）。")
    print("  建議安裝 Noto CJK 或 wqy-microhei。例如（EndeavourOS/Arch）:")
    print("    sudo pacman -S noto-fonts-cjk wqy-microhei")
    print("  或 Debian/Ubuntu:")
    print("    sudo apt update && sudo apt install fonts-noto-cjk fonts-wqy-microhei")
    return None

# 呼叫字體設定
set_chinese_font()

# ---------- 繪圖函式（子行程會執行） ----------
def plot_item_timeseries_task(task):
    """
    task: dict 包含序列化可以跨 process 傳送的項目：
      {
        'datetimes': ndarray of datetime64[ns] or ISO strings,
        'values': ndarray of float,
        'item': str,
        'site': str,
        'output_path': str,
        'show_markers': bool,
        'figsize': (w,h),
        'alpha': float,
        'info': dict (name, unit, color)
      }
    回傳 output_path 或 None
    """
    try:
        # 因為是在子行程，需再次設定 backend 參數（但我們已在主程式 global 設定 Agg）
        # 將 datetimes 轉回 python datetime （如果是 numpy datetime64）
        datetimes = task['datetimes']
        if datetimes.dtype.type is np.datetime64:
            dt_list = pd.to_datetime(datetimes).to_pydatetime()
        else:
            # 可能是字串
            dt_list = pd.to_datetime(datetimes).to_pydatetime()

        values = task['values']
        item = task['item']
        site = task['site']
        output_path = task['output_path']
        show_markers = task['show_markers']
        figsize = task['figsize']
        alpha = task['alpha']
        info = task['info']

        fig, ax = plt.subplots(figsize=figsize)
        if show_markers:
            ax.plot(dt_list, values, marker='o', markersize=3, linestyle='-', linewidth=1.5, alpha=alpha, color=info.get('color', '#333333'))
        else:
            ax.plot(dt_list, values, linestyle='-', linewidth=1.5, alpha=alpha, color=info.get('color', '#333333'))

        start_time = dt_list[0].strftime('%Y-%m-%d') if len(dt_list) else ''
        end_time = dt_list[-1].strftime('%Y-%m-%d') if len(dt_list) else ''
        time_range = f"{start_time} ~ {end_time}"
        ax.set_title(f"{site} - {info.get('name', item)} ({item})\n{time_range}", fontsize=12, fontweight='bold')
        ax.set_xlabel('時間', fontsize=10)
        ax.set_ylabel(f"{info.get('name', item)} ({info.get('unit','')})", fontsize=10)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.grid(True, alpha=0.25, linestyle='--')

        plt.tight_layout()
        # 儲存圖片
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    except Exception as e:
        # 在子行程中少印訊息，回傳 None
        # 回傳錯誤訊息可以有助於主程式 log（這裡只回 None）
        return None

# ---------- 資料準備（更快） ----------
def prepare_plot_tasks_from_df(df, file_name, show_markers, sites_to_plot):
    """
    以 groupby('site','item') 產生每個任務，
    並用 numpy arrays (datetime64, float) 傳遞以減少序列化負擔。
    回傳 tasks_list, n_sites, n_items
    """
    if sites_to_plot:
        df = df[df['site'].isin(sites_to_plot)]

    # 一次性轉換 datetime（向量化）
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    # 依 datetime 排序（整張表）
    df = df.sort_values('datetime')

    all_sites = df['site'].unique()
    all_items = df['item'].unique()

    file_output_dir = os.path.join(output_dir, file_name)
    os.makedirs(file_output_dir, exist_ok=True)

    tasks = []
    # groupby 避免逐筆過濾 DataFrame，速度大幅提升
    grouped = df.groupby(['site', 'item'])
    for (site, item), group in grouped:
        # 將 group 轉成 numpy arrays
        # 注意：保留原始順序（已全表排序）
        dt_arr = group['datetime'].to_numpy(dtype='datetime64[ns]')
        val_arr = group['value'].to_numpy(dtype=float)
        if val_arr.size == 0:
            continue
        out_path_safe_site = "".join(c if (c.isalnum() or c in (' ', '-', '_')) else '_' for c in str(site))
        out_path = os.path.join(file_output_dir, f"{out_path_safe_site}_{item}.png")
        info = items_info.get(item, {"name": item, "unit": "", "color": "#95A5A6"})
        task = {
            'datetimes': dt_arr,
            'values': val_arr,
            'item': item,
            'site': str(site),
            'output_path': out_path,
            'show_markers': show_markers,
            'figsize': (16, 6),
            'alpha': 0.7,
            'info': info
        }
        tasks.append(task)

    return tasks, len(all_sites), len(all_items)

# ---------- 主要處理單一檔案（使用 ProcessPoolExecutor） ----------
def plot_all_items_from_file(file_path, show_markers=False, sites_to_plot=None, n_workers=None):
    print(f"\n📊 處理檔案: {os.path.basename(file_path)}")
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
    print("🔧 正在準備資料（向量化轉換）...")
    tasks, n_sites, n_items = prepare_plot_tasks_from_df(df, file_name, show_markers, sites_to_plot)
    total_plots = len(tasks)
    print(f"📋 發現 {n_sites} 個站點 和 {n_items} 個測項")
    print(f"📈 預計產生 {total_plots} 張圖表")

    if total_plots == 0:
        print("⚠️ 無資料可繪製")
        return

    if n_workers is None:
        # 限制最多不超過 CPU 數
        n_workers = min((os.cpu_count() or 4), total_plots)

    print(f"⚡ 使用 {n_workers} 個 worker（Process）進行平行處理")

    successful_plots = 0
    futures = []
    # 使用 ProcessPoolExecutor 以真正使用多核心（適合 CPU-bound）
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for task in tasks:
            futures.append(executor.submit(plot_item_timeseries_task, task))

        # 使用 as_completed + tqdm 取得即時進度
        for fut in tqdm(as_completed(futures), total=total_plots, desc="繪圖進度", unit="img"):
            try:
                res = fut.result()
                if res is not None:
                    successful_plots += 1
            except Exception as e:
                # 若子行程拋例外，可在這裡印出
                print(f"❌ 子行程例外: {e}")

    print(f"\n✅ 成功產生 {successful_plots}/{total_plots} 張圖表")
    if successful_plots < total_plots:
        print(f"⚠️ {total_plots - successful_plots} 張圖表失敗")

# ---------- 批次處理所有檔案 ----------
def batch_plot_all_hourly_files(show_markers=False, sites_to_plot=None, n_workers=None):
    pattern = os.path.join(base_dir, "hourly_2019*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"❌ 在 {base_dir} 找不到符合 'hourly_2019*.csv' 的檔案")
        print("   請確認您的 csv 檔案已放入正確的 data 資料夾中。")
        return
    print(f"🗂️ 發現 {len(files)} 個檔案")
    # 依序處理每個檔案（你也可改為多檔並行，但請注意 I/O）
    for file_path in files:
        plot_all_items_from_file(file_path, show_markers=show_markers, sites_to_plot=sites_to_plot, n_workers=n_workers)
    print(f"\n🎉 所有任務完成！圖片已儲存至: {output_dir}")

# ---------- 主程式 ----------
if __name__ == "__main__":
    print("=== AQI 批次繪圖工具（改良版） ===")
    SHOW_MARKERS = False
    SITES_TO_PLOT = None
    # 你可以指定數字，例如 4；若為 None，會自動選擇 min(CPU, total_tasks)
    N_WORKERS = 10 #None

    batch_plot_all_hourly_files(show_markers=SHOW_MARKERS, sites_to_plot=SITES_TO_PLOT, n_workers=N_WORKERS)
