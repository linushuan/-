#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改寫版 v5：AQI 批次繪圖 + 區域平均功能 (優化版)
優化重點：
 1. 避免重複讀取 CSV 檔案
 2. 一次處理所有區域，而非逐個處理
 3. 使用 vectorized 操作取代迴圈
 4. 改善記憶體使用效率
"""

import os
import glob
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['axes.unicode_minus'] = False

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
output_dir = os.path.join(current_dir, "output_pictures_v5")
os.makedirs(output_dir, exist_ok=True)

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

# ---------- 字體設定 ----------
def set_chinese_font():
    # 嘗試檔案路徑 addfont
    p = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
    found_name = None
    fontManager.addfont(p)   # 將字檔加入 matplotlib font manager
    prop = FontProperties(fname=p)
    found_name = prop.get_name()
    if found_name:
        import matplotlib as mpl
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = [found_name]
        print(f"✅ 以字檔載入並設定: {p} -> {found_name}")
    return found_name

set_chinese_font()

# ---------- 繪圖 Task ----------
def plot_overlay_task(task):
    try:
        item = task['item']
        site_data = task['site_data']
        info = task['info']
        note = task.get('note', '')
        fig, ax = plt.subplots(figsize=task['figsize'])

        region_styles = {
            "AVG_北": {"color": "blue", "label": "北部平均"},
            "AVG_中": {"color": "green", "label": "中部平均"},
            "AVG_南": {"color": "red", "label": "南部平均"},
            "AVG_東": {"color": "orange", "label": "東部平均"}
        }

        for site_name, data in site_data.items():
            dt_arr = pd.to_datetime(data['times']).to_pydatetime()
            val_arr = data['values']
            if site_name in region_styles:
                style = region_styles[site_name]
                ax.plot(dt_arr, val_arr, label=style['label'], 
                       color=style['color'], lw=3, ls='-', zorder=10)
            else:
                ax.plot(dt_arr, val_arr, lw=1.5, alpha=0.5, 
                       color='gray', zorder=5)

        ax.set_title(f"區域分析 - {info.get('name', item)} {note}", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(f"{info.get('name', item)} ({info.get('unit','')})")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(task['output_path'], dpi=150, bbox_inches='tight')
        plt.close(fig)
        return task['output_path']
    except Exception as e:
        print(f"繪圖錯誤: {e}")
        return None

# ---------- 資料處理 (優化版) ----------
def process_dataframe(df, sites_to_plot, resample_hours, add_regional_means):
    """優化：使用 vectorized 操作處理資料"""
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    
    dfs_to_concat = [df]

    if add_regional_means:
        # 優化：一次計算所有區域平均
        region_dfs = []
        for region_name, region_sites in areas.items():
            avg_site_name = f"AVG_{region_name}"
            
            # 若有指定站點且此區域不在列表中，跳過
            if sites_to_plot and avg_site_name not in sites_to_plot:
                continue
                
            mask = df['site'].isin(region_sites)
            if not mask.any():
                continue
            
            mean_df = (df[mask]
                      .groupby(['datetime', 'item'], as_index=False)
                      ['value'].mean(numeric_only=True))
            mean_df['site'] = avg_site_name
            region_dfs.append(mean_df)
        
        if region_dfs:
            dfs_to_concat.extend(region_dfs)

    df = pd.concat(dfs_to_concat, ignore_index=True)

    if sites_to_plot:
        df = df[df['site'].isin(set(sites_to_plot))]

    note_str = ''
    if resample_hours and resample_hours > 0:
        note_str = f'({resample_hours}hr Avg)'
        df = (df.set_index('datetime')
              .groupby(['site', 'item'])['value']
              .resample(f'{resample_hours}h')
              .mean()
              .reset_index())
    
    df = df.sort_values('datetime')
    return df, note_str

# ---------- 任務準備 ----------
def prepare_tasks(df, file_name, note_str, output_suffix):
    tasks = []
    file_output_dir = os.path.join(output_dir, file_name)
    os.makedirs(file_output_dir, exist_ok=True)

    for item, group in df.groupby('item'):
        site_data = {
            site: {
                'times': sub['datetime'].values, 
                'values': sub['value'].values
            }
            for site, sub in group.groupby('site')
        }
        tasks.append({
            'item': item,
            'site_data': site_data,
            'output_path': os.path.join(file_output_dir, 
                                       f"REGION_{item}_{output_suffix}.png"),
            'figsize': (16, 8),
            'info': items_info.get(item, {'name': item, 'unit': ''}),
            'note': note_str
        })
    return tasks

# ---------- 處理單檔 (優化版) ----------
def plot_file_for_region(file_path, region_name, resample_h, 
                        add_regional, n_workers, df_raw=None):
    """
    優化版：處理單一檔案的單一區域
    可選擇傳入已讀取的 df_raw 以避免重複讀檔
    """
    if df_raw is None:
        df_raw = pd.read_csv(file_path)
    
    sites_to_plot = [region_name]
    df_proc, note_str = process_dataframe(df_raw, sites_to_plot, 
                                         resample_h, add_regional)
    
    if df_proc.empty:
        return 0, 0
    
    # 取得時間範圍資訊
    t_start = pd.to_datetime(df_proc['datetime']).min().strftime('%m%d_%H')
    t_end = pd.to_datetime(df_proc['datetime']).max().strftime('%m%d_%H')
    year_part = os.path.basename(file_path).split('_')[1].split('.')[0]
    output_suffix = f"{region_name}_{year_part}_{t_start}-{t_end}"
    
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    tasks = prepare_tasks(df_proc, file_name, note_str, output_suffix)
    
    total = len(tasks)
    success = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as exc:
        futures = [exc.submit(plot_overlay_task, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=total, 
                     desc=f'    繪製 {region_name}', leave=False, unit='圖'):
            if f.result():
                success += 1
    
    return success, total

# ---------- 主程式 (大幅優化版) ----------
if __name__ == '__main__':
    print("=== AQI 批次繪圖工具 v5 (優化版) ===")
    
    REGIONS = ['AVG_南', 'AVG_北', 'AVG_中', 'AVG_東']
    RESAMPLE_HOURS = 1
    ADD_REGIONAL_MEANS = True
    N_WORKERS = 12
    
    pattern = os.path.join(base_dir, 'hourly_2019*.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("⚠️ 找不到符合條件的檔案")
        exit(1)
    
    print(f"📁 找到 {len(files)} 個檔案")
    print(f"🌏 將處理 {len(REGIONS)} 個區域: {', '.join(REGIONS)}")
    
    total_success = 0
    total_tasks = 0
    
    # 優化重點：外層迴圈改為檔案，內層處理所有區域
    # 這樣每個檔案只讀取一次
    print(f"\n{'='*60}")
    for fp in tqdm(files, desc='📁 處理檔案', unit='file'):
        file_name = os.path.basename(fp)
        
        # 只讀取一次 CSV
        try:
            df_raw = pd.read_csv(fp)
        except Exception as e:
            tqdm.write(f"❌ 讀取檔案失敗 [{file_name}]: {e}")
            continue
        
        # 對所有區域使用同一份資料
        for region in tqdm(REGIONS, desc=f'🗺️  {file_name}', 
                          leave=False, unit='region'):
            success, total = plot_file_for_region(
                fp, region, RESAMPLE_HOURS, 
                ADD_REGIONAL_MEANS, N_WORKERS, 
                df_raw=df_raw  # 傳入已讀取的資料
            )
            total_success += success
            total_tasks += total
            if total > 0:
                tqdm.write(f"  ✅ {region}: {success}/{total} 張")
    
    print(f"\n{'='*60}")
    print(f"🎉 總計完成 {total_success}/{total_tasks} 張圖")
    print(f"📂 輸出目錄: {output_dir}")
    print(f"{'='*60}")
