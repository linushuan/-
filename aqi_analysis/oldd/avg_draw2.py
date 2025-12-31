#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
距平值計算與繪圖工具 (PC 版本) - 按日期分檔
功能：
 1. 計算各測項的距平值 (實際值 - 歷史平均值)
 2. 按原始檔案日期範圍分別儲存 CSV
 3. 繪製距平值時間序列圖（與原始 v5 相同格式）
 4. 支援區域平均距平值計算
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
output_dir = os.path.join(current_dir, "output_anomaly_pictures2")
anomaly_dir = os.path.join(current_dir, "anomaly_csvs2")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(anomaly_dir, exist_ok=True)

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
    "PM10": {"name": "PM10", "unit": "μg/m³", "color": "#A08DFF"},
    "PM2.5": {"name": "PM2.5", "unit": "μg/m³", "color": "#FD79A8"},
    "RAINFALL": {"name": "降雨量", "unit": "mm", "color": "#74B9FF"},
    "RH": {"name": "相對濕度", "unit": "%", "color": "#81ECEC"},
    "SO2": {"name": "二氧化硫", "unit": "ppb", "color": "#FAB1A0"},
    "WD_HR": {"name": "平均風向", "unit": "degrees", "color": "#00B894"},
    "WIND_DIREC": {"name": "風向", "unit": "degrees", "color": "#00CEC9"},
    "WIND_SPEED": {"name": "風速", "unit": "m/s", "color": "#0984E3"},
    "WS_HR": {"name": "平均風速", "unit": "m/s", "color": "#6C5CE7"}
}

# ---------- 字體設定 ----------
def set_chinese_font():
    p = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
    try:
        fontManager.addfont(p)
        prop = FontProperties(fname=p)
        found_name = prop.get_name()
        if found_name:
            mpl.rcParams['font.family'] = 'sans-serif'
            mpl.rcParams['font.sans-serif'] = [found_name]
            print(f"✅ 字體設定完成: {found_name}")
        return found_name
    except Exception as e:
        print(f"⚠️ 字體載入失敗: {e}")
        return None

set_chinese_font()

# ===============================================================
#   STEP 1: 計算距平值（按日期分檔）
# ===============================================================

def load_historical_averages():
    """載入所有測項的歷史平均值"""
    df_avg_dict = {}
    for item in items_info.keys():
        avg_file = os.path.join(base_dir, f"{item.lower()}_hourly_avg_fast.csv")
        
        if not os.path.exists(avg_file):
            continue
        
        try:
            df_avg = pd.read_csv(avg_file, encoding='utf-8-sig')
        except:
            try:
                df_avg = pd.read_csv(avg_file, encoding='utf-8')
            except:
                continue
        
        df_avg_dict[item] = df_avg
    
    return df_avg_dict


def calculate_anomalies_for_file(file_path, df_avg_dict):
    """
    計算單一檔案所有測項的距平值，並按原始檔案日期範圍儲存
    
    Returns:
    --------
    file_name: 原始檔案名稱（用於建立對應的距平檔案）
    """
    try:
        # 讀取原始資料
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        
        if df.empty:
            return None, None
        
        # 取得原始檔案名稱
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 計算 day_of_year 和 hour
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['hour'] = df['datetime'].dt.hour
        
        all_anomalies = []
        
        # 對每個測項計算距平值
        for item in df['item'].unique():
            if item not in df_avg_dict:
                continue
            
            df_avg = df_avg_dict[item]
            df_item = df[df['item'] == item].copy()
            
            # 對每個站點計算距平值
            for site in df_item['site'].unique():
                site_data = df_item[df_item['site'] == site].copy()
                
                # 找到該站點在平均值表中的資料
                if '測站' in df_avg.columns:
                    avg_row = df_avg[df_avg['測站'] == site]
                elif 'site' in df_avg.columns:
                    avg_row = df_avg[df_avg['site'] == site]
                else:
                    continue
                
                if avg_row.empty:
                    continue
                
                # 批次計算距平值
                for idx, row in site_data.iterrows():
                    day = int(row['day_of_year'])
                    hour = int(row['hour'])
                    actual_value = row['value']
                    
                    col_name = f"{day}_{hour}"
                    
                    if col_name in avg_row.columns:
                        avg_value = avg_row[col_name].values[0]
                        
                        if pd.notna(avg_value) and pd.notna(actual_value):
                            anomaly = actual_value - avg_value
                            all_anomalies.append({
                                'datetime': row['datetime'],
                                'site': site,
                                'item': item,
                                'value': anomaly  # 改用 value 儲存距平值，保持格式一致
                            })
        
        if not all_anomalies:
            return file_name, None
        
        df_anomalies = pd.DataFrame(all_anomalies)
        df_anomalies = df_anomalies.sort_values('datetime')
        
        # 儲存為與原始檔案對應的距平檔案
        output_file = os.path.join(anomaly_dir, f"anomaly_{file_name}.csv")
        df_anomalies.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        return file_name, len(all_anomalies)
        
    except Exception as e:
        print(f"❌ 處理檔案失敗 {file_path}: {e}")
        return None, None


def calculate_all_anomalies(file_pattern='hourly_*.csv'):
    """
    計算所有檔案的距平值並按日期分別儲存
    """
    print("\n" + "="*60)
    print("  📊 開始計算距平值（按日期分檔）")
    print("="*60)
    
    # 載入所有測項的歷史平均值
    print("\n🔄 載入歷史平均值...")
    df_avg_dict = load_historical_averages()
    
    if not df_avg_dict:
        print("❌ 沒有找到任何歷史平均值檔案！")
        return
    
    print(f"  ✅ 載入了 {len(df_avg_dict)} 個測項的歷史平均值")
    
    # 找出所有 hourly 檔案
    pattern = os.path.join(base_dir, file_pattern)
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"❌ 找不到符合條件的檔案: {pattern}")
        return
    
    print(f"\n📁 找到 {len(files)} 個檔案")
    
    # 處理每個檔案
    total_records = 0
    successful_files = 0
    
    for file_path in tqdm(files, desc="計算距平值", unit="file"):
        file_name, record_count = calculate_anomalies_for_file(file_path, df_avg_dict)
        
        if record_count is not None and record_count > 0:
            total_records += record_count
            successful_files += 1
            tqdm.write(f"  ✅ {file_name}: {record_count} 筆")
    
    print(f"\n🎉 成功處理 {successful_files}/{len(files)} 個檔案")
    print(f"📊 總計 {total_records} 筆距平值記錄")


# ===============================================================
#   STEP 2: 繪製距平值圖表（與 v5 相同格式）
# ===============================================================

def plot_overlay_task(task):
    """
    繪製疊加圖（與 v5 格式相同）
    """
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

        # 繪製個別站點（灰色細線）
        for site_name, data in site_data.items():
            if site_name not in region_styles:
                dt_arr = pd.to_datetime(data['times']).to_pydatetime()
                val_arr = data['values']
                ax.plot(dt_arr, val_arr, lw=1.5, alpha=0.5, 
                       color='gray', zorder=5)
        
        # 繪製區域平均（彩色粗線）
        for site_name, data in site_data.items():
            if site_name in region_styles:
                style = region_styles[site_name]
                dt_arr = pd.to_datetime(data['times']).to_pydatetime()
                val_arr = data['values']
                ax.plot(dt_arr, val_arr, label=style['label'], 
                       color=style['color'], lw=3, ls='-', zorder=10)

        # 添加零線
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

        ax.set_title(f"區域分析 - {info.get('name', item)} 距平值 {note}", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(f"{info.get('name', item)} 距平值 ({info.get('unit','')})")
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
        if 'fig' in locals():
            plt.close(fig)
        return None


def process_anomaly_dataframe(df, regions_to_plot):
    """
    處理距平值資料，計算區域平均
    """
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    
    dfs_to_concat = [df]

    # 計算區域平均
    region_dfs = []
    for region_name, region_sites in areas.items():
        avg_site_name = f"AVG_{region_name}"
        
        if regions_to_plot and avg_site_name not in regions_to_plot:
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

    df_result = pd.concat(dfs_to_concat, ignore_index=True)
    df_result = df_result.sort_values('datetime')
    
    return df_result


def prepare_plot_tasks(df, file_name, regions_to_plot):
    """
    準備繪圖任務
    """
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
        
        # 建立輸出檔名
        region_str = '_'.join([r.replace('AVG_', '') for r in regions_to_plot])
        output_filename = f"ANOMALY_{item}_{region_str}.png"
        
        tasks.append({
            'item': item,
            'site_data': site_data,
            'output_path': os.path.join(file_output_dir, output_filename),
            'figsize': (16, 8),
            'info': items_info.get(item, {'name': item, 'unit': ''}),
            'note': ''
        })
    return tasks


def plot_anomaly_file(file_path, regions_to_plot, n_workers):
    """
    繪製單一距平檔案的圖表
    """
    try:
        df = pd.read_csv(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 處理資料（計算區域平均）
        df_proc = process_anomaly_dataframe(df, regions_to_plot)
        
        if df_proc.empty:
            return 0, 0
        
        # 準備繪圖任務
        tasks = prepare_plot_tasks(df_proc, file_name, regions_to_plot)
        
        total = len(tasks)
        success = 0
        
        with ProcessPoolExecutor(max_workers=n_workers) as exc:
            futures = [exc.submit(plot_overlay_task, t) for t in tasks]
            for f in as_completed(futures):
                if f.result():
                    success += 1
        
        return success, total
        
    except Exception as e:
        print(f"❌ 處理檔案失敗: {e}")
        return 0, 0


def plot_all_anomalies(regions_to_plot, n_workers=4, file_pattern='anomaly_hourly_*.csv'):
    """
    繪製所有距平值圖表
    """
    print("\n" + "="*60)
    print("  🎨 開始繪製距平值圖表")
    print("="*60)
    
    # 找出所有距平檔案
    pattern = os.path.join(anomaly_dir, file_pattern)
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"❌ 找不到距平檔案: {pattern}")
        return
    
    print(f"📁 找到 {len(files)} 個距平檔案")
    print(f"🌏 繪製區域: {', '.join(regions_to_plot)}")
    
    total_success = 0
    total_tasks = 0
    
    for file_path in tqdm(files, desc="繪製圖表", unit="file"):
        file_name = os.path.basename(file_path)
        success, total = plot_anomaly_file(file_path, regions_to_plot, n_workers)
        total_success += success
        total_tasks += total
        if total > 0:
            tqdm.write(f"  ✅ {file_name}: {success}/{total} 張")
    
    print(f"\n🎉 總計完成 {total_success}/{total_tasks} 張圖")
    print(f"📂 輸出目錄: {output_dir}")


# ===============================================================
#   主程式
# ===============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🌡️  距平值計算與繪圖工具 (PC 版本)")
    print("="*60)
    
    # 設定參數
    FILE_PATTERN = 'hourly_2019*.csv'           # 要處理的檔案範圍
    REGIONS = ['AVG_南', 'AVG_北', 'AVG_中', 'AVG_東']  # 要繪製的區域
    N_WORKERS = 10                                # 平行處理數量
    
    # Step 1: 計算距平值（按日期分檔）
    calculate_all_anomalies(file_pattern=FILE_PATTERN)
    
    # Step 2: 繪製距平值圖表
    plot_all_anomalies(regions_to_plot=REGIONS, n_workers=N_WORKERS)
    
    print("\n" + "="*60)
    print("  ✨ 所有任務完成！")
    print(f"  📁 距平值 CSV: {anomaly_dir}")
    print(f"  📁 距平值圖表: {output_dir}")
    print("="*60 + "\n")
