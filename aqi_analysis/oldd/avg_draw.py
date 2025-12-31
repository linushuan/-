#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
距平值計算與繪圖工具 (PC 版本)
功能：
 1. 計算各測項的距平值 (實際值 - 歷史平均值)
 2. 繪製距平值時間序列圖
 3. 支援區域平均距平值計算
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
output_dir = os.path.join(current_dir, "output_anomaly_pictures")
anomaly_dir = os.path.join(current_dir, "anomaly_csvs")
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
#   STEP 1: 計算距平值
# ===============================================================

def load_historical_averages(item):
    """載入歷史平均值"""
    avg_file = os.path.join(base_dir, f"{item.lower()}_hourly_avg_fast.csv")
    
    if not os.path.exists(avg_file):
        print(f"⚠️ 找不到平均值檔案: {avg_file}")
        return None
    
    try:
        df_avg = pd.read_csv(avg_file, encoding='utf-8-sig')
    except:
        try:
            df_avg = pd.read_csv(avg_file, encoding='utf-8')
        except Exception as e:
            print(f"❌ 讀取平均值檔案失敗: {e}")
            return None
    
    return df_avg


def calculate_anomalies_for_file(file_path, df_avg_dict):
    """
    計算單一檔案所有測項的距平值
    
    Parameters:
    -----------
    file_path : str
        hourly CSV 檔案路徑
    df_avg_dict : dict
        {item: df_avg} 的字典，包含所有測項的歷史平均值
    
    Returns:
    --------
    DataFrame with anomalies for all items
    """
    try:
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        
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
                                'actual_value': actual_value,
                                'avg_value': avg_value,
                                'anomaly': anomaly
                            })
        
        if not all_anomalies:
            return None
        
        return pd.DataFrame(all_anomalies)
        
    except Exception as e:
        print(f"❌ 處理檔案失敗 {file_path}: {e}")
        return None


def calculate_all_anomalies(file_pattern='hourly_*.csv'):
    """
    計算所有檔案的距平值並儲存
    """
    print("\n" + "="*60)
    print("  📊 開始計算距平值")
    print("="*60)
    
    # 載入所有測項的歷史平均值
    print("\n🔄 載入歷史平均值...")
    df_avg_dict = {}
    for item in items_info.keys():
        df_avg = load_historical_averages(item)
        if df_avg is not None:
            df_avg_dict[item] = df_avg
            print(f"  ✅ {item}")
    
    if not df_avg_dict:
        print("❌ 沒有找到任何歷史平均值檔案！")
        return
    
    # 找出所有 hourly 檔案
    pattern = os.path.join(base_dir, file_pattern)
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"❌ 找不到符合條件的檔案: {pattern}")
        return
    
    print(f"\n📁 找到 {len(files)} 個檔案")
    
    # 儲存所有測項的距平值
    all_item_anomalies = {item: [] for item in items_info.keys()}
    
    # 處理每個檔案
    for file_path in tqdm(files, desc="計算距平值", unit="file"):
        df_anomalies = calculate_anomalies_for_file(file_path, df_avg_dict)
        
        if df_anomalies is not None and not df_anomalies.empty:
            # 按測項分類
            for item in df_anomalies['item'].unique():
                item_data = df_anomalies[df_anomalies['item'] == item]
                all_item_anomalies[item].append(item_data)
    
    # 合併並儲存每個測項的距平值
    print("\n💾 儲存距平值 CSV...")
    for item in items_info.keys():
        if all_item_anomalies[item]:
            df_combined = pd.concat(all_item_anomalies[item], ignore_index=True)
            df_combined = df_combined.sort_values('datetime')
            
            output_file = os.path.join(anomaly_dir, f"anomaly_{item}.csv")
            df_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"  ✅ {item}: {len(df_combined)} 筆記錄")
    
    print("\n🎉 距平值計算完成！")


# ===============================================================
#   STEP 2: 繪製距平值圖表
# ===============================================================

def plot_anomaly_task(task):
    """
    繪製單一站點或區域的距平值圖
    """
    try:
        anomaly_df = task['data']
        item = task['item']
        site = task['site']
        info = task['info']
        output_path = task['output_path']
        is_region = task.get('is_region', False)
        
        if anomaly_df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # 繪製距平值（使用顏色區分正負值）
        positive = anomaly_df[anomaly_df['anomaly'] >= 0]
        negative = anomaly_df[anomaly_df['anomaly'] < 0]
        
        if not positive.empty:
            ax.scatter(positive['datetime'], positive['anomaly'], 
                      c='red', alpha=0.6, s=30, label='正距平', zorder=5)
        if not negative.empty:
            ax.scatter(negative['datetime'], negative['anomaly'], 
                      c='blue', alpha=0.6, s=30, label='負距平', zorder=5)
        
        # 添加零線
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
        
        # 設定標題
        title_prefix = "區域平均" if is_region else "測站"
        time_range = f"{anomaly_df['datetime'].min().strftime('%Y-%m-%d')} ~ {anomaly_df['datetime'].max().strftime('%Y-%m-%d')}"
        ax.set_title(f"{title_prefix} {site} - {info['name']} 距平值\n{time_range}",
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('時間', fontsize=12)
        ax.set_ylabel(f"距平值 ({info['unit']})", fontsize=12)
        
        # 格式化 x 軸
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        # 網格線和圖例
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
        
    except Exception as e:
        print(f"❌ 繪圖錯誤: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def calculate_regional_anomalies(df_anomaly):
    """
    計算區域平均距平值
    """
    regional_data = []
    
    for region_name, region_sites in areas.items():
        # 過濾該區域的站點
        region_df = df_anomaly[df_anomaly['site'].isin(region_sites)].copy()
        
        if region_df.empty:
            continue
        
        # 計算區域平均距平值
        regional_avg = (region_df.groupby(['datetime', 'item'], as_index=False)
                       .agg({
                           'anomaly': 'mean',
                           'actual_value': 'mean',
                           'avg_value': 'mean'
                       }))
        regional_avg['site'] = f"AVG_{region_name}"
        regional_data.append(regional_avg)
    
    if regional_data:
        return pd.concat(regional_data, ignore_index=True)
    return pd.DataFrame()


def plot_anomalies_for_item(item, include_regions=True, n_workers=4):
    """
    繪製單一測項的所有距平值圖表
    """
    anomaly_file = os.path.join(anomaly_dir, f"anomaly_{item}.csv")
    
    if not os.path.exists(anomaly_file):
        print(f"⚠️ 找不到距平值檔案: {anomaly_file}")
        return 0, 0
    
    try:
        df = pd.read_csv(anomaly_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
    except Exception as e:
        print(f"❌ 讀取檔案失敗: {e}")
        return 0, 0
    
    # 建立輸出資料夾
    item_output_dir = os.path.join(output_dir, f"anomaly_{item}")
    os.makedirs(item_output_dir, exist_ok=True)
    
    tasks = []
    info = items_info.get(item, {"name": item, "unit": "", "color": "#95A5A6"})
    
    # 為每個站點準備繪圖任務
    for site in df['site'].unique():
        site_data = df[df['site'] == site].copy()
        
        if not site_data.empty:
            output_path = os.path.join(item_output_dir, f"{site}_anomaly.png")
            tasks.append({
                'data': site_data,
                'item': item,
                'site': site,
                'info': info,
                'output_path': output_path,
                'is_region': False
            })
    
    # 計算並繪製區域平均
    if include_regions:
        df_regional = calculate_regional_anomalies(df)
        if not df_regional.empty:
            for region_site in df_regional['site'].unique():
                region_data = df_regional[df_regional['site'] == region_site].copy()
                
                if not region_data.empty:
                    output_path = os.path.join(item_output_dir, 
                                              f"{region_site}_anomaly.png")
                    tasks.append({
                        'data': region_data,
                        'item': item,
                        'site': region_site,
                        'info': info,
                        'output_path': output_path,
                        'is_region': True
                    })
    
    # 平行處理繪圖
    total = len(tasks)
    success = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as exc:
        futures = [exc.submit(plot_anomaly_task, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=total, 
                     desc=f'  繪製 {item}', leave=False, unit='圖'):
            if f.result():
                success += 1
    
    return success, total


def plot_all_anomalies(include_regions=True, n_workers=4):
    """
    繪製所有測項的距平值圖表
    """
    print("\n" + "="*60)
    print("  🎨 開始繪製距平值圖表")
    print("="*60)
    
    total_success = 0
    total_tasks = 0
    
    for item in tqdm(items_info.keys(), desc="處理測項", unit="item"):
        success, total = plot_anomalies_for_item(item, include_regions, n_workers)
        total_success += success
        total_tasks += total
        if total > 0:
            tqdm.write(f"  ✅ {item}: {success}/{total} 張")
    
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
    FILE_PATTERN = 'hourly_2019*.csv'  # 可調整要處理的檔案範圍
    INCLUDE_REGIONS = True              # 是否包含區域平均
    N_WORKERS = 10                  # 平行處理數量
    
    # Step 1: 計算距平值
    calculate_all_anomalies(file_pattern=FILE_PATTERN)
    
    # Step 2: 繪製距平值圖表
    plot_all_anomalies(include_regions=INCLUDE_REGIONS, n_workers=N_WORKERS)
    
    print("\n" + "="*60)
    print("  ✨ 所有任務完成！")
    print(f"  📁 距平值 CSV: {anomaly_dir}")
    print(f"  📁 距平值圖表: {output_dir}")
    print("="*60 + "\n")
