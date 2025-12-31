#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AQI T檢定統計分析工具 v1
功能：
 1. 計算各區域平均值
 2. 進行區域間的 T 檢定（兩兩比較）
 3. 輸出統計結果表格和視覺化圖表
 4. 生成詳細的統計報告
"""

import os
import glob
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import fontManager, FontProperties
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ---------- Config ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, "data")
output_dir = os.path.join(current_dir, "ttest_results")
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
    "AMB_TEMP": {"name": "環境溫度", "unit": "°C"},
    "CO": {"name": "一氧化碳", "unit": "ppm"},
    "NO": {"name": "一氧化氮", "unit": "ppb"},
    "NO2": {"name": "二氧化氮", "unit": "ppb"},
    "NOx": {"name": "氮氧化物", "unit": "ppb"},
    "O3": {"name": "臭氧", "unit": "ppb"},
    "PM10": {"name": "懸浮微粒", "unit": "μg/m³"},
    "PM2.5": {"name": "細懸浮微粒", "unit": "μg/m³"},
    "RAINFALL": {"name": "降雨量", "unit": "mm"},
    "RH": {"name": "相對濕度", "unit": "%"},
    "SO2": {"name": "二氧化硫", "unit": "ppb"},
    "WD_HR": {"name": "風向", "unit": "degrees"},
    "WIND_DIREC": {"name": "風向", "unit": "degrees"},
    "WIND_SPEED": {"name": "風速", "unit": "m/s"},
    "WS_HR": {"name": "平均風速", "unit": "m/s"}
}

# ---------- 字體設定 ----------
def set_chinese_font():
    p = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
    if os.path.exists(p):
        fontManager.addfont(p)
        prop = FontProperties(fname=p)
        found_name = prop.get_name()
        if found_name:
            mpl.rcParams['font.family'] = 'sans-serif'
            mpl.rcParams['font.sans-serif'] = [found_name]
            print(f"✅ 字體設定: {found_name}")
            return found_name
    print("⚠️ 使用預設字體")
    return None

set_chinese_font()

# ---------- 計算區域平均 ----------
def calculate_regional_means(df):
    """計算各區域的平均值"""
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    
    regional_data = {}
    
    for region_name, region_sites in areas.items():
        mask = df['site'].isin(region_sites)
        if not mask.any():
            continue
        
        # 計算該區域的平均值（按 datetime 和 item 分組）
        region_df = (df[mask]
                    .groupby(['datetime', 'item'], as_index=False)
                    ['value'].mean(numeric_only=True))
        region_df['region'] = region_name
        regional_data[region_name] = region_df
    
    if regional_data:
        return pd.concat(regional_data.values(), ignore_index=True)
    return pd.DataFrame()

# ---------- T檢定分析 ----------
def perform_ttest_analysis(df, item):
    """對單一測項進行所有區域間的T檢定"""
    item_data = df[df['item'] == item]
    regions = item_data['region'].unique()
    
    results = []
    
    # 兩兩比較所有區域
    for r1, r2 in combinations(regions, 2):
        data1 = item_data[item_data['region'] == r1]['value'].dropna()
        data2 = item_data[item_data['region'] == r2]['value'].dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            continue
        
        # 進行獨立樣本 T 檢定
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        # 計算效應量 (Cohen's d)
        pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
        cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'item': item,
            'region_1': r1,
            'region_2': r2,
            'mean_1': data1.mean(),
            'mean_2': data2.mean(),
            'std_1': data1.std(),
            'std_2': data2.std(),
            'n_1': len(data1),
            'n_2': len(data2),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'significance_level': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })
    
    return pd.DataFrame(results)

# ---------- 繪製箱型圖 ----------
def plot_boxplot(df, item, output_path):
    """繪製各區域的箱型圖比較"""
    item_data = df[df['item'] == item].copy()
    
    if item_data.empty:
        return
    
    info = items_info.get(item, {'name': item, 'unit': ''})
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用 seaborn 繪製箱型圖
    sns.boxplot(data=item_data, x='region', y='value', ax=ax, palette='Set2')
    sns.stripplot(data=item_data, x='region', y='value', ax=ax, 
                 color='black', alpha=0.3, size=2)
    
    ax.set_title(f"{info['name']} - 區域比較", fontsize=16, fontweight='bold')
    ax.set_xlabel("區域", fontsize=12)
    ax.set_ylabel(f"{info['name']} ({info['unit']})", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------- 繪製T檢定熱圖 ----------
def plot_ttest_heatmap(ttest_results, item, output_path):
    """繪製T檢定p值的熱圖"""
    if ttest_results.empty:
        return
    
    info = items_info.get(item, {'name': item, 'unit': ''})
    regions = sorted(set(ttest_results['region_1'].unique()) | 
                    set(ttest_results['region_2'].unique()))
    
    # 創建p值矩陣
    p_matrix = pd.DataFrame(1.0, index=regions, columns=regions)
    
    for _, row in ttest_results.iterrows():
        p_matrix.loc[row['region_1'], row['region_2']] = row['p_value']
        p_matrix.loc[row['region_2'], row['region_1']] = row['p_value']
    
    # 對角線設為 NaN（自己和自己比較）
    for r in regions:
        p_matrix.loc[r, r] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用 -log10(p) 來視覺化（越大表示越顯著）
    plot_data = -np.log10(p_matrix.astype(float))
    
    sns.heatmap(plot_data, annot=p_matrix, fmt='.4f', cmap='RdYlGn_r', 
                center=1.3, vmin=0, vmax=3, ax=ax, cbar_kws={'label': '-log10(p-value)'})
    
    ax.set_title(f"{info['name']} - T檢定 P值熱圖\n(值越大越顯著, p<0.05為顯著)", 
                fontsize=14, fontweight='bold')
    
    # 添加顯著性閾值線
    ax.axhline(y=0, color='red', linewidth=2, linestyle='--', alpha=0.5)
    ax.text(len(regions), 0.5, 'p=0.05', color='red', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------- 生成統計報告 ----------
def generate_report(all_results, output_path):
    """生成詳細的統計報告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("AQI 區域間 T檢定統計分析報告\n")
        f.write("=" * 80 + "\n\n")
        
        for item in all_results['item'].unique():
            item_results = all_results[all_results['item'] == item]
            info = items_info.get(item, {'name': item, 'unit': ''})
            
            f.write(f"\n{'='*80}\n")
            f.write(f"測項: {info['name']} ({item})\n")
            f.write(f"單位: {info['unit']}\n")
            f.write(f"{'='*80}\n\n")
            
            # 各區域描述統計
            f.write("區域描述統計:\n")
            f.write("-" * 80 + "\n")
            for region in item_results['region_1'].unique():
                region_data = item_results[item_results['region_1'] == region].iloc[0]
                f.write(f"  {region}: 平均={region_data['mean_1']:.2f}, "
                       f"標準差={region_data['std_1']:.2f}, "
                       f"樣本數={region_data['n_1']}\n")
            f.write("\n")
            
            # T檢定結果
            f.write("T檢定結果 (兩兩比較):\n")
            f.write("-" * 80 + "\n")
            significant_count = item_results['significant'].sum()
            f.write(f"總比較次數: {len(item_results)}\n")
            f.write(f"顯著差異數: {significant_count} ({significant_count/len(item_results)*100:.1f}%)\n\n")
            
            # 顯著差異的配對
            sig_results = item_results[item_results['significant']].sort_values('p_value')
            if not sig_results.empty:
                f.write("顯著差異的配對:\n")
                for _, row in sig_results.iterrows():
                    diff = row['mean_1'] - row['mean_2']
                    f.write(f"  {row['region_1']} vs {row['region_2']}: "
                           f"t={row['t_statistic']:.3f}, p={row['p_value']:.4f} {row['significance_level']}, "
                           f"差異={diff:.2f}, Cohen's d={row['cohens_d']:.3f}\n")
            else:
                f.write("  無顯著差異\n")
            
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("註:\n")
        f.write("  *** : p < 0.001 (極顯著)\n")
        f.write("  **  : p < 0.01  (非常顯著)\n")
        f.write("  *   : p < 0.05  (顯著)\n")
        f.write("  ns  : p ≥ 0.05  (不顯著)\n")
        f.write("  Cohen's d: 效應量指標 (|d|>0.8為大效應, 0.5-0.8中效應, 0.2-0.5小效應)\n")
        f.write("=" * 80 + "\n")

# ---------- 處理單檔 ----------
def process_file(file_path):
    """處理單一CSV檔案"""
    file_name = os.path.basename(file_path)
    file_base = os.path.splitext(file_name)[0]
    
    print(f"\n📊 處理: {file_name}")
    
    # 讀取資料
    df_raw = pd.read_csv(file_path)
    
    # 計算區域平均
    print("  計算區域平均...")
    regional_df = calculate_regional_means(df_raw)
    
    if regional_df.empty:
        print("  ⚠️ 無區域資料")
        return
    
    # 創建輸出目錄
    file_output_dir = os.path.join(output_dir, file_base)
    os.makedirs(file_output_dir, exist_ok=True)
    
    # 對每個測項進行分析
    items = regional_df['item'].unique()
    all_results = []
    
    print(f"  分析 {len(items)} 個測項...")
    for item in tqdm(items, desc="  T檢定", unit="項"):
        # T檢定
        ttest_results = perform_ttest_analysis(regional_df, item)
        
        if not ttest_results.empty:
            all_results.append(ttest_results)
            
            # 繪製箱型圖
            boxplot_path = os.path.join(file_output_dir, f"{item}_boxplot.png")
            plot_boxplot(regional_df, item, boxplot_path)
            
            # 繪製熱圖
            heatmap_path = os.path.join(file_output_dir, f"{item}_ttest_heatmap.png")
            plot_ttest_heatmap(ttest_results, item, heatmap_path)
    
    # 合併所有結果
    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        
        # 儲存CSV
        csv_path = os.path.join(file_output_dir, "ttest_results.csv")
        all_results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 生成報告
        report_path = os.path.join(file_output_dir, "statistical_report.txt")
        generate_report(all_results_df, report_path)
        
        print(f"  ✅ 完成! 輸出至: {file_output_dir}")
        print(f"     - {len(items)} 個測項")
        print(f"     - {len(all_results_df)} 個配對比較")
        print(f"     - {all_results_df['significant'].sum()} 個顯著差異")

# ---------- 主程式 ----------
if __name__ == '__main__':
    print("=" * 80)
    print("AQI 區域間 T檢定統計分析工具")
    print("=" * 80)
    
    pattern = os.path.join(base_dir, 'hourly_2019*.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("⚠️ 找不到符合條件的檔案")
        exit(1)
    
    print(f"\n📁 找到 {len(files)} 個檔案")
    
    # 處理所有檔案
    for fp in tqdm(files, desc='處理檔案', unit='file'):
        try:
            process_file(fp)
        except Exception as e:
            tqdm.write(f"❌ 處理失敗 [{os.path.basename(fp)}]: {e}")
    
    print(f"\n{'='*80}")
    print(f"🎉 分析完成!")
    print(f"📂 結果目錄: {output_dir}")
    print(f"{'='*80}")
    print("\n輸出檔案說明:")
    print("  - *_boxplot.png : 箱型圖（顯示各區域分布）")
    print("  - *_ttest_heatmap.png : T檢定熱圖（顯示顯著性）")
    print("  - ttest_results.csv : 完整統計結果")
    print("  - statistical_report.txt : 統計分析報告")
