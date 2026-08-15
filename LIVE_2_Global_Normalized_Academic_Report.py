import os
import sys
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats

# 国际标准 5参数 Logistic 非线性单调映射函数
def logistic_5param(X, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (X - b3)))) + b4 * X + b5

class DoubleStreamLogger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def compute_live2_frozen_ablation_report_pipeline(master_csv_path, base_output_dir):
    if not os.path.exists(master_csv_path):
        print(f"❌ 错误：找不到唯一的 LIVE-2 特征大表 -> {master_csv_path}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir_name = f"LIVE2_Ultimate_Refined_v_{timestamp}"
    final_archive_path = os.path.join(base_output_dir, archive_dir_name)
    os.makedirs(final_archive_path, exist_ok=True)

    log_file_path = os.path.join(final_archive_path, "LIVE2_Refined_Report_Trace.log")
    sys.stdout = DoubleStreamLogger(log_file_path)

    print("=" * 135)
    print(f"   🏆 终极精塑指标解算中枢：LIVE-2 数据集 【分失真动态参数估计 + 局部极性修正】标准学术总表系统   ")
    print("=" * 135)
    print(f"特征源大表   : {master_csv_path}")
    print(f"归档夹路径   : {final_archive_path}\n")

    df = pd.read_csv(master_csv_path)
    df.columns = df.columns.str.lower().str.strip()
    
    baseline_methods = ['brisque', 'clipiqa+', 'maniqa', 'niqe', 'nima', 'paq2piq', 'cnniqa', 'dbcnn', 'musiq', 'nrqm', 'pi']

    # 1. 跨图方法级 Z-score 标准化平权算子
    print("🔄 [🔥 大修平权] 正在对 11 种异构算法的 Consensus 列执行 Z-score 全局归一...")
    for m in baseline_methods:
        c_col = f'{m}_consensus'
        if c_col in df.columns:
            mean_val = df[c_col].mean()
            std_val = df[c_col].std()
            df[c_col] = (df[c_col] - mean_val) / (std_val + 1e-8)

    # [STAGE 1] 全局主观对账方向探测
    Y_all_mos = df['mos'].values
    adaptive_polarity = {}
    for m in baseline_methods:
        X_raw = df[f'{m}_consensus'].values
        res = scipy.stats.spearmanr(X_raw, Y_all_mos)
        corr_val = res.statistic if hasattr(res, 'statistic') else res
        adaptive_polarity[m] = 1 if corr_val < 0 else -1

    test_score_columns = []
    for m in baseline_methods:
        test_score_columns.append(f'{m}_full_raw')
        test_score_columns.append(f'{m}_consensus')
    test_score_columns.append('global_full_pool_fusion')
    test_score_columns.append('our_framework_q_pgt')

    distortion_mapping = {'FF': 'FF', 'GB': 'GB', 'JP2K': 'JP2K', 'JPEG': 'JPEG', 'WN': 'WN'}

    print("\n📊 [STAGE 2] 正在运行分失真自适应边界单调拉伸回归总线...")
    master_report_data = []

    for col in test_score_columns:
        row_metrics = {'Score_Type': col.upper().replace('_CONSENSUS', ' (Consensus)').replace('_FULL_RAW', ' (Full_Raw)')}
        
        # --- 全局 Overall 指标解算 ---
        X_all_raw = df[col].values
        Y_all = df['mos'].values
        
        x_min_all, x_max_all = np.min(X_all_raw), np.max(X_all_raw)
        X_all_scaled = (X_all_raw - x_min_all) / (x_max_all - x_min_all + 1e-8)
        
        try:
            init_g = [np.max(Y_all) - np.min(Y_all), 1.0, 0.5, 1.0, np.mean(Y_all)]
            bounds_l = ([0.0, -100.0, -5.0, -100.0, 0.0], [200.0, 100.0, 5.0, 100.0, 200.0])
            params, _ = curve_fit(logistic_5param, X_all_scaled, Y_all, p0=init_g, bounds=bounds_l, maxfev=40000)
            X_all_aligned = logistic_5param(X_all_scaled, *params)
        except:
            p = np.polyfit(X_all_scaled, Y_all, 3)
            X_all_aligned = np.polyval(p, X_all_scaled)

        res_srcc_all = scipy.stats.spearmanr(X_all_raw, Y_all)
        row_metrics['Overall_SRCC'] = abs(res_srcc_all.statistic if hasattr(res_srcc_all, 'statistic') else res_srcc_all)
        res_plcc_all = scipy.stats.pearsonr(X_all_aligned, Y_all)
        row_metrics['Overall_PLCC'] = abs(res_plcc_all.statistic if hasattr(res_plcc_all, 'statistic') else res_plcc_all)
        
        # --- 🌟 细节精塑：分失真子块独立自适应边界估计 ---
        for file_abbr, clean_label in distortion_mapping.items():
            df_sub = df[df['distortion_type'].astype(str).str.upper().str.strip() == file_abbr]
            if len(df_sub) < 5:
                row_metrics[f'{clean_label}_SRCC'] = np.nan
                row_metrics[f'{clean_label}_PLCC'] = np.nan
                continue
                
            X_sub_raw = df_sub[col].values
            Y_sub = df_sub['mos'].values
            
            # 细节微调二：局部子块内部动态极性二次对齐锁，破除假性多尺度倒置
            res_sub_direction = scipy.stats.spearmanr(X_sub_raw, Y_sub)
            sub_corr = res_sub_direction.statistic if hasattr(res_sub_direction, 'statistic') else res_sub_direction
            
            x_min_sub, x_max_sub = np.min(X_sub_raw), np.max(X_sub_raw)
            X_sub_scaled = (X_sub_raw - x_min_sub) / (x_max_sub - x_min_sub + 1e-8)
            
            # 细节微调一：拟合边界与猜测值完全基于当前子块 MOS 极值流自适应适配，释放非凸单调收敛潜能！
            mos_range_sub = np.max(Y_sub) - np.min(Y_sub)
            init_sub = [mos_range_sub, 1.0 if sub_corr < 0 else -1.0, 0.5, 1.0, np.mean(Y_sub)]
            bounds_sub = ([0.0, -150.0, -10.0, -150.0, 0.0], [mos_range_sub * 3.0, 150.0, 10.0, 150.0, 250.0])
            
            try:
                params_sub, _ = curve_fit(logistic_5param, X_sub_scaled, Y_sub, p0=init_sub, bounds=bounds_sub, maxfev=40000)
                X_sub_aligned = logistic_5param(X_sub_scaled, *params_sub)
            except:
                p_sub = np.polyfit(X_sub_scaled, Y_sub, 3)
                X_sub_aligned = np.polyval(p_sub, X_sub_scaled)
                
            res_srcc_sub = scipy.stats.spearmanr(X_sub_raw, Y_sub)
            row_metrics[f'{clean_label}_SRCC'] = abs(res_srcc_sub.statistic if hasattr(res_srcc_sub, 'statistic') else res_srcc_sub)
            res_plcc_sub = scipy.stats.pearsonr(X_sub_aligned, Y_sub)
            row_metrics[f'{clean_label}_PLCC'] = abs(res_plcc_sub.statistic if hasattr(res_plcc_sub, 'statistic') else res_plcc_sub)
            
        master_report_data.append(row_metrics)

    # 3. 过滤提炼 13 行打榜矩阵
    final_full_table = pd.DataFrame(master_report_data).reset_index(drop=True)
    paper_rows = [f'{m.upper()} (Consensus)'.upper() for m in baseline_methods] + ['GLOBAL_FULL_POOL_FUSION', 'OUR_FRAMEWORK_Q_PGT']
    final_full_table['score_type_upper'] = final_full_table['Score_Type'].astype(str).str.upper()
    final_paper_table = final_full_table[final_full_table['score_type_upper'].isin(paper_rows)].copy()
    
    final_paper_table['Score_Type'] = final_paper_table['Score_Type'].replace({
        'GLOBAL_FULL_POOL_FUSION': 'GLOBAL_FULL_POOL (W/O Purification)',
        'OUR_FRAMEWORK_Q_PGT': 'OURS (Q_PGT - Elite Purified)'
    })
    
    df_baselines_part = final_paper_table[~final_paper_table['Score_Type'].str.contains('POOL|OURS')].sort_values(by='Overall_SRCC', ascending=False)
    df_fusions_part = final_paper_table[final_paper_table['Score_Type'].str.contains('POOL|OURS')].sort_values(by='Overall_SRCC', ascending=True)
    final_paper_table = pd.concat([df_baselines_part, df_fusions_part]).reset_index(drop=True)

    output_full_csv = os.path.join(final_archive_path, "LIVE2_Refined_Ablation_Full_24_Matrix.csv")
    output_paper_csv = os.path.join(final_archive_path, "LIVE2_Refined_TIP_13_Benchmark_Table.csv")
    final_full_table.drop(columns=['score_type_upper'], errors='ignore').to_csv(output_full_csv, index=False, encoding='utf-8')
    final_paper_table.drop(columns=['score_type_upper'], errors='ignore').to_csv(output_paper_csv, index=False, encoding='utf-8')
    
    print("\n" + "=" * 125)
    print("        🏆 终极全自愈：LIVE-2 全量 13 行对照大横表 (分失真自适应参数估计 + 局部极性锁死版)         ")
    print("=" * 125)
    print(final_paper_table.drop(columns=['score_type_upper'], errors='ignore').round(4).to_string(index=False))
    print("=" * 125 + "\n")
    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # 🌟 指向您最新抗死锁大修生成的真实大表路径
    frozen_csv_in = r"C:\2025-12-04-Image_Quality_Assessments\2026-07-11-IQA-II\MOS_Extractor\LIVE_MOS\LIVE2_Master_Rebuilt_20260813_214116\LIVE_25_Scores_Unnormalized_Master.csv"
    base_dir = r"C:\2025-12-04-Image_Quality_Assessments\2026-07-11-IQA-II\MOS_Extractor\LIVE_MOS"
    compute_live2_frozen_ablation_report_pipeline(frozen_csv_in, base_dir)
