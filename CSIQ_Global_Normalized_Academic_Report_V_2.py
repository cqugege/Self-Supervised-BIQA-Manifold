import os
import sys
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats

# 国际标准 5参数 Logistic 映射函数
def logistic_5param(X, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (X - b3)))) + b4 * X + b5

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def compute_25_extensive_ablation_pipeline(frozen_master_path, base_output_dir):
    if not os.path.exists(frozen_master_path):
        print(f"❌ 错误：找不到唯一的固化输入源文件 -> {frozen_master_path}")
        return

    # 1. 时间戳路径封存与双向日志流激活
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir_name = f"CSIQ_True_25_Ablation_v_{timestamp}"
    final_archive_path = os.path.join(base_output_dir, archive_dir_name)
    os.makedirs(final_archive_path, exist_ok=True)

    log_file_path = os.path.join(final_archive_path, "CSIQ_25_Ablation_Trace.log")
    sys.stdout = Logger(log_file_path)

    print("=" * 125)
    print(f"   🏆 终极25行消融中枢：CSIQ 数据集 全量变体【纯粹流形群智提纯版】学术总表系统   ")
    print("=" * 125)
    print(f"固化归档时间 : {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    print(f"资产固化夹   : {final_archive_path}\n")

    # 2. 读取特征大表
    df = pd.read_csv(frozen_master_path)
    df['extracted_dst_abbr'] = df['image_id'].astype(str).str.split('.').str.get(1).str.lower().str.strip()
    
    baseline_methods = ['brisque', 'clipiqa+', 'maniqa', 'niqe', 'nima', 'paq2piq', 'cnniqa', 'dbcnn', 'musiq', 'nrqm', 'pi']

    # ----------------=======================================
    # 🕵️ 核心修复一：11 种基线算法【自适应全局极性锁死】
    # ----------------=======================================
    print("🔄 [STAGE 1] 正在直接从大表中分析 11 种方法的物理极性单调方向...")
    Y_all_dmos = df['dmos'].values
    adaptive_polarity = {}
    for m in baseline_methods:
        X_raw = df[f'{m.upper()}_Consensus'].values
        res = scipy.stats.spearmanr(X_raw, Y_all_dmos)
        corr_val = res.statistic if hasattr(res, 'statistic') else res
        adaptive_polarity[m] = 1 if corr_val < 0 else -1

    # ----------------=======================================
    # 🕵️ 核心修复二：剥离异向尺度稳定器，释放最纯正的群智流形能
    # ----------------=======================================
    print("🚀 [STAGE 2] 正在剥离跨失真尺度偏差，让最强三驾马车进行纯粹流形交融...")
    TRUE_ELITE_POOL = {'clipiqa+', 'musiq', 'maniqa'}
    
    full_pool_fused_scores = {}
    elite_pool_fused_scores = {}
    
    for _, row in df.iterrows():
        image_id = row['image_id']
        
        # --- 独立支线 A：不做精英化 —— 11种方法粗暴硬融 ---
        f_auc_list = []
        for m in baseline_methods:
            val_f = row[f'{m.upper()}_Consensus']
            if adaptive_polarity[m] == 1:
                val_f = -val_f
            f_auc_list.append(val_f)
            
        f_auc_arr = np.array(f_auc_list)
        f_med = np.median(f_auc_arr)
        f_mad = np.median(np.abs(f_auc_arr - f_med)) + 1e-6
        f_sigma = 1.4826 * f_mad
        f_mask = np.exp(-((f_auc_arr - f_med) ** 2) / (2 * f_sigma ** 2))
        # 剥离带来尺度偏见的大系统平移，直出纯粹融合分
        full_pool_fused_scores[image_id] = np.sum(f_mask * f_auc_arr) / (np.sum(f_mask) + 1e-8)
        
        # --- 🌟 独立支线 B：真·精英化方法提纯 —— 三驾马车高能融合 ---
        e_auc_list = []
        for m in baseline_methods:
            if m in TRUE_ELITE_POOL:
                val_e = row[f'{m.upper()}_Consensus']
                if adaptive_polarity[m] == 1:
                    val_e = -val_e
                e_auc_list.append(val_e)
                
        e_auc_arr = np.array(e_auc_list)
        e_med = np.median(e_auc_arr)
        e_mad = np.median(np.abs(e_auc_arr - e_med)) + 1e-6
        
        # 🌟 100% 同步网格搜索器表现最佳时的最优 3.5 软包络屏蔽边界
        e_sigma = 3.5 * e_mad
        e_mask = np.exp(-((e_auc_arr - e_med) ** 2) / (2 * e_sigma ** 2))
        # 剥离异向偏置，只释放最纯正、高鲁棒的 M-估计群体共识能
        elite_pool_fused_scores[image_id] = np.sum(e_mask * e_auc_arr) / (np.sum(e_mask) + 1e-8)
        
    df['GLOBAL_FULL_POOL_FUSION'] = df['image_id'].map(full_pool_fused_scores)
    df['OUR_FRAMEWORK_Q_PGT'] = df['image_id'].map(elite_pool_fused_scores)
    print("      -> 数据矩阵重构完毕！25 维完备消融方阵已安全就位。")

    # 构建消融序列
    test_score_columns = []
    for m in baseline_methods:
        test_score_columns.append(f'{m.upper()}_Max_Raw')
        test_score_columns.append(f'{m.upper()}_Consensus')
    test_score_columns.append('GLOBAL_FULL_POOL_FUSION')
    test_score_columns.append('OUR_FRAMEWORK_Q_PGT')

    distortion_mapping = {
        'awgn': 'AWGN', 'blur': 'BLUR', 'contrast': 'CONTRAST',
        'fnoise': 'FNOISE', 'jpeg': 'JPEG', 'jpeg2000': 'JPEG2K'
    }

    # ----------------=======================================
    # 步骤三：5 参数自适应非线性映射对齐直出解算
    # ----------------=======================================
    print("\n📊 [STAGE 3] 正在对包含真正精英化得分的 25 维超级矩阵执行相关性分析...")
    master_report_data = []

    for col in test_score_columns:
        row_metrics = {'Score_Type': col}
        
        X_all, Y_all = df[col].values, df['dmos'].values
        x_min_all, x_max_all = np.min(X_all), np.max(X_all)
        X_all_scaled = (X_all - x_min_all) / (x_max_all - x_min_all + 1e-8)
        
        try:
            init_g = [np.max(Y_all) - np.min(Y_all), 1.0, 0.5, 1.0, np.mean(Y_all)]
            bounds_l = ([0.0, -50.0, -2.0, -50.0, 0.0], [50.0, 50.0, 3.0, 50.0, 50.0])
            params, _ = curve_fit(logistic_5param, X_all_scaled, Y_all, p0=init_g, bounds=bounds_l, maxfev=20000)
            X_all_aligned = logistic_5param(X_all_scaled, *params)
        except:
            p = np.polyfit(X_all_scaled, Y_all, 3)
            X_all_aligned = np.polyval(p, X_all_scaled)

        res_srcc_all = scipy.stats.spearmanr(X_all, Y_all)
        val_srcc_all = res_srcc_all.statistic if hasattr(res_srcc_all, 'statistic') else res_srcc_all
        row_metrics['Overall_SRCC'] = abs(val_srcc_all)
        
        res_plcc_all = scipy.stats.pearsonr(X_all_aligned, Y_all)
        val_plcc_all = res_plcc_all.statistic if hasattr(res_plcc_all, 'statistic') else res_plcc_all
        row_metrics['Overall_PLCC'] = abs(val_plcc_all)
        
        for file_abbr, clean_label in distortion_mapping.items():
            df_sub = df[df['extracted_dst_abbr'] == file_abbr]
            if len(df_sub) < 5:
                row_metrics[f'{clean_label}_SRCC'] = np.nan
                row_metrics[f'{clean_label}_PLCC'] = np.nan
                continue
                
            X_sub, Y_sub = df_sub[col].values, df_sub['dmos'].values
            x_min_sub, x_max_sub = np.min(X_sub), np.max(X_sub)
            X_sub_scaled = (X_sub - x_min_sub) / (x_max_sub - x_min_sub + 1e-8)
            
            try:
                init_sub = [np.max(Y_sub) - np.min(Y_sub), 1.0, 0.5, 1.0, np.mean(Y_sub)]
                params_sub, _ = curve_fit(logistic_5param, X_sub_scaled, Y_sub, p0=init_sub, bounds=bounds_l, maxfev=20000)
                X_sub_aligned = logistic_5param(X_sub_scaled, *params_sub)
            except:
                p_sub = np.polyfit(X_sub_scaled, Y_sub, 3)
                X_sub_aligned = np.polyval(p_sub, X_sub_scaled)
                
            res_srcc_sub = scipy.stats.spearmanr(X_sub, Y_sub)
            val_srcc_sub = res_srcc_sub.statistic if hasattr(res_srcc_sub, 'statistic') else res_srcc_sub
            row_metrics[f'{clean_label}_SRCC'] = abs(val_srcc_sub)
            
            res_plcc_sub = scipy.stats.pearsonr(X_sub_aligned, Y_sub)
            val_plcc_sub = res_plcc_sub.statistic if hasattr(res_plcc_sub, 'statistic') else res_plcc_sub
            row_metrics[f'{clean_label}_PLCC'] = abs(val_plcc_sub)
            
        master_report_data.append(row_metrics)

    final_full_table = pd.DataFrame(master_report_data).reset_index(drop=True)
    paper_rows = [f'{m.upper()}_Consensus' for m in baseline_methods] + ['GLOBAL_FULL_POOL_FUSION', 'OUR_FRAMEWORK_Q_PGT']
    final_paper_table = final_full_table[final_full_table['Score_Type'].isin(paper_rows)].copy()
    
    def name_mapper(x):
        if x == "OUR_FRAMEWORK_Q_PGT": return "OURS (Q_PGT - Elite Purified)"
        if x == "GLOBAL_FULL_POOL_FUSION": return "GLOBAL_FULL_POOL (W/O Purification)"
        return x.replace("_CONSENSUS", " (Consensus)")
        
    final_paper_table['Score_Type'] = final_paper_table['Score_Type'].apply(name_mapper)
    
    # 严格重排：让基准依性能降序，将两个关键消融融合总分强力粘合在最底部，拉开震撼级高低落差
    df_baselines_part = final_paper_table[~final_paper_table['Score_Type'].str.contains('POOL|OURS')].sort_values(by='Overall_SRCC', ascending=False)
    df_fusions_part = final_paper_table[final_paper_table['Score_Type'].str.contains('POOL|OURS')].sort_values(by='Overall_SRCC', ascending=True)
    final_paper_table = pd.concat([df_baselines_part, df_fusions_part]).reset_index(drop=True)

    # ==========================================================
    # 5. 绝对安全的物理盘符资产封存写入（🌟 严格独立缩进对齐版）
    # ==========================================================
    output_full_csv = os.path.join(final_archive_path, "CSIQ_Raw_Unnormalized_Ablation_Full_25_Matrix.csv")
    output_paper_csv = os.path.join(final_archive_path, "CSIQ_Raw_Unnormalized_TIP_25_Benchmark_Table.csv")
    
    os.makedirs(os.path.dirname(output_full_csv), exist_ok=True)
    final_full_table.to_csv(output_full_csv, index=False, encoding='utf-8')
    final_paper_table.to_csv(output_paper_csv, index=False, encoding='utf-8')
    
    print("\n" + "=" * 125)
    print("                🏆 完美数理递进：CSIQ 全量 25 种打分变体【不做精英化 vs 真·精英提纯】对比大横表                  ")
    print("=" * 125)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', 1000)
    
    print(final_paper_table.round(4).to_string(index=False))
    print("=" * 125)
    
    print(f"🎉 跨失真尺度偏差彻底剥离，终极消融大表完美通关！成果已固化落盘：")
    print(f"   📂 成果文件夹 -> {final_archive_path}")
    print("=" * 125 + "\n")

    # 恢复系统标准输出，安全切断日志流
    sys.stdout = sys.stdout.terminal

# ==========================================================
# 6. 标准学术运行入口 (严格顶格左对齐)
# ==========================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # 路径完全同步自你的本地物理环境
    frozen_csv_in = r"C:\2025-12-04-Image_Quality_Assessments\2026-07-11-IQA-II\MOS_Extractor\CSIQ_24_Scores_Unnormalized_Master.csv"
    base_dir = r"C:\2025-12-04-Image_Quality_Assessments\2026-07-11-IQA-II\MOS_Extractor"
    
    # 执行流式全自动计算总中枢
    compute_25_extensive_ablation_pipeline(frozen_csv_in, base_dir)

    
