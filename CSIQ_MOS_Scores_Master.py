import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge

# 11 种异构基准算法的标准物理极性 (1:越大越好, -1:越小越好)
POLARITY_DICT = {
    'brisque': -1, 'clipiqa+': 1, 'maniqa': 1, 'niqe': -1, 'nima': 1,
    'paq2piq': 1, 'cnniqa': 1, 'dbcnn': 1, 'musiq': 1, 'nrqm': 1, 'pi': -1
}
IQA_METHODS = list(POLARITY_DICT.keys())

# 零偏置阻尼动力学控制方程
def decay_model(x, Q_0, Q_pure, alpha, gamma):
    xi = 1e-7
    abundance_ratio = (1.0 - x) / (x + xi)
    return Q_0 + (Q_pure - Q_0) * np.exp(-alpha * (abundance_ratio ** gamma))

def freeze_raw_unnormalized_matrix_precise(dmos_csv_path, experiment_dir, output_csv_path):
    """
    根据主表的 image_id 精准无缝拼接文件名，提取 24 维【纯原始未全局归一化】的粗得分并固化落盘。
    """
    if not os.path.exists(dmos_csv_path):
        print(f"❌ 错误：找不到基准 DMOS 文件 -> {dmos_csv_path}")
        return

    # 1. 读取干净的 DMOS 主表作为对齐基准（包含行标 image_id 和 dmos）
    df_master = pd.read_csv(dmos_csv_path)
    
    # 建立全局成果收集字典
    frozen_records = {}

    # 论文中被锁定的 7 种精英方法集合 (用于最终方法的提纯融合)
    ELITE_SET = {'pi', 'clipiqa+', 'nrqm', 'musiq', 'nima', 'brisque', 'niqe'}

    print("🚂 P-GT 粗结果固化引擎启动，正在进行【全匹配路径检测】...")
    
    success_count = 0
    missing_count = 0

    # 2. 核心逻辑修改：以主表为绝对基准，直接利用 image_id 拼出对应的目标 CSV 文件
    for _, row in df_master.iterrows():
        image_id = str(row['image_id']).strip()
        
        # 🌟 精准无缝拼接： 1600.AWGN.1 -> 1600.AWGN.1_IQA_Experiment.csv
        target_csv_name = f"{image_id}_IQA_Experiment.csv"
        full_csv_path = os.path.join(experiment_dir, target_csv_name)
        
        # 安全防御：如果有些高级失真（如 contrast 5）你没生成或者文件不存在，安全跳过
        if not os.path.exists(full_csv_path):
            missing_count += 1
            continue
            
        try:
            df_raw = pd.read_csv(full_csv_path)
            if len(df_raw) == 0: 
                continue
            
            # 列名清洗脱敏
            df_raw.columns = df_raw.columns.str.lower().str.strip()
            id_col = 'id' if 'id' in df_raw.columns else (
                'id_numeric' if 'id_numeric' in df_raw.columns else df_raw.columns[0])

            df_raw['id_str'] = df_raw[id_col].astype(str).str.lower().str.strip()
            df_min = df_raw[df_raw['id_str'] == 'min']
            df_max = df_raw[df_raw['id_str'] == 'max']
            
            if df_min.empty and len(df_raw) > 0: 
                df_min = df_raw.iloc[0:1]
            
            df_raw['id_num'] = pd.to_numeric(df_raw['id_str'], errors='coerce')
            df_samples = df_raw[df_raw['id_num'].between(2.0, 101.0)].copy()

            if len(df_samples) < 3: 
                continue

            # 构建多元空间高阶矩控制矩阵 Gamma
            x1_min, y1_min, x2_min, y2_min = df_min[['x1', 'y1', 'x2', 'y2']].values.flatten()
            mu_target_x, mu_target_y = (x1_min + x2_min) / 2.0, (y1_min + y2_min) / 2.0
            x1, y1 = np.nan_to_num(df_samples['x1'].values), np.nan_to_num(df_samples['y1'].values)
            x2, y2 = np.nan_to_num(df_samples['x2'].values, nan=1.0), np.nan_to_num(df_samples['y2'].values, nan=1.0)

            if 'area_ratio' in df_samples.columns:
                x_k = np.nan_to_num(df_samples['area_ratio'].values, nan=0.5)
            else:
                x_k = np.clip((x2 - x1) * (y2 - y1) / ((x2.max() - x1.min()) * (y2.max() - y1.min()) + 1e-5), 0.01, 1.0)

            mu_x, mu_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            D_diag = np.sqrt((x2.max() - x1.min()) ** 2 + (y2.max() - y1.min()) ** 2) + 1e-5
            C_k = np.sqrt((mu_x - mu_target_x) ** 2 + (mu_y - mu_target_y) ** 2) / D_diag
            
            with np.errstate(divide='ignore', invalid='ignore'):
                aspect_ratios = (x2 - x1) / (y2 - y1 + 1e-5)
            mean_ar = np.nanmean(aspect_ratios)
            std_ar = np.nanstd(aspect_ratios) + 1e-5
            S_k = np.nan_to_num(((aspect_ratios - mean_ar) ** 3) / (std_ar ** 3))
            k_k = np.nan_to_num(((aspect_ratios - mean_ar) ** 4) / (std_ar ** 4) - 3)
            Gamma = np.stack([C_k, S_k, k_k], axis=1)

            # 初始化单图像字典
            frozen_records[image_id] = {}

            full_auc_list = []
            full_weight_list = []
            elite_auc_list = []
            elite_weight_list = []

            # 3. 压榨 11 种基准算法的原始无偏差粗标量分值
            for method in IQA_METHODS:
                if method not in df_samples.columns: df_samples[method] = 0.0
                if method not in df_max.columns: df_max[method] = 0.0

                sample_scores = np.nan_to_num(df_samples[method].values)
                max_val = float(df_max[method].iloc[0]) if not df_max.empty else float(np.nanmean(sample_scores))

                # 🌟 【粗结果 1】：直接汇报基线 max 采样框的纯原始得分
                frozen_records[image_id][f'{method.upper()}_Max_Raw'] = max_val

                # 定理一：空间矩投影矩阵多维洗涤
                ridge = Ridge(alpha=1e-4)
                ridge.fit(Gamma, sample_scores)
                Q_tilde = sample_scores - ridge.predict(Gamma) + np.mean(sample_scores)

                q_0_guess = np.clip(np.min(Q_tilde), -1e5, np.max(Q_tilde))
                q_pure_guess = np.clip(np.max(Q_tilde), -1e5, np.max(Q_tilde)*2)
                if q_pure_guess <= q_0_guess: q_pure_guess = q_0_guess + 1e-3

                p0 = [q_0_guess, q_pure_guess, 1.0, 1.0]
                bounds = ([-np.inf, -np.inf, 0.01, 0.01], [np.inf, np.inf, 10.0, 5.0])

                try:
                    popt, pcov = curve_fit(decay_model, x_k, Q_tilde, p0=p0, bounds=bounds, maxfev=4000)
                    diag_vars = np.diag(pcov) if pcov is not None else np.array([1e-5, 1e-5, 1e-5, 1e-5])
                    fit_se = float(np.sqrt(np.abs(diag_vars)))
                except:
                    fit_se = 99.0

                sort_idx = np.argsort(x_k)
                auc_mdc = float(np.trapz(Q_tilde[sort_idx], x_k[sort_idx]))

                # 🌟 【粗结果 2】：直接汇报动力学解算出来的原始共识退化能分
                frozen_records[image_id][f'{method.upper()}_Consensus'] = auc_mdc
                
                # 数据原始分流存储
                full_auc_list.append(auc_mdc)
                full_weight_list.append(1.0 / (fit_se ** 2 + 1e-5))
                if method in ELITE_SET:
                    elite_auc_list.append(auc_mdc)
                    elite_weight_list.append(1.0 / (fit_se ** 2 + 1e-5))

            # ----------------------------------------------------
            # 🌟 【粗结果 3】：11 种基线【全量无洗涤群体融合】原始分（1 个）
            # ----------------------------------------------------
            f_auc = np.array(full_auc_list)
            f_w = np.array(full_weight_list)
            f_med = np.median(f_auc)
            f_mad = np.median(np.abs(f_auc - f_med)) + 1e-6
            f_mask = np.exp(-((f_auc - f_med) ** 2) / (2 * (1.4 * f_mad) ** 2))
            f_cw = f_w * f_mask
            frozen_records[image_id]['GLOBAL_FULL_POOL_FUSION'] = np.sum(f_cw * f_auc) / (np.sum(f_cw) + 1e-8)

            # ----------------------------------------------------
            # 🌟 【粗结果 4】：11 种基线【精英提纯后】解算出的系统总分（本文方法粗得分，1 个）
            # ----------------------------------------------------
            e_auc = np.array(elite_auc_list)
            e_w = np.array(elite_weight_list)
            e_med = np.median(e_auc)
            e_mad = np.median(np.abs(e_auc - e_med)) + 1e-6
            e_mask = np.exp(-((e_auc - e_med) ** 2) / (2 * (1.4 * e_mad) ** 2))
            e_cw = e_w * e_mask
            frozen_records[image_id]['OUR_FRAMEWORK_Q_PGT'] = np.sum(e_cw * e_auc) / (np.sum(e_cw) + 1e-8)

            success_count += 1

        except Exception as e:
            print(f"❌ 读取具体样本 {image_id} 发生未知结构阻塞: {e}")

    # 4. 数据转换与大总表固化
    df_features = pd.DataFrame.from_dict(frozen_records, orient='index').reset_index()
    df_features = df_features.rename(columns={'index': 'image_id'})
    
    # 拼回主表
    df_raw_master = pd.merge(df_master, df_features, on='image_id', how='inner').dropna()

    # 创建目标子文件夹并写出
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_raw_master.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"\n" + "="*65)
    print(f"📁 【无偏差原始粗结果精准匹配完成】！")
    print(f"   主表读取总行数: {len(df_master)} 行")
    print(f"   本地硬盘成功对接并计算的有效图像文件: {success_count} 张")
    print(f"   未在文件夹下找到的残留冗余项目(如contrast 4/5): {missing_count} 个")
    print(f"   最终大表成功锁死行数：{len(df_raw_master)} 行（含dmos标签与24个原始得分维度）")
    print(f"   成果路径 -> {output_csv_path}")
    print("="*65 + "\n")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # 路径完全同步自你的本地物理路径
    master_csv = r"C:\2025-12-04-Image_Quality_Assessments\2026-07-11-IQA-II\MOS_Extractor\MOS_ExtractorCSIQ_Extracted_DMOS.csv"
    experiment_dir = r"C:\2025-12-04-Image_Quality_Assessments\2026-08-03-IQA-Results-Public-Image_Sets\2026-08-03-IQA-Results-Public-Image-Sets\Experiment_v_20260728_214924\CSIQ_image_tables"
    raw_output_path = r"C:\2025-12-04-Image_Quality_Assessments\2026-07-11-IQA-II\MOS_Extractor\CSIQ_24_Scores_Unnormalized_Master.csv"
    
    freeze_raw_unnormalized_matrix_precise(master_csv, experiment_dir, raw_output_path)
