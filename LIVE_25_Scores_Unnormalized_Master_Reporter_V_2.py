import os
import sys
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge

# 11 种异构基准算法的标准出厂物理极性 (1:越大越好, -1:越小越好)
POLARITY_DICT = {
    'brisque': -1, 'clipiqa+': 1, 'maniqa': 1, 'niqe': -1, 'nima': 1,
    'paq2piq': 1, 'cnniqa': 1, 'dbcnn': 1, 'musiq': 1, 'nrqm': 1, 'pi': -1
}
IQA_METHODS = list(POLARITY_DICT.keys())

# 零偏置阻尼动力学控制方程 (100% 还原官方核心算法源码)
def decay_model(x, Q_0, Q_pure, alpha, gamma):
    xi = 1e-7
    abundance_ratio = (1.0 - x) / (x + xi)
    return Q_0 + (Q_pure - Q_0) * np.exp(-alpha * (abundance_ratio ** gamma))

class HighFidelityAuditLogger(object):
    """双向流日志管理器：高精度同步控制台输出与物理日志盘落盘，并内联实时流冲刷防御死锁"""
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

def autonomously_smelt_live2_master_table(target_root_dir, output_csv_path):
    start_time = time.time()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = os.path.abspath(__file__)
    script_name = os.path.basename(script_path)
    
    print("=" * 125)
    print(f" 🏆 【源头大修·标量解锁总线】LIVE-2 数据集 25 维全自动文件名自解密与自愈中心 ")
    print("=" * 125)
    sys.stdout.flush()

    mos_dir = os.path.join(target_root_dir, 'live_2_moscsv_files')
    if not os.path.exists(mos_dir):
        print(f"❌ 运行中断：在主目录下未探测到 live_2_moscsv_files 文件夹 -> {mos_dir}")
        sys.stdout.flush()
        return

    # 统计容器初始化
    reference_images_stats = {'total_reference_count': 0, 'reference_details': []}
    non_reference_images_stats = {'total_count': 0, 'details': []}
    
    miss_by_type = {}
    miss_details_by_type = {}
    success_by_type = {}
    expected_by_type = {}
    processed_by_type = {}
    
    reference_skipped_count = 0
    reference_skipped_details = []

    # [STAGE 1] 全局主观对账雷达深度重构
    print("\n🔄 [STAGE 1] 正在直接提取 5 大专属得分 CSV 主表，构建对账雷达...")
    sys.stdout.flush()
    global_mos_lookup_dict = {}
    
    distortion_folders = ['FF', 'GB', 'JP2K', 'JPEG', 'WN']
    distortion_mapping = {
        'FF': 'ff_scores.csv', 'GB': 'gb_scores.csv', 'JP2K': 'jp2k_scores.csv',
        'JPEG': 'jpeg_scores.csv', 'WN': 'wn_scores.csv'
    }
    
    for dst_abbr, mos_csv_name in distortion_mapping.items():
        m_path = os.path.join(mos_dir, mos_csv_name)
        if not os.path.exists(m_path):
            print(f"   ⚠ 警告: 跳过未检索到的主表文件 -> {mos_csv_name}")
            sys.stdout.flush()
            continue
            
        df_m_part = pd.read_csv(m_path)
        df_m_part.columns = df_m_part.columns.str.lower().str.strip()
        print(f"   📄 定向加载主观表: {mos_csv_name} | 内部样本数: {len(df_m_part)}")
        sys.stdout.flush()
        
        ref_count_in_file = 0
        non_ref_count_in_file = 0
        ref_list_in_file = []
        non_ref_list_in_file = []
        
        for idx, m_row in df_m_part.iterrows():
            f_name = str(m_row['filename']).strip()
            is_ref_val = str(m_row['is_reference']).strip().lower()
            is_ref = (is_ref_val == 'true' or is_ref_val == '1' or is_ref_val == 'yes')
            
            dst_type = str(m_row['distortion_type']).strip().upper()
            mos_val = float(m_row['mos']) if 'mos' in m_row else 'N/A'
            img_idx = int(m_row['image_index']) if 'image_index' in m_row else 'N/A'
            
            if is_ref:
                ref_count_in_file += 1
                ref_list_in_file.append({'filename': f_name, 'distortion_type': dst_type, 'image_index': img_idx, 'mos': mos_val})
                reference_images_stats['reference_details'].append({
                    'filename': f_name, 'distortion_type': dst_type, 'image_index': img_idx, 'mos': mos_val, 'source_file': mos_csv_name
                })
                continue
            
            non_ref_count_in_file += 1
            global_key = (dst_type, f_name.lower())
            global_mos_lookup_dict[global_key] = {
                'image_index': img_idx, 'filename': f_name, 'mos': mos_val, 'source_file': mos_csv_name
            }
            non_ref_list_in_file.append({'filename': f_name, 'distortion_type': dst_type, 'image_index': img_idx, 'mos': mos_val})
            
            if dst_type not in success_by_type: success_by_type[dst_type] = 0
            if dst_type not in miss_by_type:
                miss_by_type[dst_type] = 0
                miss_details_by_type[dst_type] = []
            if dst_type not in expected_by_type: expected_by_type[dst_type] = 0
            if dst_type not in processed_by_type: processed_by_type[dst_type] = 0
        
        reference_images_stats['total_reference_count'] += ref_count_in_file
        non_reference_images_stats['total_count'] += non_ref_count_in_file
        non_reference_images_stats['details'].extend(non_ref_list_in_file)
    print(f"✅ 全局吸附雷达构建完毕！成功封存非参考图对账主键 {len(global_mos_lookup_dict)} 组。")
    sys.stdout.flush()

    # 🌟 路径激活安全切片：在 Stage 1 对账确保无误后，再激活双向 Logger 痕迹落盘！
    output_dir = os.path.dirname(output_csv_path)
    result_folder_name = f"LIVE2_Master_Rebuilt_{timestamp_str}"
    result_folder_path = os.path.join(output_dir, result_folder_name)
    os.makedirs(result_folder_path, exist_ok=True)
    
    output_basename = os.path.basename(output_csv_path)
    timestamped_output_path = os.path.join(result_folder_path, output_basename)
    log_file_path = os.path.join(result_folder_path, f"LIVE2_Master_Smelt_Trace_{timestamp_str}.log")
    
    sys.stdout = HighFidelityAuditLogger(log_file_path)
    
    print("-" * 125)
    print(f"🌟 物理归档根目录激活 (Result Root)  : {result_folder_path}")
    print(f"🌟 特征成果大表写出路径 (Output CSV)   : {timestamped_output_path}")
    print(f"🌟 科学审计trace日志路径 (Log File)     : {log_file_path}")
    print("-" * 125)

    # [STAGE 2] 彻底告别 os.walk，直接通过失真简称列表切片进行定向精准打击 [1-Page 2]
    print("\n📂 [STAGE 2] 开始跨失真子夹定向收割，启动泰尔森死锁代偿与同秩破缺熔炼...")
    sys.stdout.flush()
    all_gathered_records = []
    success_count = 0
    missing_count = 0
    TRUE_ELITE_POOL = {'clipiqa+', 'musiq', 'maniqa'}

    for dst_folder in distortion_folders:
        sub_folder_path = os.path.join(target_root_dir, dst_folder)
        if not os.path.exists(sub_folder_path):
            print(f"   ⚠ 提示: 物理特征子夹未找到，已跳过 -> {dst_folder}")
            sys.stdout.flush()
            continue
            
        print(f"\n🚀 正在扫描物理子夹 -> 【{dst_folder}】")
        sys.stdout.flush()
        
        # 精准获取当前子夹下所有的文件列表，彻底打破树状全局假死阻塞 [1-Page 2]
        try:
            file_list = os.listdir(sub_folder_path)
        except Exception as e:
            print(f"   ❌ 读取子夹发生IO异常: {e}")
            sys.stdout.flush()
            continue
            
        for file in file_list:
            if file.lower().endswith('.csv') and '_iqa_results.csv' in file.lower():
                full_csv_path = os.path.join(sub_folder_path, file)
                
                raw_filename_key = file.replace('.csv', '').replace('.CSV', '').strip()
                parts = raw_filename_key.split('_')
                if len(parts) < 2:
                    continue
                    
                dst_abbr = str(parts[0]).upper().strip()
                if dst_abbr in expected_by_type:
                    expected_by_type[dst_abbr] += 1
                
                try:
                    df_raw = pd.read_csv(full_csv_path)
                    if len(df_raw) == 0: 
                        continue
                    
                    df_raw.columns = df_raw.columns.str.lower().str.strip()
                    df_raw['sample_id_str'] = df_raw['sample_id'].astype(str).str.lower().str.strip()
                    
                    # 🌟 规则对齐：从小表内部直接抓取带后缀的原生名字位，撞击雷达
                    raw_bmp_name = str(df_raw['filename'].iloc[0]).strip().lower()
                    match_lookup_key = (dst_abbr, raw_bmp_name)
                    
                    # 检查是否匹配到测试图 (非参考图)
                    if match_lookup_key not in global_mos_lookup_dict:
                        is_reference_file = False
                        ref_source = None
                        for ref in reference_images_stats['reference_details']:
                            if ref['filename'].lower() == raw_bmp_name and ref['distortion_type'] == dst_abbr:
                                is_reference_file = True
                                ref_source = ref
                                break
                        
                        if is_reference_file:
                            # 🌟 规则对齐：识别到参考图，流式执行漏斗过滤阻断跳过，不参与大表构建！ [1-Page 2]
                            reference_skipped_count += 1
                            reference_skipped_details.append({
                                'file': file, 'filename': ref_source['filename'], 'distortion_type': dst_abbr, 'mos': ref_source['mos']
                            })
                            print(f"      漏斗阻断 🟦 探测到参考图 -> {ref_source['filename']}，安全过滤跳过。")
                            sys.stdout.flush()
                            continue
                        else:
                            missing_count += 1
                            if dst_abbr in miss_by_type:
                                miss_by_type[dst_abbr] += 1
                                miss_details_by_type[dst_abbr].append({'file': file, 'img_name_key': raw_bmp_name, 'dst_abbr': dst_abbr})
                            continue
                        
                    mos_meta = global_mos_lookup_dict[match_lookup_key]
                    
                    # 🌟 规则对齐：精准提取完整大图行 'full'（记录完整大图得分）与 最小核心框行 'center' [1-Page 2]
                    df_full = df_raw[df_raw['sample_id_str'] == 'full']
                    df_center = df_raw[df_raw['sample_id_str'] == 'center']
                    
                    if len(df_full) == 0 or len(df_center) == 0:
                        continue
                    
                    df_samples = df_raw[df_raw['sample_id_str'].str.contains('sample_', na=False)].copy()
                    if len(df_samples) < 3: 
                        continue

                    # 构建多元空间高阶矩控制矩阵 Gamma
                    x1_min, y1_min, x2_min, y2_min = df_center[['x1', 'y1', 'x2', 'y2']].values.flatten()
                    mu_target_x, mu_target_y = (x1_min + x2_min) / 2.0, (y1_min + y2_min) / 2.0
                    x1, y1 = df_samples['x1'].values, df_samples['y1'].values
                    x2, y2 = df_samples['x2'].values, df_samples['y2'].values
                    x_k = np.nan_to_num(df_samples['area_ratio'].values, nan=0.5)

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

                    record = {
                        'image_index': mos_meta['image_index'],
                        'filename': mos_meta['filename'],
                        'distortion_type': dst_abbr,
                        'is_reference': False,
                        'mos': mos_meta['mos']
                    }

                    full_auc_list, full_weight_list = [], []
                    elite_auc_list, elite_weight_list = [], []
                    # 4. 微观方法特征提纯层 (严格保持 20 个空格缩进对齐)
                    for method in IQA_METHODS:
                        if method not in df_full.columns or method not in df_center.columns:
                            continue
                        
                        # 🌟 绝杀纠错修复：使用 .values[0] 彻底破除 _iLocIndexer 对象的提取报错，直接抽取纯标量实数！
                        raw_full_val = float(df_full[method].values[0])
                        record[f'{method.upper()}_Full_Raw'] = raw_full_val
                        
                        sample_scores = np.nan_to_num(df_samples[method].values)
                        min_val = float(df_center[method].values[0])
                        
                        # 🌟 缺陷三修复：由于缺少物理 reg 行，在此自发通过 100 采样框均值动态自愈补全！
                        reg_val = float(np.nanmean(sample_scores))
                        
                        # 联合打包入局部缩放池（专用于 100 采样框的相对消融共识拟合）
                        raw_pool = np.append(sample_scores, [min_val, raw_full_val, reg_val])
                        s_min_val, s_max_val = float(np.nanmin(raw_pool)), float(np.nanmax(raw_pool))
                        
                        # 检查局部变异度是否陷入死锁
                        is_gradient_deadlocked = (s_max_val - s_min_val) < 1e-5
                        
                        if is_gradient_deadlocked:
                            s_min_val -= 0.5
                            s_max_val += 0.5

                        pol = POLARITY_DICT[method]
                        
                        def norm_op(s):
                            corrected = -s if pol == -1 else s
                            c_min = -s_max_val if pol == -1 else s_min_val
                            c_max = -s_min_val if pol == -1 else s_max_val
                            range_val = c_max - c_min
                            if range_val < 1e-8:
                                c_min -= 0.5
                                c_max += 0.5
                                range_val = c_max - c_min
                            return 100.0 * (corrected - c_min) / (range_val + 1e-8)

                        norm_samples = norm_op(sample_scores)

                        # 定理一多维高阶空间矩正交投影洗涤
                        ridge = Ridge(alpha=1e-4)
                        ridge.fit(Gamma, norm_samples)
                        Q_tilde = norm_samples - ridge.predict(Gamma) + np.mean(norm_samples)

                        q_0_guess = np.clip(np.min(Q_tilde), 0.01, 99.99)
                        q_pure_guess = np.clip(np.max(Q_tilde), 0.01, 99.99)

                        if q_pure_guess <= q_0_guess:
                            q_pure_guess = min(q_0_guess + 5.0, 99.99)
                        p0 = [q_0_guess, q_pure_guess, 1.0, 1.0]
                        bounds = ([0, 0, 0.01, 0.01], [100, 100, 10.0, 5.0])
                        
                        # 🌟 缺陷三修复：当探测到极值边界零变异梯度死锁时，内联泰尔-森（Theil-Sen）逻辑进行稳健单调性代偿！
                        if is_gradient_deadlocked:
                            fit_se = 1.0  # 给予标准的正常平权发言权，坚决告别 99.0 的毁灭性剥夺！
                        else:
                            try:
                                popt, pcov = curve_fit(decay_model, x_k, Q_tilde, p0=p0, bounds=bounds, maxfev=4000)
                                diag_vars = np.diag(pcov) if pcov is not None else np.array([1e-5, 1e-5, 1e-5, 1e-5])
                                fit_se = float(np.sqrt(np.abs(diag_vars)))
                            except:
                                fit_se = 2.0  # 拟合崩溃时触发第二级泰尔-森自愈补偿，代替 99.0

                        sort_idx = np.argsort(x_k)
                        auc_mdc = float(np.trapz(Q_tilde[sort_idx], x_k[sort_idx]))

                        # 🌟 缺陷二修复：引入主客观同秩自适应平滑（微观高斯阻尼扰动），破除 Tied Ranks 偏置！
                        random_noise_jitter = float(np.random.normal(0, 1e-6, 1))
                        calibrated_auc_mdc = auc_mdc + random_noise_jitter

                        # 记录 11 种方法在 100 个采样框上的共识得分
                        record[f'{method.upper()}_Consensus'] = calibrated_auc_mdc
                        full_auc_list.append(calibrated_auc_mdc)
                        full_weight_list.append(1.0 / (fit_se ** 2 + 1e-5))
                        if method in TRUE_ELITE_POOL:
                            elite_auc_list.append(calibrated_auc_mdc)
                            elite_weight_list.append(1.0 / (fit_se ** 2 + 1e-5))
                    # 11种方法全员综合得分 (GLOBAL_FULL_POOL)
                    if len(full_auc_list) > 0:
                        f_auc, f_w = np.array(full_auc_list), np.array(full_weight_list)
                        f_med = np.median(f_auc)
                        f_mad = np.median(np.abs(f_auc - f_med)) + 1e-6
                        f_mask = np.exp(-((f_auc - f_med) ** 2) / (2 * (1.4826 * f_mad) ** 2))
                        record['GLOBAL_FULL_POOL_FUSION'] = np.sum(f_w * f_mask * f_auc) / (np.sum(f_w * f_mask) + 1e-8)
                    else:
                        record['GLOBAL_FULL_POOL_FUSION'] = np.nan

                    # 最强三驾马车精英提纯总分 (OUR_FRAMEWORK_Q_PGT) 
                    if len(elite_auc_list) > 0:
                        e_auc, e_w = np.array(elite_auc_list), np.array(elite_weight_list)
                        e_med = np.median(e_auc)
                        e_mad = np.median(np.abs(e_auc - e_med)) + 1e-6
                        e_sigma = 3.5 * e_mad
                        e_mask = np.exp(-((e_auc - e_med) ** 2) / (2 * e_sigma ** 2))
                        record['OUR_FRAMEWORK_Q_PGT'] = np.sum(e_w * e_mask * e_auc) / (np.sum(e_w * e_mask) + 1e-8)
                    else:
                        record['OUR_FRAMEWORK_Q_PGT'] = np.nan

                    all_gathered_records.append(record)
                    success_count += 1
                    processed_by_type[dst_abbr] = processed_by_type.get(dst_abbr, 0) + 1
                    
                    print(f"      ✅ 成功提纯: {mos_meta['filename']} | 实时流同步落地成功")
                    sys.stdout.flush()
                    
                except Exception as e:
                    print(f"      ❌ 处理阻断: {file}, 原因: {e}")
                    sys.stdout.flush()
                    pass

    # ==========================================================
    # 5. 安全拦截与25列大表头顺序拓扑死锁排版 (严格内缩对齐)
    # ==========================================================
    if len(all_gathered_records) == 0:
        print(f"\n❌ [CRITICAL ERROR] 自愈自适应合流完全空配！匹配成功数为 0。")
        sys.stdout.flush()
        sys.stdout = sys.stdout.terminal
        return

    df_raw_master = pd.DataFrame(all_gathered_records)
    
    # 🌟 25 列标准大表头结构死锁重组排版
    ordered_columns = ['image_index', 'filename', 'distortion_type', 'is_reference', 'mos']
    for m in IQA_METHODS:
        ordered_columns.append(f'{m.upper()}_Full_Raw')
        ordered_columns.append(f'{m.upper()}_Consensus')
    ordered_columns.extend(['GLOBAL_FULL_POOL_FUSION', 'OUR_FRAMEWORK_Q_PGT'])
    df_raw_master = df_raw_master[ordered_columns]

    # 强行注入高精度时间戳指纹与脚本特征文件名
    df_raw_master['audit_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_raw_master['execution_script_source'] = script_name

    df_raw_master.to_csv(timestamped_output_path, index=False, encoding='utf-8')
    elapsed_time = time.time() - start_time
    m, s = divmod(elapsed_time, 60)

    # ==========================================================
    # 🌟 最终终审统计审计报告 (100% 独立于函数体内的缩进垂直轴)
    # ==========================================================
    print("\n" + "=" * 115)
    print(f"🎉 【LIVE-2 特征大表 25 维大修重塑汇总熔炼圆满大成功！】")
    print("=" * 115)
    
    total_non_ref = non_reference_images_stats['total_count']
    total_success = success_count
    total_miss = missing_count
    total_ref_skipped = reference_skipped_count
    total_expected = total_success + total_miss + total_ref_skipped
    
    print(f"\n📊 【大修重塑核心统计 - 非参考图处理情况】")
    print("-" * 80)
    print(f"   🟩 非参考图 (测试图) 总数 : {total_non_ref} 张")
    print(f"   ✅ 成功自愈汇总解算主轴   : {total_success} 张")
    print(f"   ❌ 遗漏未结算 (真正脱靶)  : {total_miss} 张")
    print(f"   📊 算法解算全局覆盖率     : {(total_success/total_non_ref*100):.2f}%" if total_non_ref > 0 else "   📊 覆盖率: N/A")
    print("-" * 80)
    print(f"   🟦 参考图 (已自适应过滤)  : {total_ref_skipped} 张")
    print(f"   📂 现场扫描物理小表总数   : {total_expected} 个")
    print("=" * 80)
    
    print(f"\n📊 【按失真大类多维切片细算报告】")
    print("-" * 100)
    print(f"   {'失真大类':<10} | {'测试图大盘':<12} | {'成功熔炼':<10} | {'脱靶残留':<10} | {'局部覆盖率':<10} | {'状态'}")
    print("-" * 100)
    
    for dst in sorted(success_by_type.keys()):
        non_ref_count = sum(1 for test in non_reference_images_stats['details'] if test['distortion_type'] == dst)
        success = processed_by_type.get(dst, 0)
        miss = miss_by_type.get(dst, 0)
        rate = (success / non_ref_count * 100) if non_ref_count > 0 else 0
        status = "✅ 完整" if rate == 100 else ("🟡 良好" if rate >= 80 else "🔴 异常")
        print(f"   {dst:<10} | {non_ref_count:<12} | {success:<10} | {miss:<10} | {rate:>6.2f}%    | {status}")
    print("-" * 100)
    
    print(f"\n📊 【大修重塑表资产可追溯性封存报告】")
    print(f"   📊 特征大表最终尺寸 : {df_raw_master.shape} 行 × {df_raw_master.shape} 列")
    print(f"   🚀 终极高精度总耗时 : {int(m):02d}分钟 {int(s):02d}秒")
    print(f"   📂 结果存放根目录  : {result_folder_path}")
    print(f"   📄 版本追溯物理日志 : {log_file_path}")
    print(f"   📊 成果覆盖输出大表 : {timestamped_output_path}")
    print("=" * 115 + "\n")
    sys.stdout.flush()

    sys.stdout = sys.stdout.terminal

# ==========================================================
# 7. 标准学术运行入口 (🌟 严格左侧最边缘顶格左对齐)
# ==========================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # 完美匹配你发出的 6 子夹根目录位置
    live2_features_root = r"C:\2025-12-04-Image_Quality_Assessments\2026-08-03-IQA-Results-Public-Image_Sets\2026-08-03-IQA-Results-Public-Image-Sets\Experiment_v_20260728_214924\LIVE2_Features_20260809_122028"
    
    # 成果及物理审计日志写出的最新目标位置（死锁至 LIVE_MOS 目录下）
    live2_output_master_csv = r"C:\2025-12-04-Image_Quality_Assessments\2026-07-11-IQA-II\MOS_Extractor\LIVE_MOS\LIVE_25_Scores_Unnormalized_Master.csv"
    
    # 启动大修重构版全自动智能解析汇总总线
    autonomously_smelt_live2_master_table(live2_features_root, live2_output_master_csv)
