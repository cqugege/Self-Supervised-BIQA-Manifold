import os
import random
import torch
import cv2
import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import statsmodels.api as sm
from pyiqa import create_metric
from concurrent.futures import ProcessPoolExecutor

# 解决HuggingFace下载超时问题
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class Config:
    # 基础输入根目录
    BASE_SET_DIR = r"D:\2026_04_13_output_iqa\2026-07-28-Public_Image_Sets"

    # 外部指定的总输出根目录
    OUTPUT_BASE_ROOT = r"D:\2026_04_13_output_iqa\public_image_set_out_put"

    # 【动态生成时间戳子文件夹】每次运行代码都会生成一个独立不重名的版本文件夹
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_BASE_DIR = os.path.join(OUTPUT_BASE_ROOT, f"Experiment_v_{TIMESTAMP}")

    # 3个公开数据集的具体图像/标签配置
    DATASET_CONFIGS = {
        "CSIQ": {
            "img_dir": os.path.join(BASE_SET_DIR, "CSIQ", "dst_imgs"),
            "label_file": os.path.join(BASE_SET_DIR, "CSIQ", "csiq.csv"),
            "label_type": "csv"
        },
        "LIVE-2": {
            "img_dir": os.path.join(BASE_SET_DIR, "LIVE-2"),
            "label_file": os.path.join(BASE_SET_DIR, "LIVE-2", "dmos.mat"),
            "label_type": "mat_live2"
        },
        "LIVEC": {
            "img_dir": os.path.join(BASE_SET_DIR, "LIVEC", "Images"),
            "label_file": os.path.join(BASE_SET_DIR, "LIVEC", "Data", "AllMOS_release.mat"),
            "label_type": "mat_livec"
        }
    }

    # 11种质量指标
    METRIC_NAMES = ["brisque", "clipiqa+", "maniqa", "niqe", "nima",
                    "paq2piq", "cnniqa", "dbcnn", "musiq", "nrqm", "pi"]

    # 双卡配置 (1080ti * 2)
    GPUS = [0, 1] if torch.cuda.device_count() >= 2 else [0]


class DatasetEvaluator:
    def __init__(self, config, device_id, k_samples=100, min_ratio=1 / 9, label_dict=None):
        self.config = config
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.metric_names = config.METRIC_NAMES
        self.k_samples = k_samples
        self.min_ratio = min_ratio
        self.label_dict = label_dict if label_dict is not None else {}
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

        print(f"--- 正在初始化设备 {self.device} 的度量指标 ---")
        self.metrics = {}
        for m_name in config.METRIC_NAMES:
            try:
                self.metrics[m_name] = create_metric(m_name, device=self.device).to(self.device).eval()
            except Exception as e:
                print(f"[{self.device}] 指标 {m_name} 加载失败: {e}")

    def get_center_min_rect(self, img_h, img_w):
        side_ratio = np.sqrt(self.min_ratio)
        w_min = max(1, int(img_w * side_ratio))
        h_min = max(1, int(img_h * side_ratio))
        x_m = (img_w - w_min) // 2
        y_m = (img_h - h_min) // 2
        return x_m, y_m, w_min, h_min

    def process_single_image(self, img_path, image_tables_dir):
        p = Path(img_path)
        save_name = os.path.join(image_tables_dir, f"{p.stem}_IQA_Experiment.csv")

        if os.path.exists(save_name):
            return

        img = cv2.imread(str(p))
        if img is None:
            print(f"[{self.device}] 文件:{p.name} 图像读取失败")
            return

        img_h, img_w = img.shape[:2]
        img_area_full = img_h * img_w

        x_m, y_m, w_m, h_m = self.get_center_min_rect(img_h, img_w)
        x_max, y_max = x_m + w_m, y_m + h_m
        min_area = w_m * h_m

        true_score = self.label_dict.get(p.stem, np.nan)

        sample_positions = [
            {'id': 'max', 'x1': 0, 'y1': 0, 'x2': img_w, 'y2': img_h, 'area': img_area_full, 'area_ratio': 1.0},
            {'id': 'min', 'x1': x_m, 'y1': y_m, 'x2': x_max, 'y2': y_max, 'area': min_area,
             'area_ratio': min_area / img_area_full}
        ]
        sampled_keys = {f"0_0_{img_w}_{img_h}", f"{x_m}_{y_m}_{x_max}_{y_max}"}

        in_sample_number = 2
        max_attempts, attempts = 100, 0

        while in_sample_number < (self.k_samples + 2):
            x1, y1 = random.randint(0, x_m), random.randint(0, y_m)
            x2, y2 = random.randint(x_max, img_w), random.randint(y_max, img_h)
            width, height = x2 - x1, y2 - y1
            key = f"{x1}_{y1}_{x2}_{y2}"

            if key not in sampled_keys and width > 0 and height > 0:
                sampled_keys.add(key)
                sample_positions.append({
                    'id': str(in_sample_number - 1), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'area': width * height, 'area_ratio': (width * height) / img_area_full
                })
                in_sample_number += 1
                attempts = 0
            else:
                attempts += 1
            if attempts >= max_attempts:
                break

        columns = ['id', 'true_score', 'x1', 'y1', 'x2', 'y2', 'area', 'area_ratio'] + list(self.metrics.keys())
        results = []

        for pos in sample_positions:
            row = {'id': pos['id'], 'true_score': true_score, 'x1': pos['x1'], 'y1': pos['y1'],
                   'x2': pos['x2'], 'y2': pos['y2'], 'area': pos['area'], 'area_ratio': pos['area_ratio']}

            c_img = img[pos['y1']:pos['y2'], pos['x1']:pos['x2']]
            if c_img.size == 0: continue

            tensor = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

            with torch.no_grad():
                for name, func in self.metrics.items():
                    try:
                        row[name] = func(tensor).item()
                    except:
                        row[name] = np.nan
            results.append(row)

        df = pd.DataFrame(results, columns=columns)

        X_with_const = sm.add_constant(df['area_ratio'])
        pred_row = {'id': "reg", 'true_score': true_score, 'x1': None, 'y1': None, 'x2': None, 'y2': None, 'area': None,
                    'area_ratio': 1.0}

        for name in self.metrics.keys():
            if name in df.columns:
                try:
                    model = sm.OLS(df[name], X_with_const).fit()
                    pred_val = model.predict(np.array([[1.0, 1.0]]))
                    pred_row[name] = float(pred_val[0])
                except:
                    pred_row[name] = None

        pred_df = pd.DataFrame([pred_row], columns=df.columns)
        df = pd.concat([df, pred_df], ignore_index=True)
        df.to_csv(save_name, index=False)


def load_dataset_labels(dataset_name, cfg_dict):
    """从原始位置和格式中精准加载 mos / dmos 得分 (含LIVEC模糊兼容)"""
    label_dict = {}
    lbl_file = cfg_dict["label_file"]
    lbl_type = cfg_dict["label_type"]

    if not os.path.exists(lbl_file):
        print(f"警告: 标签文件不存在 {lbl_file}")
        return label_dict

    try:
        if lbl_type == "csv":
            df_csiq = pd.read_csv(lbl_file)
            for _, r in df_csiq.iterrows():
                t_type = str(r['dst_type']).upper()
                key_name = f"{r['image']}.{t_type}.{r['dst_idx']}"
                label_dict[key_name] = r['dmos']

        elif lbl_type == "mat_live2":
            mat_data = sio.loadmat(lbl_file)
            dmos = mat_data['dmos'].flatten()
            names = mat_data['names'].flatten()
            for i in range(len(names)):
                img_name = names[i] if isinstance(names[i], (np.ndarray, list)) else names[i]
                img_stem = Path(str(img_name).strip()).stem
                label_dict[img_stem] = dmos[i]

        elif lbl_type == "mat_livec":
            mat_data = sio.loadmat(lbl_file)
            all_mos_key = None
            all_img_key = None
            for k in mat_data.keys():
                if k.startswith('__'): continue
                if 'mos' in k.lower(): all_mos_key = k
                if 'image' in k.lower() or 'name' in k.lower(): all_img_key = k

            if not all_mos_key: all_mos_key = 'AllMOS'
            if not all_img_key: all_img_key = 'AllImages_names'

            all_mos = mat_data[all_mos_key].flatten()
            img_names = mat_data[all_img_key].flatten()

            for i in range(len(img_names)):
                img_name = img_names[i] if isinstance(img_names[i], (np.ndarray, list)) else img_names[i]
                img_stem = Path(str(img_name).strip()).stem
                label_dict[img_stem] = all_mos[i]
    except Exception as e:
        print(f"解析数据集 {dataset_name} 标签文件时出错: {e}")

    return label_dict


def scan_image_paths_v2(dataset_name, img_dir):
    """根据最新目录结构进行图像过滤筛选，修复元组lower报错"""
    image_paths = []
    formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    if not os.path.exists(img_dir):
        return image_paths

    if dataset_name in ["CSIQ", "LIVE-2"]:
        for root, dirs, files in os.walk(img_dir):
            if root == img_dir and dataset_name == "LIVE-2":
                continue
            for f in files:
                filename_str = str(f)
                ext = os.path.splitext(filename_str)[1].lower()
                if ext in formats:
                    image_paths.append(os.path.join(root, filename_str))

    elif dataset_name == "LIVEC":
        for f in os.listdir(img_dir):
            file_path = os.path.join(img_dir, f)
            if os.path.isfile(file_path):
                filename_str = str(f)
                ext = os.path.splitext(filename_str)[1].lower()
                if ext in formats:
                    image_paths.append(file_path)

    return image_paths


def worker_task(device_id, image_tasks, folder_name, k_samples, min_ratio, label_dict, runtime_output_dir):
    """单卡独立进程任务，写入当前带有时间戳的专属目录中"""
    config = Config()
    image_tables_dir = os.path.join(runtime_output_dir, f"{folder_name}_image_tables")
    os.makedirs(image_tables_dir, exist_ok=True)

    evaluator = DatasetEvaluator(config, device_id, k_samples=k_samples, min_ratio=min_ratio, label_dict=label_dict)
    for idx, img_path in enumerate(image_tasks):
        evaluator.process_single_image(img_path, image_tables_dir)
        if idx % 10 == 0:
            print(f"[{evaluator.device}] {folder_name} 进度: {idx}/{len(image_tasks)} - {Path(img_path).name}")


def extract_distortion_type_v2(dataset_name, img_path):
    p_path = Path(img_path)
    if dataset_name in ["CSIQ", "LIVE-2"]:
        return p_path.parent.name.lower()
    elif dataset_name == "LIVEC":
        return "authentic_real_world"
    return "unknown"


def generate_total_summary(folder_name, config, all_img_paths, runtime_output_dir):
    """按子文件夹特征聚合汇总全集大表，显式提取float标量移除中括号"""
    image_tables_dir = os.path.join(runtime_output_dir, f"{folder_name}_image_tables")

    score_type = "DMOS" if folder_name in ["CSIQ", "LIVE-2"] else "MOS"
    total_summary_path = os.path.join(runtime_output_dir, f"{folder_name}_{score_type}_Total_Summary.csv")
    distortion_summary_dir = os.path.join(runtime_output_dir, "distortion_type_summaries")
    os.makedirs(distortion_summary_dir, exist_ok=True)

    if not os.path.exists(image_tables_dir): return

    path_map = {Path(p).stem: p for p in all_img_paths}
    summary_rows = []
    print(f"--> 正在为 {folder_name} 聚合汇总表并提取污染子类型表(终极清除中括号)...")

    for file in os.listdir(image_tables_dir):
        if file.endswith("_IQA_Experiment.csv"):
            try:
                sub_df = pd.read_csv(os.path.join(image_tables_dir, file))
                reg_row = sub_df[sub_df['id'] == 'reg']
                if not reg_row.empty:
                    img_stem = file.replace("_IQA_Experiment.csv", "")
                    orig_path = path_map.get(img_stem, "")
                    dist_type = extract_distortion_type_v2(folder_name, orig_path) if orig_path else "unknown"

                    # 强制转换为浮点标量，彻底告别中括号
                    t_val = reg_row['true_score'].values[0]
                    row_data = {
                        'image_name': img_stem,
                        'true_score': float(t_val) if pd.notna(t_val) else np.nan,
                        'distortion_type': dist_type
                    }
                    for metric in config.METRIC_NAMES:
                        if metric in reg_row.columns:
                            val = reg_row[metric].values[0]
                            row_data[metric] = float(val) if pd.notna(val) else np.nan
                    summary_rows.append(row_data)
            except Exception as e:
                print(f"读取文件 {file} 失败: {e}")

    if summary_rows:
        total_df = pd.DataFrame(summary_rows)
        ordered_cols = ['image_name', 'true_score', 'distortion_type'] + [m for m in config.METRIC_NAMES if
                                                                          m in total_df.columns]
        total_df = total_df[ordered_cols]
        total_df.to_csv(total_summary_path, index=False)
        print(f"成功！已保存时间戳版本集合总表: {total_summary_path}")

        grouped = total_df.groupby('distortion_type')
        for dist_name, group_df in grouped:
            dist_csv_name = os.path.join(distortion_summary_dir,
                                         f"{folder_name}_{score_type}_{dist_name.upper()}_Summary.csv")
            group_df.to_csv(dist_csv_name, index=False)
            print(f"   [已分流独立总表] -> {folder_name}_{score_type}_{dist_name.upper()}_Summary.csv")


def run_experiment(k_samples=100, min_ratio=1 / 9, is_test_mode=False):
    config = Config()
    gpus = config.GPUS
    num_gpus = len(gpus)

    # 【核心改动：版本时间戳隔离】每次启动程序，在指定的输出目录下自动创建一个全新的时间戳文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_output_dir = os.path.join(config.OUTPUT_BASE_DIR, f"Experiment_v_{timestamp}")
    os.makedirs(runtime_output_dir, exist_ok=True)

    print(f"双卡并行动态驱动... 可用GPU: {gpus}")
    print(f"本次运行的所有结果将统一存储至时间戳目录: {runtime_output_dir}\n")

    for folder_name, cfg_dict in config.DATASET_CONFIGS.items():
        print(f"\n================ 载入 {folder_name} 真值标签 ================")
        label_dict = load_dataset_labels(folder_name, cfg_dict)
        print(f"共计载入 {len(label_dict)} 条真值得分。")

        print(f"\n================ 定制化扫描 {folder_name} 图像存储层级 ================")
        all_images = scan_image_paths_v2(folder_name, cfg_dict["img_dir"])
        if not all_images:
            print(f"警告: 路径 {cfg_dict['img_dir']} 下未扫描到图片，跳过。")
            continue

        # 判断是否为测试模式
        if is_test_mode:
            all_images = all_images[:2]
            print(f"⚠️ [测试短路激活] 强行限抽2张图进行看板测试: {[Path(p).name for p in all_images]}")
        else:
            print(f"成功筛选出有效图片共 {len(all_images)} 张，正在分配多卡子进程...")

        chunks = [all_images[i::num_gpus] for i in range(num_gpus)]

        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, gpu_id in enumerate(gpus):
                if len(chunks[i]) > 0:
                    futures.append(
                        executor.submit(worker_task, gpu_id, chunks[i], folder_name, k_samples, min_ratio, label_dict,
                                        runtime_output_dir))
            for future in futures:
                future.result()

        generate_total_summary(folder_name, config, all_images, runtime_output_dir)

    print(f"\n🎉 计算完毕！该版本所有表格、分流对比表均已安全装入: {runtime_output_dir}")


if __name__ == "__main__":
    # =========================================================================
    # 控制台参数调度面板：
    # 1. k_samples: 随机向外扩展的采样框数量 k
    # 2. min_ratio: 最小基础框占整图面积比例 (1/9 对应原中心1/3区域宽高)
    # 3. is_test_mode: 如果设为 True，则每个文件夹只跑2张图且只算全图一行，方便测通。
    #                  如果设为 False，则关闭测试，全量开跑 100 个采样框。
    # =========================================================================
    #  run_experiment(k_samples=2, min_ratio=1 / 9, is_test_mode= False)
    run_experiment(k_samples= 50, min_ratio=1 / 9, is_test_mode= False)


