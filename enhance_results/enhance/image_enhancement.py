import cv2
import numpy as np
import os
import datetime
import csv
import json
import tempfile
import random
import traceback
from skimage.metrics import structural_similarity as ssim

base_dir = os.path.dirname(__file__)

# 读取配置文件（如果存在），否则使用内置默认值
def _load_config():
    cfg_path = os.path.join(base_dir, 'config.json')
    default = {
        'results': {
            'csv_dir': 'results/csv',
            'images_dir': 'results/images',
            'csv_filename': 'processed_comparison.csv'
        },
        'save_image': {
            'scale_factor': 0.6,
            'label_original': 'ORIGINAL',
            'label_processed': 'PROCESSED'
        },
        'evaluation': {
            'local_contrast_kernel': 15,
            'multiscale_scales': [5, 15, 25]
        }
    }
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)
            # 简单合并（仅一层）
            for k, v in user_cfg.items():
                if k in default and isinstance(v, dict):
                    default[k].update(v)
                else:
                    default[k] = v
    except Exception:
        # 无配置或解析失败时使用默认
        pass
    return default

CONFIG = _load_config()

def _resolve_path(p):
    if p is None:
        return None
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(base_dir, p))

# 计算信息熵
def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # 归一化
    hist = hist[hist > 0]  # 过滤掉0概率
    return -np.sum(hist * np.log2(hist))

# 计算局部对比度图
def local_contrast_map(image, kernel_size=15):
    """计算局部对比度图"""
    # 将PIL Image对象转换为NumPy数组
    img_array = None
    if hasattr(image, 'mode'):
        img_array = np.array(image)
    else:
        img_array = image
    
    # 确保img_array已正确定义
    if img_array is None:
        raise ValueError("无法将图像转换为NumPy数组")
    
    # 确保处理的是NumPy数组
    if len(img_array.shape) == 3:
        # 对于PIL读取的图像（RGB顺序），需要转换为BGR才能正确使用cv2.cvtColor
        if img_array.shape[2] == 3:
            # 检查是否需要颜色空间转换（PIL是RGB，OpenCV是BGR）
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif img_array.shape[2] == 4:
            # 对于RGBA图像，转换为灰度
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        else:
            gray = img_array[:,:,0]  # 取第一个通道作为灰度
    else:
        gray = img_array
    
    # 使用局部标准差作为对比度度量
    # 局部区域E[X^2]
    local_contrast = cv2.blur(gray**2, (kernel_size, kernel_size))
    # 局部区域E[X]
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    # 计算局部区域标准差Var[X] = E[X^2] - (E[X])^2
    local_std = np.sqrt(local_contrast - local_mean**2)
    
    return local_std

# 多尺度的局部对比度
def multiscale_contrast_analysis(image, scales=[5, 15, 25]):
    """
    多尺度对比度分析
    适用于检测不同大小的漆面缺陷
    """
    # 将PIL Image对象转换为NumPy数组
    img_array = None
    if hasattr(image, 'mode'):
        img_array = np.array(image)
    else:
        img_array = image
    
    # 确保img_array已正确定义
    if img_array is None:
        raise ValueError("无法将图像转换为NumPy数组")
    
    # 确保处理的是NumPy数组
    if len(img_array.shape) == 3:
        # 对于PIL读取的图像（RGB顺序），需要转换为BGR才能正确使用cv2.cvtColor
        if img_array.shape[2] == 3:
            # 检查是否需要颜色空间转换（PIL是RGB，OpenCV是BGR）
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif img_array.shape[2] == 4:
            # 对于RGBA图像，转换为灰度
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        else:
            gray = img_array[:,:,0]  # 取第一个通道作为灰度
    else:
        gray = img_array
    
    contrast_results = {}
    
    for scale in scales:
        local_std = local_contrast_map(gray, scale)
        contrast_results[f'scale_{scale}'] = {
            'mean_contrast': np.mean(local_std),
            'max_contrast': np.max(local_std),
            'contrast_map': local_std
        }
    
    return contrast_results

# 定量评估
def evaluate_image(original, enhanced, filename, output_csv_dir=None):
    """
    计算并输出图像的质量评估指标

    参数:
        original: 原始图像(PIL Image)
        enhanced: 增强后的图像(PIL Image)
        filename: 图像文件名，用于输出到文件
    """

    # 将PIL Image对象转换为NumPy数组
    if hasattr(original, 'mode'):
        original_array = np.array(original)
        # PIL读取的图像是RGB/RGBA顺序，需要转换为BGR/BGRA才能与OpenCV兼容
        if len(original_array.shape) == 3:
            if original_array.shape[2] == 3:
                original_array = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
            elif original_array.shape[2] == 4:
                original_array = cv2.cvtColor(original_array, cv2.COLOR_RGBA2BGRA)
    else:
        original_array = original
    
    # 对enhanced做同样的处理
    if hasattr(enhanced, 'mode'):
        enhanced_array = np.array(enhanced)
        # PIL读取的图像是RGB/RGBA顺序，需要转换为BGR/BGRA才能与OpenCV兼容
        if len(enhanced_array.shape) == 3:
            if enhanced_array.shape[2] == 3:
                enhanced_array = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR)
            elif enhanced_array.shape[2] == 4:
                enhanced_array = cv2.cvtColor(enhanced_array, cv2.COLOR_RGBA2BGRA)
    else:
        enhanced_array = enhanced

    # 确保图像是灰度图用于质量评估
    if len(original_array.shape) > 2:
        original_gray = cv2.cvtColor(original_array, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original_array.copy()
    
    if len(enhanced_array.shape) > 2:
        enhanced_gray = cv2.cvtColor(enhanced_array, cv2.COLOR_BGR2GRAY)
    else:
        enhanced_gray = enhanced_array.copy()
    
    # 初始化结果字典
    results = {
        'filename': filename,
        'original_sharpness': 0,
        'enhanced_sharpness': 0,
        'original_contrast': 0,
        'enhanced_contrast': 0,
        'original_entropy': 0,
        'enhanced_entropy': 0,
        'psnr': 0,  # 峰值信噪比
        'ssim': 0   # 结构相似性指数
    }
    
    try:
        # 1. 计算清晰度 (拉普拉斯方差)
        # 值越高表示图像越清晰
        results['original_sharpness'] = cv2.Laplacian(original_gray, cv2.CV_64F).var()
        results['enhanced_sharpness'] = cv2.Laplacian(enhanced_gray, cv2.CV_64F).var()
    except Exception as e:
        print(f"计算清晰度时出错: {e}")
    
    try:
        # results['original_contrast'] = np.std(original_gray)
        # results['enhanced_contrast'] = np.std(enhanced_gray)
        '''
        # 2. 计算对比度 (Michelson对比度)
        # 值范围为[0,1]，值越高表示对比度越强
        # Michelson对比度公式: (I_max - I_min) / (I_max + I_min)
        # 处理原始图像
        original_max = np.max(original_gray)
        original_min = np.min(original_gray)
        # 避免除以零
        denominator = original_max + original_min
        results['original_contrast'] = (original_max - original_min) / denominator if denominator != 0 else 0
        
        # 处理增强后的图像
        enhanced_max = np.max(enhanced_gray)
        enhanced_min = np.min(enhanced_gray)
        # 避免除以零
        denominator = enhanced_max + enhanced_min
        results['enhanced_contrast'] = (enhanced_max - enhanced_min) / denominator if denominator != 0 else 0
        '''
        results['original_contrast'] = np.mean(local_contrast_map(original_gray))
        results['enhanced_contrast'] = np.mean(local_contrast_map(enhanced_gray))
    except Exception as e:
        print(f"计算对比度时出错: {e}")
    
    try:
        # 3. 计算熵 (信息熵)
        # 值越高表示图像信息量越大
        results['original_entropy'] = calculate_entropy(original_gray)
        results['enhanced_entropy'] = calculate_entropy(enhanced_gray)
    except Exception as e:
        print(f"计算熵时出错: {e}")
    
    try:
        # 4. 计算峰值信噪比(PSNR)
        # 值越高表示图像质量越好，通常在20-50dB之间
        # 确保两个图像的尺寸相同
        if original_gray.shape == enhanced_gray.shape:
            # 对于8位图像，最大值为255
            max_pixel = 255.0
            results['psnr'] = cv2.PSNR(original_gray, enhanced_gray, max_pixel)
        else:
            print("计算PSNR时出错: 原始图像和增强图像尺寸不匹配")
    except Exception as e:
        print(f"计算PSNR时出错: {e}")
    
    try:
        # 5. 计算结构相似性指数(SSIM)
        # 值范围为[-1,1]，值越接近1表示结构相似性越好
        # 确保两个图像的尺寸相同
        if original_gray.shape == enhanced_gray.shape:
            # 使用OpenCV的SSIM计算，winSize设置为11，sigma设置为1.5
            # 确保只提取返回值的第一个元素
            ssim_result, _ = cv2.quality.QualitySSIM_compute(original_gray, enhanced_gray, (11, 11))
            results['ssim'] = ssim_result[0]
        else:
            print("计算SSIM时出错: 原始图像和增强图像尺寸不匹配")
    except Exception as e:
        # 如果OpenCV版本较旧，尝试使用另一种方式计算SSIM
        try:
            if original_gray.shape == enhanced_gray.shape:
                ssim_result = ssim(original_gray, enhanced_gray)
                results['ssim'] = ssim_result
            else:
                print("计算SSIM时出错: 原始图像和增强图像尺寸不匹配")
        except Exception as inner_e:
            print(f"计算SSIM时出错: {inner_e}")
    
    # 将结果写入 CSV 文件，目录和文件名由 CONFIG 决定
    csv_filename = CONFIG.get('results', {}).get('csv_filename', 'processed_comparison.csv')
    if output_csv_dir:
        results_csv_dir = output_csv_dir
    else:
        csv_dir_cfg = CONFIG.get('results', {}).get('csv_dir', 'results/csv')
        results_csv_dir = _resolve_path(csv_dir_cfg)
    os.makedirs(results_csv_dir, exist_ok=True)

    output_file = os.path.join(results_csv_dir, csv_filename)
    file_exists = os.path.isfile(output_file)

    # 使用 csv.writer 写入，确保兼容 Excel/其他工具
    try:
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                # 写入表头（CSV）
                writer.writerow([
                    '文件名',
                    '原始清晰度',
                    '增强后清晰度',
                    '原始局部15对比度',
                    '增强后局部15对比度',
                    '原始熵',
                    '增强后熵',
                    '峰值信噪比(PSNR)',
                    '结构相似性(SSIM)'
                ])

            # 写入数据行，数值保留4位小数
            writer.writerow([
                results['filename'],
                f"{results['original_sharpness']:.4f}",
                f"{results['enhanced_sharpness']:.4f}",
                f"{results['original_contrast']:.4f}",
                f"{results['enhanced_contrast']:.4f}",
                f"{results['original_entropy']:.4f}",
                f"{results['enhanced_entropy']:.4f}",
                f"{results['psnr']:.4f}",
                f"{results['ssim']:.4f}"
            ])

        print(f"质量评估结果已保存到 {output_file}")
    except Exception as e:
        print(f"写入 CSV 文件时出错: {e}")
    return results

# 保存处理前后的图像
def save_image(src, result, image_path,  output_path=None):
    # 创建并排显示的图像
    # 设置缩放比例，缩小图像尺寸（从 CONFIG 读取）
    scale_factor = CONFIG.get('save_image', {}).get('scale_factor', 0.6)
    
    # 将PIL Image对象转换为NumPy数组
    if hasattr(src, 'mode'):
        src_array = np.array(src)
        # PIL读取的图像是RGB/RGBA顺序，需要转换为BGR/BGRA才能与OpenCV兼容
        if len(src_array.shape) == 3:
            if src_array.shape[2] == 3:
                src_array = cv2.cvtColor(src_array, cv2.COLOR_RGB2BGR)
            elif src_array.shape[2] == 4:
                src_array = cv2.cvtColor(src_array, cv2.COLOR_RGBA2BGRA)
    else:
        src_array = src
    
    # 对result做同样的处理
    if hasattr(result, 'mode'):
        result_array = np.array(result)
        # PIL读取的图像是RGB/RGBA顺序，需要转换为BGR/BGRA才能与OpenCV兼容
        if len(result_array.shape) == 3:
            if result_array.shape[2] == 3:
                result_array = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
            elif result_array.shape[2] == 4:
                result_array = cv2.cvtColor(result_array, cv2.COLOR_RGBA2BGRA)
    else:
        result_array = result
    
    # 获取原始图像尺寸并缩小
    h1, w1 = src_array.shape[:2]
    new_w1 = int(w1 * scale_factor)
    new_h1 = int(h1 * scale_factor)
    
    # 获取处理后图像尺寸并缩小（保持相同的缩放比例）
    h2, w2 = result_array.shape[:2]
    new_w2 = int(w2 * scale_factor)
    new_h2 = int(h2 * scale_factor)
    
    # 缩放到统一的尺寸（取较小的宽度和高度，确保两张图尺寸一致）
    unified_w = min(new_w1, new_w2)
    unified_h = min(new_h1, new_h2)
    
    # 缩放图像
    src_resized = cv2.resize(src_array, (unified_w, unified_h))
    result_resized = cv2.resize(result_array, (unified_w, unified_h))
    
    # 检测图像通道数，处理灰度图和彩色图
    src_channels = 1 if len(src_resized.shape) == 2 else src_resized.shape[2]
    result_channels = 1 if len(result_resized.shape) == 2 else result_resized.shape[2]
    
    # 确定最终图像的通道数：如果有一个是4通道，最终就是4通道；否则是3通道
    final_channels = 4 if (src_channels == 4 or result_channels == 4) else 3
    
    # 处理灰度图转换为彩色图
    # 只对灰度图（单通道）应用颜色空间转换
    # 对于src_resized
    if len(src_resized.shape) == 2:  # 灰度图
        if final_channels == 3:
            src_resized = cv2.cvtColor(src_resized, cv2.COLOR_GRAY2BGR)
        else:  # final_channels == 4
            src_resized = cv2.cvtColor(src_resized, cv2.COLOR_GRAY2BGRA)
    elif src_channels != final_channels:  # 确保通道数一致
        if final_channels == 3 and src_channels == 4:
            src_resized = cv2.cvtColor(src_resized, cv2.COLOR_BGRA2BGR)
        elif final_channels == 4 and src_channels == 3:
            src_resized = cv2.cvtColor(src_resized, cv2.COLOR_BGR2BGRA)
    
    # 对于result_resized
    if len(result_resized.shape) == 2:  # 灰度图
        if final_channels == 3:
            result_resized = cv2.cvtColor(result_resized, cv2.COLOR_GRAY2BGR)
        else:  # final_channels == 4
            result_resized = cv2.cvtColor(result_resized, cv2.COLOR_GRAY2BGRA)
    elif result_channels != final_channels:  # 确保通道数一致
        if final_channels == 3 and result_channels == 4:
            result_resized = cv2.cvtColor(result_resized, cv2.COLOR_BGRA2BGR)
        elif final_channels == 4 and result_channels == 3:
            result_resized = cv2.cvtColor(result_resized, cv2.COLOR_BGR2BGRA)
   
    # 创建新图像，宽度为两个图像宽度之和，高度为统一高度
    # 对于四通道图像，使用np.ones而不是np.zeros，以便更好地处理透明度
    if final_channels == 4:
        # 对于RGBA图像，创建带透明度的图像，初始设置为完全不透明
        vis = np.ones((unified_h, unified_w * 2, final_channels), np.uint8) * 255
        # Alpha通道初始化为不透明
        vis[:, :, 3] = 255
    else:
        # 对于RGB图像，使用黑色背景
        vis = np.zeros((unified_h, unified_w * 2, final_channels), np.uint8)
    
    # 将缩放后的图像复制到新图像中，包括所有通道
    # 对于四通道图像，这样会正确复制RGB和Alpha通道
    vis[:, :unified_w] = src_resized
    vis[:, unified_w:unified_w*2] = result_resized
   
    # 为了解决中文乱码问题，我们可以使用图像拼接的方式添加标签
    # 创建标签区域
    label_height = 40
    # 根据通道数创建带标签的图像
    labeled_image = np.ones((unified_h + label_height, unified_w * 2, final_channels), np.uint8) * 255  # 白色背景
    
    # 如果是四通道图像，确保Alpha通道为不透明，以便标签清晰可见
    if final_channels == 4:
        labeled_image[:, :, 3] = 255
    
    # 将图像内容复制到带标签的图像中，保留所有通道信息
    labeled_image[label_height:, :] = vis
    
    # 添加标签（文本从 CONFIG 读取，默认为英文以避免字体问题）
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_orig = CONFIG.get('save_image', {}).get('label_original', 'ORIGINAL')
    label_proc = CONFIG.get('save_image', {}).get('label_processed', 'PROCESSED')
    cv2.putText(labeled_image, label_orig, (unified_w//4 - 30, 25), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(labeled_image, label_proc, (unified_w + unified_w//4 - 30, 25), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    
    # 保存合并后的图像到由 CONFIG 指定的目录（支持相对或绝对路径）
    images_dir_cfg = CONFIG.get('results', {}).get('images_dir', 'results/images')
    default_images_dir = _resolve_path(images_dir_cfg)
    results_dir = output_path if output_path else default_images_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取原始文件名（不含路径）
    original_filename = os.path.basename(image_path)
    # 获取文件名（不含扩展名）和扩展名
    name_without_ext, ext = os.path.splitext(original_filename)
    
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 构造保存的文件名
    save_filename = f"{name_without_ext}_{timestamp}{ext}"
    save_path = os.path.join(results_dir, save_filename)
    
    # 保存图像
    cv2.imwrite(save_path, labeled_image)
    print(f"对比图像已保存至: {save_path}")


def save_processed_image(processed, image_path, output_path=None):
    """
    保存仅处理后的图像（不合并原图）。

    参数:
        processed: 处理后的图像，可以是 PIL Image 或 NumPy 数组（uint8）
        image_path: 原始图像路径（用于生成保存文件名）
        output_path: 可选的输出目录，若为 None 则使用 CONFIG 中的 images_dir/processed
    返回:
        保存的文件完整路径
    """
    # 将PIL Image对象转换为NumPy数组
    if hasattr(processed, 'mode'):
        proc_array = np.array(processed)
        if len(proc_array.shape) == 3:
            # 若为 RGB/RGBA, 转为 BGR/BGRA
            if proc_array.shape[2] == 3:
                proc_array = cv2.cvtColor(proc_array, cv2.COLOR_RGB2BGR)
            elif proc_array.shape[2] == 4:
                proc_array = cv2.cvtColor(proc_array, cv2.COLOR_RGBA2BGRA)
    else:
        proc_array = processed

    # 确保是 uint8
    if proc_array.dtype != np.uint8:
        proc_array = np.clip(proc_array, 0, 255).astype(np.uint8)

    # 输出目录：CONFIG images_dir + 'processed' 子目录（除非由 output_path 覆盖）
    images_dir_cfg = CONFIG.get('results', {}).get('images_dir', 'results/images')
    default_images_dir = _resolve_path(images_dir_cfg)
    # 默认的 processed 子目录（若未传入 output_path）
    processed_dir = os.path.join(default_images_dir, 'processed')
    results_dir = output_path if output_path else processed_dir
    os.makedirs(results_dir, exist_ok=True)

    # 构造保存文件名，保持扩展名一致
    original_filename = os.path.basename(image_path)
    name_without_ext, ext = os.path.splitext(original_filename)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_filename = f"{name_without_ext}_proc_{timestamp}{ext}"
    save_path = os.path.join(results_dir, save_filename)

    cv2.imwrite(save_path, proc_array)
    print(f"处理后图像已保存至: {save_path}")
    return save_path


# ------------------ 自适应检测与增强 ------------------
def _to_numpy(img):
    """将 PIL 或 NumPy 图像统一为 NumPy ndarray（BGR 或 单通道灰度）"""
    if hasattr(img, 'mode'):
        arr = np.array(img)
        # PIL -> RGB/RGBA, 转为 BGR/BGRA
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        return arr
    else:
        return img


def detect_distortion(image):
    """检测图像的常见失真类型与程度。

    返回字典，包含 metrics 和 types 列表，例:
    {
        'sharpness': ..., 'contrast': ..., 'uneven': ..., 'specular_ratio': ..., 'types': [...] }
    """
    arr = _to_numpy(image)
    # 转成灰度用于度量
    try:
        if arr is None:
            return {'types': [], 'sharpness': 0, 'contrast': 0, 'uneven': 0, 'specular_ratio': 0}
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = arr.copy()
    except Exception:
        gray = np.array(image, dtype=np.uint8)

    # 清晰度：拉普拉斯方差
    try:
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        sharpness = 0.0

    # 对比度：局部对比度均值
    try:
        contrast = float(np.mean(local_contrast_map(gray, CONFIG.get('evaluation', {}).get('local_contrast_kernel', 15))))
    except Exception:
        contrast = 0.0

    # 不均匀照明：将图像分块，计算块均值的标准差归一化
    try:
        h, w = gray.shape[:2]
        gs = 32
        hh = max(1, h // gs)
        ww = max(1, w // gs)
        means = []
        for y in range(0, h, hh):
            for x in range(0, w, ww):
                patch = gray[y:y+hh, x:x+ww]
                if patch.size:
                    means.append(float(np.mean(patch)))
        if len(means) > 1:
            uneven = float(np.std(means) / (np.mean(means) + 1e-9))
        else:
            uneven = 0.0
    except Exception:
        uneven = 0.0

    # 反光比率：基于 HSV 的 V 通道极亮像素占比
    try:
        spec_cfg = CONFIG.get('enhance', {}).get('specular', {})
        max_thr = int(spec_cfg.get('max_thr', 250))
        factor = float(spec_cfg.get('std_factor', 2.0))
        if arr.ndim == 3:
            try:
                hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
            except Exception:
                hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            v = hsv[:, :, 2]
        else:
            v = gray
        thr = min(max_thr, int(np.mean(v) + factor * np.std(v)))
        mask_spec = (v >= thr)
        specular_ratio = float(np.sum(mask_spec) / (v.size + 1e-9))
    except Exception:
        specular_ratio = 0.0

    types = []
    det_cfg = CONFIG.get('detection', {})
    if specular_ratio > float(det_cfg.get('specular_ratio', CONFIG.get('enhance', {}).get('specular', {}).get('specular_ratio_thresh', 0.01))):
        types.append('specular')
    if contrast < float(det_cfg.get('contrast_thresh', 0.05)):
        types.append('low_contrast')
    if sharpness < float(det_cfg.get('sharpness_thresh', 50)):
        types.append('blur')
    if uneven > float(det_cfg.get('uneven_thresh', 0.08)):
        types.append('uneven_illumination')

    return {
        'types': types,
        'sharpness': sharpness,
        'contrast': contrast,
        'uneven': uneven,
        'specular_ratio': specular_ratio
    }


def enhance_by_type(image, detection=None, params=None):
    """根据检测结果对图像应用不同增强策略，返回增强后的图像（NumPy uint8 BGR）。"""
    arr = _to_numpy(image)
    if arr is None:
        return image

    work = arr.copy()
    if params is None:
        params = CONFIG.get('enhance', {})

    # 1) 处理反光区域（inpaint）
    if detection and 'specular' in detection.get('types', []):
        try:
            if work.ndim == 3:
                hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
                v = hsv[:, :, 2]
            else:
                v = work
            spec_cfg = params.get('specular', {})
            max_thr = int(spec_cfg.get('max_thr', 250))
            factor = float(spec_cfg.get('std_factor', 2.0))
            thr = min(max_thr, int(np.mean(v) + factor * np.std(v)))
            mask = (v >= thr).astype('uint8') * 255
            inpaint_cfg = params.get('inpaint', {})
            dk = int(inpaint_cfg.get('dilation_kernel', 7))
            dit = int(inpaint_cfg.get('dilation_iterations', 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
            mask = cv2.dilate(mask, kernel, iterations=dit)
            if work.ndim == 3:
                r = int(inpaint_cfg.get('inpaint_radius', 3))
                work = cv2.inpaint(work, mask, r, cv2.INPAINT_TELEA)
            else:
                tmp = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
                r = int(inpaint_cfg.get('inpaint_radius', 3))
                tmp = cv2.inpaint(tmp, mask, r, cv2.INPAINT_TELEA)
                work = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass

    # 2) 对比度不足 / 照明不均衡：CLAHE 或 同态滤波
    if detection and ('low_contrast' in detection.get('types', []) or 'uneven_illumination' in detection.get('types', [])):
        try:
            clahe_cfg = params.get('clahe', {})
            clipLimit = float(clahe_cfg.get('clipLimit', 2.0))
            tileGridSize = tuple(clahe_cfg.get('tileGridSize', [8, 8]))
            if work.ndim == 3:
                lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
                l2 = clahe.apply(l)
                lab2 = cv2.merge((l2, a, b))
                work = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
                work = clahe.apply(work)
        except Exception:
            # 回退到同态滤波（灰度）
            try:
                work = homomorphic_filter(work, params.get('gammaH', 2.0), params.get('gammaL', 0.5), params.get('c', 1.0), params.get('D0', 30))
            except Exception:
                pass

    # 3) 模糊：双边滤波 + 反锐化
    if detection and 'blur' in detection.get('types', []):
        try:
            unsharp_cfg = params.get('unsharp', {})
            d = int(unsharp_cfg.get('denoise_d', 9))
            sigmaColor = float(unsharp_cfg.get('denoise_sigmaColor', 75))
            sigmaSpace = float(unsharp_cfg.get('denoise_sigmaSpace', 75))
            w_main = float(unsharp_cfg.get('weight_main', 1.5))
            w_blur = float(unsharp_cfg.get('weight_blur', -0.5))
            if work.ndim == 3:
                den = cv2.bilateralFilter(work, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
                blurred = cv2.GaussianBlur(den, (0, 0), sigmaX=3)
                work = cv2.addWeighted(den, w_main, blurred, w_blur, 0)
            else:
                den_bgr = cv2.bilateralFilter(cv2.cvtColor(work, cv2.COLOR_GRAY2BGR), d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
                den = cv2.cvtColor(den_bgr, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(den, (0, 0), sigmaX=3)
                work = cv2.addWeighted(den, w_main, blurred, w_blur, 0)
        except Exception:
            pass

    # 4) 细节增强（Laplacian叠加）
    try:
        detail_w = float(params.get('detail_weight', 0.3))
        if work.ndim == 3:
            gray_tmp = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray_tmp, cv2.CV_16S, ksize=3)
            lap = cv2.convertScaleAbs(lap)
            detail = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
            work = cv2.addWeighted(work, 1.0, detail, detail_w, 0)
        else:
            lap = cv2.Laplacian(work, cv2.CV_16S, ksize=3)
            lap = cv2.convertScaleAbs(lap)
            work = cv2.addWeighted(work, 1.0, lap, detail_w, 0)
    except Exception:
        pass

    # 保证 uint8
    try:
        if work.dtype != np.uint8:
            work = np.clip(work, 0, 255).astype(np.uint8)
    except Exception:
        pass

    return work


def auto_enhance(image, params=None):
    """自动检测并增强。返回 (enhanced_image, detection_dict)。
    enhanced_image 为 NumPy(BGR) 格式。
    """
    try:
        det = detect_distortion(image)
        enhanced = enhance_by_type(image, detection=det, params=params)
        return enhanced, det
    except Exception as e:
        # 任何错误返回原图
        try:
            return _to_numpy(image), {'types': [], 'error': str(e)}
        except Exception:
            return image, {'types': [], 'error': str(e)}


# ------------------ 后处理与掩码/IoU 计算 ------------------
def post_denoise(image, params=None):
    """对增强后的图像执行后处理：中值滤波 + 可选的小连通域移除。

    image: NumPy array (BGR) 或 PIL Image
    params: dict, 支持键：median_ksize, min_area
    返回：处理后的 NumPy BGR uint8 图像
    """
    arr = _to_numpy(image)
    if arr is None:
        return image

    cfg = params or {}
    k = int(cfg.get('median_ksize', 3))
    if k % 2 == 0:
        k = max(3, k+1)

    try:
        # median blur works on multi-channel images
        den = cv2.medianBlur(arr, k)
    except Exception:
        den = arr.copy()

    # 如果需要移除小连通域，通过先得到二值化掩码，再移除小区域并应用到图像
    min_area = int(cfg.get('min_area', 20))
    if min_area > 0:
        try:
            gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY) if den.ndim == 3 else den
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # find contours and remove small ones
            contours, _ = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(bw)
            for c in contours:
                area = cv2.contourArea(c)
                if area >= min_area:
                    cv2.drawContours(mask, [c], -1, 255, -1)
            # apply mask to keep large regions, inpaint or blend small regions with median result
            if den.ndim == 3:
                mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                out = (den & (mask3 // 255)) + (arr & (255 - mask3 // 255))
            else:
                out = (den & mask) + (arr & (255 - mask))
        except Exception:
            out = den
    else:
        out = den

    try:
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
    except Exception:
        pass

    return out


def mask_from_image(image, method='otsu', blur=5, min_area=20):
    """从图像生成二值前景掩码（返回 dtype=uint8 的 0/255 掩码）。

    method: 'otsu' or 'adaptive'
    blur: 高斯模糊内核大小（奇数）
    min_area: 去除小连通域阈值（像素）
    """
    arr = _to_numpy(image)
    if arr is None:
        return None

    try:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr.copy()
    except Exception:
        gray = np.array(image, dtype=np.uint8)

    if blur and blur > 0:
        k = int(blur)
        if k % 2 == 0:
            k += 1
        try:
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        except Exception:
            pass

    if method == 'adaptive':
        try:
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        except Exception:
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # remove small components
    try:
        contours, _ = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(bw)
        for c in contours:
            if cv2.contourArea(c) >= int(min_area):
                cv2.drawContours(cleaned, [c], -1, 255, -1)
        return cleaned
    except Exception:
        return bw


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks (uint8 0/255). Returns float in [0,1]."""
    if mask1 is None or mask2 is None:
        return float('nan')
    try:
        m1 = (mask1 > 0).astype(np.uint8)
        m2 = (mask2 > 0).astype(np.uint8)
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        if union == 0:
            return 1.0 if inter == 0 else 0.0
        return float(inter) / float(union)
    except Exception:
        return float('nan')


# ------------------ BRISQUE 支持与参数调优 ------------------
def _try_import_brisque():
    """Import and return a direct BRISQUE scorer using the `brisque` package.

    Assumes `brisque` is installed. Returns a callable score(input) -> float.
    If import fails, raises ImportError.
    """
    try:
        from brisque import BRISQUE
    except Exception as e:
        raise ImportError('Could not import brisque.BRISQUE: ' + str(e))

    b = BRISQUE()

    def _score(inp):
        # Try to coerce input into a 3-channel RGB uint8 ndarray for BRISQUE
        try:
            from PIL import Image

            # If a path (str/bytes) was provided, try to open with PIL
            if isinstance(inp, (bytes, str)):
                try:
                    img = Image.open(inp).convert('RGB')
                    arr = np.array(img)
                except Exception:
                    return float(b.score(inp))

            # If PIL Image-like, convert to RGB ndarray
            elif hasattr(inp, 'convert'):
                try:
                    img = inp.convert('RGB')
                    arr = np.array(img)
                except Exception:
                    return float(b.score(inp))

            else:
                arr = np.array(inp)

            # Ensure 3 channels
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]

            # Convert BGR->RGB by reversing channels (common for OpenCV arrays)
            if arr.ndim == 3 and arr.shape[2] == 3:
                try:
                    arr = arr[:, :, ::-1]
                except Exception:
                    pass

            arr = arr.astype(np.uint8)
            return float(b.score(arr))
        except Exception:
            # Fallback: try direct call and let brisque raise its error
            return float(b.score(inp))

    return _score


def compute_brisque_score(image, as_path=False):
    """Compute BRISQUE score for an image.

    image: either a path (str) or an image array/PIL image.
    as_path: if True, caller ensures `image` is a filesystem path.

    Returns float score (lower is better) or raises ImportError if no brisque lib.
    """
    # Use direct brisque scorer (assumes package installed)
    scorer = _try_import_brisque()
    # If image is a numpy array or PIL image, pass directly; scorer will handle or we fall back to temp file handling in _try_import_brisque wrapper
    return float(scorer(image))


def _generate_candidates(grid):
    """Given a dict of param -> list(values), yield dicts of candidates.
    If the total combinatorial size is large, this helper samples randomly.
    """
    # Compute all combinations size
    keys = list(grid.keys())
    lists = [grid[k] for k in keys]
    total = 1
    for l in lists:
        total *= max(1, len(l))

    # If total small, yield full product
    import itertools
    if total <= 64:
        for comb in itertools.product(*lists):
            yield {k: v for k, v in zip(keys, comb)}
        return

    # Otherwise sample up to 64 combinations randomly
    samples = 64
    for _ in range(samples):
        cand = {}
        for k in keys:
            cand[k] = random.choice(grid[k])
        yield cand


def tune_enhance_params(image_paths, param_grid=None, sample_count=8, max_evals=48, save_best=True):
    """Tune enhancement parameters by minimizing BRISQUE score on a sample of images.

    - image_paths: iterable of image file paths to sample from.
    - param_grid: dict mapping parameter path (dot separated) to list of candidate values.
      If None, uses a sensible default grid for CLAHE/unsharp/detail/inpaint.
    - sample_count: how many images to sample from image_paths (random).
    - max_evals: maximum number of candidate parameter sets to evaluate.
    - save_best: if True, write best params back into `config.json` under `enhance`.

    Returns: (best_params_dict, best_score)
    """
    global CONFIG

    # Ensure we have a list
    all_paths = list(image_paths)
    if not all_paths:
        raise ValueError('No image paths provided for tuning')

    sample_paths = random.sample(all_paths, min(sample_count, len(all_paths)))

    # default grid
    if param_grid is None:
        param_grid = {
            'clahe.clipLimit': [0.5, 1.0, 2.0, 3.0],
            'clahe.tileGridSize': [(8,8), (16,16)],
            'unsharp.weight_main': [1.0, 1.2, 1.5],
            'unsharp.weight_blur': [-0.6, -0.4, -0.2],
            'detail_weight': [0.0, 0.2, 0.4],
            'inpaint.dilation_kernel': [3, 7]
        }

    # Convert grid to nested param dicts
    # First create list of flat candidate dicts
    flat_grid = {}
    for k, v in param_grid.items():
        flat_grid[k] = v

    candidates = []
    for cand in _generate_candidates(flat_grid):
        candidates.append(cand)
        if len(candidates) >= max_evals:
            break

    if not candidates:
        raise ValueError('No parameter candidates generated')

    scorer = _try_import_brisque()
    if scorer is None:
        raise ImportError('No BRISQUE implementation found. Install `brisque` or `imquality`.')

    best_score = float('inf')
    best_params = None

    print(f"调参样本数={len(sample_paths)}; 候选集大小={len(candidates)}")

    # Prepare CSV to log per-image scores for each candidate
    csv_dir_cfg = CONFIG.get('results', {}).get('csv_dir', 'results/csv')
    results_csv_dir = _resolve_path(csv_dir_cfg)
    os.makedirs(results_csv_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tuning_csv = os.path.join(results_csv_dir, f"tuning_details_{timestamp}.csv")
    try:
        csv_f = open(tuning_csv, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(['candidate_index', 'candidate_flat', 'image_path', 'original_brisque', 'enhanced_brisque'])
        csv_f.flush()
        print(f"Tuning details CSV: {tuning_csv}")
    except Exception as e:
        csv_f = None
        csv_writer = None
        print(f"无法创建 tuning CSV: {e}")

    for i, cand in enumerate(candidates, 1):
        # build nested params dict from flat cand (dot-separated keys)
        params = CONFIG.get('enhance', {}).copy()
        for fk, val in cand.items():
            parts = fk.split('.')
            d = params
            for p in parts[:-1]:
                if p not in d or not isinstance(d[p], dict):
                    d[p] = {}
                d = d[p]
            d[parts[-1]] = val

        scores = []
        try:
            for pth in sample_paths:
                try:
                    img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        print(f"无法读取示例图像 {pth}, 跳过")
                        continue
                    # original BRISQUE (pass path directly; scorer can handle paths)
                    try:
                        orig_sc = float(scorer(pth))
                    except Exception:
                        orig_sc = float('nan')

                    det = detect_distortion(img)
                    enhanced = enhance_by_type(img, detection=det, params=params)

                    # compute brisque on enhanced image (write tmp file and call scorer)
                    fd, tmp = tempfile.mkstemp(suffix='.png')
                    os.close(fd)
                    cv2.imwrite(tmp, enhanced)
                    try:
                        enhanced_sc = float(scorer(tmp))
                    finally:
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass

                    scores.append(enhanced_sc)

                    # write per-image row to tuning CSV (if available)
                    if csv_writer is not None:
                        try:
                            csv_writer.writerow([i, json.dumps(cand, ensure_ascii=False), pth, orig_sc, enhanced_sc])
                            csv_f.flush()
                        except Exception:
                            pass
                except Exception:
                    traceback.print_exc()
                    continue

            if not scores:
                avg = float('inf')
            else:
                avg = float(np.mean(scores))

            print(f"[{i}/{len(candidates)}] avg BRISQUE={avg:.4f} params={cand}")

            if avg < best_score:
                best_score = avg
                best_params = cand.copy()
        except Exception:
            traceback.print_exc()
            continue

    if best_params is None:
        raise RuntimeError('无法找到最佳参数（评估失败）')

    # Convert best_params flat -> nested
    nested_best = CONFIG.get('enhance', {}).copy()
    for fk, val in best_params.items():
        parts = fk.split('.')
        d = nested_best
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = val

    if save_best:
        try:
            cfg_path = os.path.join(base_dir, 'config.json')
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            cfg = CONFIG.copy()

        # update enhance section
        if 'enhance' not in cfg or not isinstance(cfg['enhance'], dict):
            cfg['enhance'] = {}
        # merge nested_best into cfg['enhance']
        def _merge(dst, src):
            for kk, vv in src.items():
                if isinstance(vv, dict) and isinstance(dst.get(kk), dict):
                    _merge(dst[kk], vv)
                else:
                    dst[kk] = vv

        _merge(cfg['enhance'], nested_best)

        # store a metadata section with tuning timestamp and score
        cfg.setdefault('_tuning', {})
        cfg['_tuning']['last_tuned'] = datetime.datetime.now().isoformat()
        cfg['_tuning']['best_brisque'] = float(best_score)

        try:
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"已将最佳参数写入 {cfg_path}，best_brisque={best_score:.4f}")
        except Exception as e:
            print(f"写入 config.json 失败: {e}")

        # reload CONFIG in-memory
        try:
            CONFIG = _load_config()
        except Exception:
            pass

    return nested_best, best_score