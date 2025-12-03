import cv2
import numpy as np
import os
import datetime
from skimage.metrics import structural_similarity as ssim

# 构造高斯型高通同态滤波器
def create_high_pass_filter(size, gammaH, gammaL, c, D0):
    H = np.zeros(size, dtype=np.float32)
    center = (size[1] // 2, size[0] // 2)      # 中心坐标
    for u in range(size[0]):
        for v in range(size[1]):
            D = np.sqrt((u - center[1]) ** 2 + (v - center[0]) ** 2)    # 频率点到中心的距离
            H[u, v] = (gammaH - gammaL) * (1.0 - np.exp(-c * (D * D) / (D0 * D0))) + gammaL
    return H

# 单通道同态滤波
def homomorphic_filter(image, gammaH, gammaL, c, D0):
    ### @return arr : 返回的是同态滤波后的图像数组，格式是np.array(np.uint8)

    # 将PIL Image对象转换为NumPy数组
    if hasattr(image, 'mode'):
        img_array = np.array(image)
    else:
        img_array = image
    
    # 转换为float32类型，方便后续计算
    float_img = img_array.astype(np.float32)
    # 防止log(0)出现无穷大
    float_img += 1
    # 对图像取对数：ln(f(x, y))
    log_float_img = np.log(float_img)
    
    # 对图像扩展到最佳DFT尺寸
    m = cv2.getOptimalDFTSize(img_array.shape[0])
    n = cv2.getOptimalDFTSize(img_array.shape[1])
    # 填充图像
    padded = np.pad(log_float_img, ((0, m - img_array.shape[0]), (0, n - img_array.shape[1])), 
                    mode='constant', constant_values=0)
    
    # 创建复数矩阵，实部为padded，虚部为0
    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv2.merge(planes)
    
    # 进行傅里叶变换
    cv2.dft(complexI, complexI)
    
    # 创建高通同态滤波器，尺寸和padded一致
    H = create_high_pass_filter(padded.shape, gammaH, gammaL, c, D0)
    
    # 将频谱从笛卡尔坐标转换为极坐标（幅值和相位）
    planes = cv2.split(complexI)
    mag, phase = cv2.cartToPolar(planes[0], planes[1])
    
    # 在频率域应用滤波器（只对幅值部分）
    mag = mag * H
    
    # 将极坐标转换回笛卡尔坐标
    # 将元组转换为列表以便修改元素
    planes = list(planes)
    planes[0], planes[1] = cv2.polarToCart(mag, phase)
    complexI = cv2.merge(planes)
    
    # 傅里叶逆变换，回到空域
    cv2.idft(complexI, complexI)
    
    # 提取实部，去掉填充部分
    planes = cv2.split(complexI)
    realI = planes[0][:img_array.shape[0], :img_array.shape[1]]
    
    # 归一化到[0, 10]，避免指数溢出
    cv2.normalize(realI, realI, 0, 10, cv2.NORM_MINMAX)
    
    # 对结果取指数，还原到线性空间
    img_out = np.exp(realI)
    
    # 最后归一化到[0, 255]并转换为8位图像
    min_val, max_val = np.min(img_out), np.max(img_out)
    img_out = ((img_out - min_val) / (max_val - min_val)) * 255.0
    img_out = img_out.astype(np.uint8)
    
    return img_out

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
def evaluate_image(original, enhanced, filename):
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
    
    # 将结果写入文件
    output_file = 'processed_comparsion.txt'
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        if not file_exists:
            # 写入表头
            f.write("文件名\t原始清晰度\t增强后清晰度\t原始局部15对比度\t增强后局部15对比度\t原始熵\t增强后熵\t峰值信噪比(PSNR)\t结构相似性(SSIM)\n")
        
        # 写入数据，使用制表符分隔
        f.write(f"{results['filename']}\t")
        f.write(f"{results['original_sharpness']:.4f}\t")
        f.write(f"{results['enhanced_sharpness']:.4f}\t")
        f.write(f"{results['original_contrast']:.4f}\t")
        f.write(f"{results['enhanced_contrast']:.4f}\t")
        f.write(f"{results['original_entropy']:.4f}\t")
        f.write(f"{results['enhanced_entropy']:.4f}\t")
        f.write(f"{results['psnr']:.4f}\t")
        f.write(f"{results['ssim']:.4f}\n")
    
    print(f"质量评估结果已保存到 {output_file}")
    return results

# 保存处理前后的图像
def save_image(src, result, image_path,  output_path=None):
    # 创建并排显示的图像
    # 设置缩放比例，缩小图像尺寸
    scale_factor = 0.6  # 可以根据需要调整缩放比例
    
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
    
    # 添加标签（使用ASCII字符作为替代，避免中文乱码）
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(labeled_image, 'ORIGINAL', (unified_w//4 - 30, 25), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(labeled_image, 'PROCESSED', (unified_w + unified_w//4 - 30, 25), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    
    # 保存合并后的图像到enhance_results文件夹，使用原始图片名称+时间戳命名
    # 确保enhance_results文件夹存在
    results_dir = output_path if output_path else 'enhance_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
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