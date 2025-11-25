import cv2
import numpy as np
import os
import datetime

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
    # 转换为float32类型，方便后续计算
    float_img = image.astype(np.float32)
    # 防止log(0)出现无穷大
    float_img += 1
    # 对图像取对数：ln(f(x, y))
    log_float_img = np.log(float_img)
    
    # 对图像扩展到最佳DFT尺寸
    m = cv2.getOptimalDFTSize(image.shape[0])
    n = cv2.getOptimalDFTSize(image.shape[1])
    # 填充图像
    padded = np.pad(log_float_img, ((0, m - image.shape[0]), (0, n - image.shape[1])), 
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
    realI = planes[0][:image.shape[0], :image.shape[1]]
    
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

# 定量评估
def evaluate_image(original, enhanced, filename):
    """
    计算并输出图像的质量评估指标

    参数:
        original: 原始图像
        enhanced: 增强后的图像
        filename: 图像文件名，用于输出到文件
    """

    # 确保图像是灰度图用于质量评估
    if len(original.shape) > 2:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original.copy()
    
    if len(enhanced.shape) > 2:
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        enhanced_gray = enhanced.copy()
    
    # 初始化结果字典
    results = {
        'filename': filename,
        'original_sharpness': 0,
        'enhanced_sharpness': 0,
        'original_contrast': 0,
        'enhanced_contrast': 0,
        'original_entropy': 0,
        'enhanced_entropy': 0
    }
    
    try:
        # 1. 计算清晰度 (拉普拉斯方差)
        # 值越高表示图像越清晰
        results['original_sharpness'] = cv2.Laplacian(original_gray, cv2.CV_64F).var()
        results['enhanced_sharpness'] = cv2.Laplacian(enhanced_gray, cv2.CV_64F).var()
    except Exception as e:
        print(f"计算清晰度时出错: {e}")
    
    try:
        # 2. 计算对比度 (标准差)
        # 值越高表示对比度越强
        results['original_contrast'] = np.std(original_gray)
        results['enhanced_contrast'] = np.std(enhanced_gray)
    except Exception as e:
        print(f"计算对比度时出错: {e}")
    
    try:
        # 3. 计算熵 (信息熵)
        # 值越高表示图像信息量越大
        results['original_entropy'] = calculate_entropy(original_gray)
        results['enhanced_entropy'] = calculate_entropy(enhanced_gray)
    except Exception as e:
        print(f"计算熵时出错: {e}")
    
    # 将结果写入文件
    output_file = 'processed_comparsion.txt'
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        if not file_exists:
            # 写入表头
            f.write("文件名\t原始清晰度\t增强后清晰度\t原始对比度\t增强后对比度\t原始熵\t增强后熵\n")
        
        # 写入数据，使用制表符分隔
        f.write(f"{results['filename']}\t")
        f.write(f"{results['original_sharpness']:.4f}\t")
        f.write(f"{results['enhanced_sharpness']:.4f}\t")
        f.write(f"{results['original_contrast']:.4f}\t")
        f.write(f"{results['enhanced_contrast']:.4f}\t")
        f.write(f"{results['original_entropy']:.4f}\t")
        f.write(f"{results['enhanced_entropy']:.4f}\n")
    
    print(f"质量评估结果已保存到 {output_file}")
    return results


# 保存处理前后的图像
def save_image(src, result, image_path):
    # 创建并排显示的图像
    # 设置缩放比例，缩小图像尺寸
    scale_factor = 0.6  # 可以根据需要调整缩放比例
    
    # 获取原始图像尺寸并缩小
    h1, w1 = src.shape[:2]
    new_w1 = int(w1 * scale_factor)
    new_h1 = int(h1 * scale_factor)
    
    # 获取处理后图像尺寸并缩小（保持相同的缩放比例）
    h2, w2 = result.shape[:2]
    new_w2 = int(w2 * scale_factor)
    new_h2 = int(h2 * scale_factor)
    
    # 缩放到统一的尺寸（取较小的宽度和高度，确保两张图尺寸一致）
    unified_w = min(new_w1, new_w2)
    unified_h = min(new_h1, new_h2)
    
    # 缩放图像
    src_resized = cv2.resize(src, (unified_w, unified_h))
    result_resized = cv2.resize(result, (unified_w, unified_h))
    
    # 确保两张图都是彩色图（3通道）
    if len(src_resized.shape) == 2:
        src_resized = cv2.cvtColor(src_resized, cv2.COLOR_GRAY2BGR)
    if len(result_resized.shape) == 2:
        result_resized = cv2.cvtColor(result_resized, cv2.COLOR_GRAY2BGR)
    
    # 创建新图像，宽度为两个图像宽度之和，高度为统一高度
    vis = np.zeros((unified_h, unified_w * 2, 3), np.uint8)
    
    # 将缩放后的图像复制到新图像中
    vis[:, :unified_w] = src_resized
    vis[:, unified_w:unified_w*2] = result_resized
    
    # 为了解决中文乱码问题，我们可以使用图像拼接的方式添加标签
    # 创建标签区域
    label_height = 40
    labeled_image = np.ones((unified_h + label_height, unified_w * 2, 3), np.uint8) * 255  # 白色背景
    
    # 将图像内容复制到带标签的图像中
    labeled_image[label_height:, :] = vis
    
    # 添加标签（使用ASCII字符作为替代，避免中文乱码）
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(labeled_image, 'ORIGINAL', (unified_w//4 - 30, 25), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(labeled_image, 'PROCESSED', (unified_w + unified_w//4 - 30, 25), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    
    # 保存合并后的图像到enhance_results文件夹，使用原始图片名称+时间戳命名
    # 确保enhance_results文件夹存在
    results_dir = 'enhance_results'
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

# 主函数
def main():
    # 从pictures文件夹中读取图像
    if not os.path.exists("pictures"):
        print("pictures文件夹不存在")
        return -1
    
    # 获取pictures文件夹中的所有图片文件
    image_files = []
    for file in os.listdir("pictures"):
        # 简单检查文件扩展名是否为常见图片格式
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
            image_files.append(file)
    
    # 如果没有找到图片文件
    if not image_files:
        print("pictures文件夹中没有图片文件")
        return -1
    
    # 加载第一张图片
    image_path = os.path.join("pictures", image_files[0])
    image_name = os.path.basename(image_path)
    src = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if src is None:
        print(f"无法加载图像: {image_path}")
        return -1
    
    result = None
    
    # 检查图像通道数
    # 这是一张灰度图像
    if len(src.shape) == 2:
        # 灰度图像处理
        result = homomorphic_filter(src, 1.5, 0.5, 1.0, 30.0)
    elif len(src.shape) == 3 and src.shape[2] == 3:
        # 彩色图像处理（逐通道）
        channels = cv2.split(src)
        result_channels = []
        
        for i in range(3):
            result_channels.append(homomorphic_filter(channels[i], 1.5, 0.5, 1.0, 30.0))
        
        result = cv2.merge(result_channels)
    else:
        print("不支持的图像通道数")
        return -1
    
    # 定量比较处理前后的图片
    evaluate_image(src, result, image_name)

    # 保存处理前后的图像
    save_image(src, result, image_path)
    
    # 不需要显示图像，直接销毁所有窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()