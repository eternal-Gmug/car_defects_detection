import cv2
import numpy as np
import os
import time
from PIL import Image

# 保存二值图像（作为中间产物）
def save_binary_image(binary_img, image_name):
    """
    保存二值化图像到binary_results目录
    
    参数:
        binary_img: ndarray, 二值化后的图像
        image_name: str, 原始图像文件名（用于生成新文件名）
    """
    # 创建binary_results目录（如果不存在）
    os.makedirs("binary_results", exist_ok=True)
    # 生成新文件名，在原文件名前添加"binary_"，在后面添加当前时间戳
    image_name_without_ext = os.path.splitext(image_name)[0]
    binary_name = "binary_" + image_name_without_ext + "_" + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".jpg"
    # 将二值化图像可视化后保存，不要保存成一个二进制文件
    binary_img_visual = Image.fromarray(binary_img)
    binary_img_visual.save(os.path.join("binary_results", binary_name))
    print(f"二值化图像已保存为: {binary_name}")

# 使用 Bradley–Roth 积分图自适应阈值得到初始二值图
def bradley_roth_threshold(image, window_size=15, threshold_coefficient=0.15):
    """
    使用 Bradley-Roth 积分图自适应阈值算法将图像二值化
    
    参数:
        image: ndarray, 输入图像（灰度图）
        window_size: int, 局部窗口大小（自适应调整）
        threshold_coefficient: float, 阈值系数,通常在0.1到0.2之间
    
    返回:
        ndarray: 二值化后的图像
    """
    # 确保输入图像为灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 转换为float32类型以避免整数溢出
    gray_float = gray.astype(np.float32)
    
    # 获取图像尺寸
    height, width = gray.shape
    
    # 计算积分图
    integral_img = cv2.integral(gray_float)
    
    # 初始化结果图像
    binary_img = np.zeros_like(gray, dtype=np.uint8)
    
    # 计算窗口的一半大小，用于边界处理
    half_window = window_size // 2
    
    # 遍历每个像素
    for i in range(height):
        for j in range(width):
            # 计算窗口的边界，确保在图像范围内
            top = max(0, i - half_window)
            bottom = min(height, i + half_window + 1)
            left = max(0, j - half_window)
            right = min(width, j + half_window + 1)
            
            # 计算窗口内的像素数量
            window_area = (bottom - top) * (right - left)
            
            # 使用积分图计算窗口内的像素总和
            # 积分图的计算方式：sum = integral[bottom,right] - integral[top,right] - integral[bottom,left] + integral[top,left]
            sum_window = integral_img[bottom, right] - integral_img[top, right] - integral_img[bottom, left] + integral_img[top, left]
            
            # 计算窗口内的平均像素值
            mean_value = sum_window / window_area
            
            # 应用Bradley-Roth阈值规则：如果像素值大于(1 - 阈值系数) * 局部平均值，则设为255，否则为0
            if gray[i, j] > (1 - threshold_coefficient) * mean_value:
                binary_img[i, j] = 255
            else:
                binary_img[i, j] = 0
    
    return binary_img

def traditional_segmentation(image, image_name):
    """
    使用传统的阈值分割方法将图像二值化

    参数:
        image: ndarray, 输入图像（灰度图）
    
    返回:
        ndarray: 带分割轮廓的原始图像
    """
    binary_img = bradley_roth_threshold(image, window_size=15, threshold_coefficient=0.145)
    # 将该二值图像保存到binary_results目录
    save_binary_image(binary_img, image_name)
    # 使用形态学操作进行去噪与填洞
    

