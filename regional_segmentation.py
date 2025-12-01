import cv2
from cv2.gapi import mask
import numpy as np
import os
import time
import yaml
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

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

def save_segmentation_image(contour_image, image_name):
    """
    保存分割图像到segmentation_results目录
    
    参数:
        contour_image: ndarray, 包含轮廓的图像
        image_name: str, 原始图像文件名（用于生成新文件名）
    """
    os.makedirs("segmentation_results", exist_ok=True)
    image_name_without_ext = os.path.splitext(image_name)[0]
    contour_name = image_name_without_ext + "_" + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".jpg"
    cv2.imwrite(os.path.join("segmentation_results", contour_name), contour_image)
    print(f"带轮廓图像已保存为: {contour_name}")

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

def local_otsu_threshold(image, window_size=15, global_weight=0.5):
    """
    使用局部OTSU自适应阈值算法将图像二值化
    
    参数:
        image: ndarray, 输入图像（灰度图）
        window_size: int, 局部窗口大小
        global_weight: float, 全局OTSU阈值与局部OTSU阈值的权重（0.0-1.0）
    
    返回:
        ndarray: 二值化后的图像
    """
    # 确保输入图像为灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 获取图像尺寸
    height, width = gray.shape
    
    # 计算全局OTSU阈值
    global_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 初始化结果图像
    binary_img = np.zeros_like(gray, dtype=np.uint8)
    
    # 计算窗口的一半大小，用于边界处理
    half_window = window_size // 2
    
    # 遍历每个像素
    for i in range(height):
        for j in range(width):
            # 计算窗口的边界，确保在图像范围内
            top = max(0, i - half_window)
            bottom = min(height, i + half_window)
            left = max(0, j - half_window)
            right = min(width, j + half_window)
            
            # 提取局部区域
            local_region = gray[top:bottom, left:right]
            
            # 计算局部区域的OTSU阈值
            # 对于过小的区域，使用全局阈值
            if local_region.size > 10:
                local_thresh, _ = cv2.threshold(local_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # 结合全局和局部阈值
                combined_thresh = global_weight * global_thresh + (1 - global_weight) * local_thresh
            else:
                combined_thresh = global_thresh
            
            # 应用阈值
            if gray[i, j] > combined_thresh:
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
    binary_img = bradley_roth_threshold(image, window_size=15, threshold_coefficient=0.13)
    # binary_img = local_otsu_threshold(image, window_size=15, global_weight=0.3)
    # 将该二值图像保存到binary_results目录
    # save_binary_image(binary_img, image_name)
    # 使用形态学操作进行去噪与填洞
    kernel = np.ones((5, 5), np.uint8)
    # 1. 首先反转图像，使前景变成白色，背景变成黑色
    binary_img = cv2.bitwise_not(binary_img)
    # 2. 应用形态学操作
    # 对图像做去噪处理，使用腐蚀操作，将图像中的小孔洞填充
    # binary_img = cv2.erode(binary_img, kernel, iterations=1)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    # binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
    # 3. 检测中间产物， 查看形态学操作后得出的结果
    time.sleep(1)
    save_binary_image(binary_img, image_name)
    # 查找轮廓
    '''
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在原始图像上绘制轮廓
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    # 将带轮廓的图像保存到segmentation_results目录
    save_segmentation_image(image_with_contours, image_name)
    return image_with_contours
    '''
    
# 使用yolov8n-seg.pt分割模型进行训练

# 验证数据集的yaml文件是否正确
def validate_data_yaml(data_yaml_path):
    """
    验证数据集的yaml文件是否正确
    """
    try:
        with open(data_yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        for key in ['train', 'val']:
            if key in data:
                img_path = Path(data[key])
                if img_path.exists():
                    img_count = len(list(img_path.glob('*.jpg'))) + len(list(img_path.glob('*.png')))
                    print(f"  {key}图像数: {img_count}")
                    # 检查对应的labels目录
                    labels_dir = str(img_path).replace('images', 'labels')
                    if os.path.exists(labels_dir):
                        label_count = len(list(Path(labels_dir).glob('*.txt')))
                        print(f"  {key}标签数: {label_count}")
                    else:
                        print(f"  ⚠️ 警告: {labels_dir} 不存在!")
                else:
                    print(f"  ⚠️ 警告: {img_path} 不存在!")
    except Exception as e:
        print(f"数据集yaml文件验证失败: {e}")

def process_labels_for_segmentation(data_yaml_path):
    """
    预处理标签文件,将多边形标注转换为YOLOv8分割任务要求的格式。
    转换格式：`class_id x_center y_center width height poly_x1 poly_y1 ... poly_xn poly_yn`
    
    参数:
        data_yaml_path: data.yaml文件路径
    """
    try:
        with open(data_yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        data_root = os.path.dirname(data_yaml_path)
        
        for split in ['train', 'val']:
            if split in data:
                img_rel_path = data[split]
                labels_rel_path = img_rel_path.replace('images', 'labels')
                labels_dir = os.path.join(data_root, labels_rel_path)
                
                if os.path.exists(labels_dir):
                    print(f"\n正在处理 {split} 集标签目录: {labels_dir}")
                    label_files = list(Path(labels_dir).glob('*.txt'))
                    
                    processed_count = 0
                    error_count = 0
                    
                    for label_file in label_files:
                        try:
                            with open(label_file, 'r') as f:
                                lines = f.readlines()
                            
                            if not lines:
                                # 如果是空文件（代表无目标的图像），直接跳过或保留空文件
                                continue
                            
                            new_lines = []
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue
                                
                                parts = list(map(float, line.split()))
                                class_id = int(parts[0])
                                polygon_coords = parts[1:]
                                
                                # 1. 基本检查
                                if len(polygon_coords) < 6:  # 至少需要3个点（6个坐标）才能构成多边形
                                    print(f"  ⚠️ 跳过：多边形坐标点不足（至少需3个点） @ {label_file.name}: {line}")
                                    error_count += 1
                                    continue
                                if len(polygon_coords) % 2 != 0:
                                    print(f"  ⚠️ 跳过：坐标数量不是偶数 @ {label_file.name}")
                                    error_count += 1
                                    continue
                                
                                # 2. 提取所有x和y坐标
                                xs = polygon_coords[0::2]  # 所有x坐标
                                ys = polygon_coords[1::2]  # 所有y坐标
                                
                                # 3. 计算最小外接矩形（边界框）
                                x_min, x_max = min(xs), max(xs)
                                y_min, y_max = min(ys), max(ys)
                                
                                # 4. 计算YOLO格式的边界框 (x_center, y_center, width, height)
                                x_center = (x_min + x_max) / 2.0
                                y_center = (y_min + y_max) / 2.0
                                width = x_max - x_min
                                height = y_max - y_min
                                
                                # 5. 构建新的标签行：类别 + 边界框 + 原始多边形
                                new_line_parts = [str(class_id), 
                                                  f"{x_center:.6f}", f"{y_center:.6f}", 
                                                  f"{width:.6f}", f"{height:.6f}"]
                                # 添加多边形坐标，保留原始精度或适当格式化
                                new_line_parts.extend([f"{coord:.6f}" for coord in polygon_coords])
                                
                                new_lines.append(' '.join(new_line_parts))
                            
                            # 6. 将转换后的内容写回文件
                            if new_lines:
                                with open(label_file, 'w') as f:
                                    f.write('\n'.join(new_lines))
                                processed_count += 1
                            else:
                                # 如果文件内所有行都被跳过，写入一个空文件（表示无目标）
                                with open(label_file, 'w') as f:
                                    pass  # 创建空文件
                                
                        except Exception as e:
                            print(f"  ❌ 处理文件 {label_file.name} 时出错: {e}")
                            error_count += 1
                    
                    print(f"    完成！成功处理 {processed_count} 个文件，{error_count} 个错误。")
                    
                else:
                    print(f"  ⚠️ 警告: 标签目录不存在 {labels_dir}")
        
    except Exception as e:
        print(f"预处理标签文件失败: {e}")

# 评估yolov8n-seg.pt模型
def evaluate_model(model, data_yaml_path, split='val'):
    """
    在验证集上评估模型并计算IoU指标
    
    参数:
        model: 训练好的模型
        data_yaml_path: data.yaml路径
        split: 评估的数据集分割 ('val' 或 'test')
    """
    
    # 同时进行常规的模型评估以获取整体指标
    metrics = model.val(
        data = data_yaml_path,
        split = split,
        device = 'cpu',
        # 评估指标为segmentation
        task = 'segment',
        imgsz = 640,
        save = True,   # 保存验证结果
        plots = True   # 生成评估图表
    )
    
    # 打印分割任务的关键指标
    print("\n整体评估结果:")
    print(f"  精确度 (Precision): {metrics.box.p:.4f}")
    print(f"  召回率 (Recall): {metrics.box.r:.4f}")
    print(f"  mAP@0.5: {metrics.box.map50:.4f}")       # 衡量了预测边界框与真实边界框重叠度达到50%以上的平均检测准确率
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")    # 计算IoU阈值从0.5到0.95，以0.05为步长（共10个阈值：0.5, 0.55, 0.6, ..., 0.95）的mAP平均值
    # 给出模型在验证集上的损失
    print(f"  验证集损失: {metrics.loss:.4f}")
    # 给出模型在验证集上的平均IoU
    print(f"  验证集平均IoU: {metrics.box.iou:.4f}")
    
    # 保存整体评估结果到文件
    with open('yolov8_segmentation_metrics.txt', 'w') as f:
        f.write(f"精确度 (Precision): {metrics.box.p:.4f}\n")
        f.write(f"召回率 (Recall): {metrics.box.r:.4f}\n")
        f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
        f.write(f"验证集损失: {metrics.loss:.4f}\n")
        f.write(f"验证集平均IoU: {metrics.box.iou:.4f}\n")
        f.write(f"\n每张图片的详细IoU信息请查看: {iou_file_path}")

# 训练yolov8n-seg.pt模型
def train_yolov8n_segmentation(data_yaml_path, epochs=150, imgsz=640):
    """
    训练YOLOv8n分割模型
    
    参数:
        data_yaml_path: data.yaml文件路径
        epochs: 训练轮次
        imgsz: 图像尺寸(数据集已经将图像resize到640x640)
    """
    # 从本地加载yolov8n-seg.pt模型
    local_model_path = './yolo_model/yolov8n-seg.pt'
    model = YOLO(local_model_path)
    # 训练模型
    results = model.train(
        data = data_yaml_path,      # 数据配置文件路径
        epochs = epochs,            # 训练轮次（该数据集规模较小，可以适当增大轮次）
        task = 'segment',           # 任务类型为分割
        imgsz = imgsz,              # 图像尺寸
        batch = 16,                 # 批处理大小
        device = 'cpu',             # 使用CPU训练
        pretrained = True,          # 从预训练模型开始训练
        seed = 42,                  # 随机种子，确保可重复性
        patience = 30,              # 训练轮次无改进时，提前停止训练，避免过拟合
        workers = 4,                # 数据加载线程数
        save = True,                # 保存检查点
        save_period = 10,           # 每10轮保存一次检查点
        val = True,                 # 开启验证集评估
        # 数据增强配置（当前数据集规模较小，可以适当增大数据增强参数）
        augment = True,             # 开启数据增强
        # 其他增强参数（根据数据集调整）
        hsv_h = 0.015,              # HSV颜色空间的色调偏移
        hsv_s = 0.7,                # 饱和度偏移
        hsv_v = 0.4,                # 亮度偏移
        degrees = 0.0,              # 旋转角度范围
        translate = 0.1,            # 平移范围
        scale = 0.5,                # 缩放范围
        shear = 0.0,                # 剪切角度范围
        perspective = 0.0,          # 透视变换范围
        flipud = 0.5,               # 上下翻转概率
        fliplr = 0.5,               # 左右翻转概率
        mosaic = 0.5,               # 马赛克增强概率
        mixup = 0.1,                # mixup增强概率
        erasing = 0.1,              # 随机擦除概率
    )
    return results, model

def train_yolov8n_segmentation_model():
    """
    使用yolov8n-seg.pt分割模型进行训练
    """
    # 1. 设置路径和参数
    data_yaml_path = './data/data.yaml'
    epochs = 50  # 训练轮次

    # 2.验证数据集结构
    # print("验证数据集结构...")
    # validate_data_yaml(data_yaml_path)
    
    # 3. 预处理标签文件
    print("预处理标签文件，确保多边形轮廓格式正确...")
    process_labels_for_segmentation(data_yaml_path)

    # 4.训练模型
    print("开始训练yolov8n-seg.pt模型...")
    results, model = train_yolov8n_segmentation(data_yaml_path, epochs=epochs)

    # 5. 评估模型
    # print("开始评估yolov8n-seg.pt模型...")
    # evaluate_model(model, data_yaml_path, split='val')

    # 6. 保存模型
    print("开始保存yolov8n-seg.pt模型...")
    model.save('yolo_model/yolov8n_segmentation_model.pt')
    print("模型训练、评估和保存完成!")

# 当前不能识别出灰尘和水渍类别,只能识别出划痕等多边形类别
def use_model_for_segmentation(image, image_name, imgsz=640):
    """
    使用训练好的分割模型对输入图像进行分割
    
    参数:
        image: 输入增强后的图像(格式为numpy.ndarray)
        image_name: 图像文件名(用于保存分割结果)
        imgsz: 图像尺寸
    
    返回:
        results: 分割结果对象
    """
    # 加载模型
    model = YOLO('yolo_model/yolov8n-seg.pt')
    
    # 执行分割推理
    results = model(image, task='segment', device='cpu', imgsz=imgsz)
    
    # 提取分割结果中的轮廓（未完成）
    contours = results[0].masks.xy

        
        
if __name__ == '__main__':
    train_yolov8n_segmentation_model()

