import cv2
import numpy as np
import os
import time
import yaml
import general_function as gf
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

########################### 传统分割模块 ###################################
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

########################### 模型训练模块 ###################################
# 处理原始标签文件（一次性函数，用于将多边形标签格式和标注框标签格式的混合格式文件夹分成两个文件夹）
def process_labels_for_segmentation(data_yaml_path):
    """
    预处理标签文件,将多边形标签格式和标注框标签格式分成两个文件夹。
    多边形标签格式：`class_id poly_x1 poly_y1 ... poly_xn poly_yn`
    标注框标签格式：`class_id x_center y_center width height`
    
    参数:
        data_yaml_path: data.yaml文件路径
    """
    try:
        with open(data_yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        data_root = os.path.dirname(data_yaml_path)
        
        for split in ['train', 'val', 'test']:
            if split in data:
                img_rel_path = data[split]
                labels_rel_path = img_rel_path.replace('images', 'labels')
                labels_dir = os.path.join(data_root, labels_rel_path)
                # 创建一个灰尘水渍的新标签文件夹
                labels_det_dir = os.path.join(data_root, labels_rel_path.replace('labels', 'labels_det'))
                os.makedirs(labels_det_dir, exist_ok=True)
                
                if os.path.exists(labels_dir):
                    print(f"\n正在处理 {split} 集标签目录: {labels_dir}")
                    label_files = list(Path(labels_dir).glob('*.txt'))
                    
                    processed_count = 0
                    error_count = 0
                    
                    for label_file in label_files:
                        try:
                            # 创建一个新的标注框txt标签文件，如果不存在则创建
                            label_det_file = os.path.join(labels_det_dir, label_file.name)
                            os.makedirs(os.path.dirname(label_det_file), exist_ok=True)
                            
                            # 确保无论如何都创建标签文件（包括空文件）
                            with open(label_det_file, 'w') as f:
                                pass  # 创建空文件

                            with open(label_file, 'r') as f:
                                lines = f.readlines()
                            
                            if not lines:
                                # 如果是空文件（代表无目标的图像），直接跳过或保留空文件
                                processed_count += 1
                                continue
                            
                            new_lines = []
                            det_lines = []  # 存储检测格式的标签行
                            
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    # 如果是空行，直接跳过
                                    continue
                                
                                parts = list(map(float, line.split()))
                                polygon_coords = parts[1:]
                                
                                # 基本检查
                                if len(polygon_coords) == 4:  # 这说明是一个标注框的yolo格式
                                    print(f"  ⚠️ 这是标注框格式: 抛给新标签文件 @ {label_file.name}: {line}")
                                    # 添加到检测标签行列表
                                    det_lines.append(line)
                                    processed_count += 1
                                    continue
                                if len(polygon_coords) % 2 != 0:
                                    print(f"  ⚠️ 跳过: 这是真的错误,坐标数量不是偶数 @ {label_file.name}")
                                    processed_count += 1
                                    error_count += 1
                                    continue
                                
                                # 正常多边形加入新列表
                                new_lines.append(line)
                                processed_count += 1
                                
                            # 将新列表写入原始标签文件
                            with open(label_file, 'w') as f:
                                if new_lines:  # 只有当有内容时才写入，避免末尾的空行
                                    f.write('\n'.join(new_lines) + '\n')
                            
                            # 将检测格式标签行写入检测标签文件
                            with open(label_det_file, 'w') as f:
                                if det_lines:  # 只有当有内容时才写入，避免末尾的空行
                                    f.write('\n'.join(det_lines) + '\n')

                        except Exception as e:
                            print(f" 处理文件 {label_file.name} 时出错: {e}")
                            processed_count += 1
                            error_count += 1
                            
                    print(f" 完成！成功处理 {processed_count} 个文件，{error_count} 个错误。")          
                else:
                    print(f" 警告: 标签目录不存在 {labels_dir}")
    except Exception as e:
        print(f"预处理标签文件失败: {e}")

# 整体评估yolov8n-seg.pt模型
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
    # 获取并打印指标
    try:
        # 获取box指标
        precision = metrics.box.mp if hasattr(metrics.box, 'mp') else 0
        recall = metrics.box.mr if hasattr(metrics.box, 'mr') else 0
        map50 = metrics.box.map50 if hasattr(metrics.box, 'map50') else 0
        map_all = metrics.box.map if hasattr(metrics.box, 'map') else 0
        
        # 获取seg指标
        seg_map50 = metrics.seg.map50 if hasattr(metrics.seg, 'map50') else 0
        seg_map = metrics.seg.map if hasattr(metrics.seg, 'map') else 0
        
        # 获取损失
        loss = metrics.loss if hasattr(metrics, 'loss') else 0
        
        print(f"  精确度 (Precision): {precision:.4f}")
        print(f"  召回率 (Recall): {recall:.4f}")
        print(f"  mAP@0.5: {map50:.4f}")
        print(f"  mAP@0.5:0.95: {map_all:.4f}")
        print(f"  验证集损失: {loss:.4f}")
        print(f"  分割任务mAP@0.5: {seg_map50:.4f}")
        print(f"  分割任务mAP@0.5:0.95: {seg_map:.4f}")
        
    except Exception as e:
        print(f"访问指标时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存整体评估结果到文件，结果为英文
    with open('segmentation_results/val_yolo_outputs/yolov8n_segmentation_metrics.txt', 'w') as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"mAP@0.5: {map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {map_all:.4f}\n")
        f.write(f"Validation Loss: {loss:.4f}\n")
        f.write(f"Segmentation mAP@0.5: {seg_map50:.4f}\n")
        f.write(f"Segmentation mAP@0.5:0.95: {seg_map:.4f}\n")

# 训练yolov8n-seg.pt模型
def train_yolov8n_segmentation(data_yaml_path, epochs=150, imgsz=640):
    """
    训练YOLOv8n分割模型
    
    参数:
        data_yaml_path: data.yaml文件路径
        epochs: 训练轮次
        imgsz: 图像尺寸(数据集已经将图像resize到640x640)
    """
    
    # 创建验证结果保存目录
    val_results_dir = './val_segmentation_results'
    os.makedirs(val_results_dir, exist_ok=True)
    
    # 从本地加载yolov8n-seg.pt模型，用中间模型进行训练提高准确性
    local_model_path = './runs/segment/train3/weights/epoch40.pt'
    model = YOLO(local_model_path)

    # 训练模型
    results = model.train(
        data = data_yaml_path,      # 数据配置文件路径
        epochs = epochs,            # 训练轮次
        task = 'segment',           # 任务类型为分割
        imgsz = imgsz,              # 图像尺寸
        batch = 16,                 # 批处理大小
        device = 'cpu',             # 使用CPU训练
        pretrained = True,          # 从预训练模型开始训练
        seed = 42,                  # 随机种子
        patience = 30,              # 提前停止
        workers = 4,                # 数据加载线程数
        save = True,                # 保存检查点
        save_period = 10,           # 每10轮保存一次检查点
        val = True,                 # 开启验证集评估
        # 数据增强配置
        augment = True,             # 开启数据增强
        hsv_h = 0.015,              # 色调偏移
        hsv_s = 0.7,                # 饱和度偏移
        hsv_v = 0.4,                # 亮度偏移
        translate = 0.1,            # 平移范围
        scale = 0.5,                # 缩放范围
        flipud = 0.5,               # 上下翻转概率
        fliplr = 0.5,               # 左右翻转概率
        mosaic = 0.5,               # 马赛克增强概率
        mixup = 0.1,                # mixup增强概率
    )
    return results, model

# 训练yolov8n.pt模型
def train_yolov8n_detection(data_yaml_path, epochs=100, imgsz=640):
    """
    训练YOLOv8n灰尘水渍检测模型
    
    参数:
        data_yaml_path: data.yaml文件路径
        epochs: 训练轮次
        imgsz: 图像尺寸(数据集已经将图像resize到640x640)
    """
    yolo_model_path = './runs/detect/train/weights/epoch10.pt'
    # 加载预训练的yolov8n.pt模型
    model = YOLO(yolo_model_path)
     # 训练模型
    results = model.train(
        data = data_yaml_path,      # 数据配置文件路径
        epochs = epochs,            # 训练轮次
        task = 'detect',           # 任务类型为检测
        imgsz = imgsz,              # 图像尺寸
        batch = 16,                 # 批处理大小
        device = 'cpu',             # 使用CPU训练
        pretrained = True,          # 从预训练模型开始训练
        seed = 42,                  # 随机种子
        patience = 30,              # 提前停止
        workers = 4,                # 数据加载线程数
        save = True,                # 保存检查点
        save_period = 10,           # 每10轮保存一次检查点
        val = True,                 # 开启验证集评估
        # 数据增强配置
        augment = True,             # 开启数据增强
        hsv_h = 0.015,              # 色调偏移
        hsv_s = 0.7,                # 饱和度偏移
        hsv_v = 0.4,                # 亮度偏移
        translate = 0.1,            # 平移范围
        scale = 0.5,                # 缩放范围
        flipud = 0.5,               # 上下翻转概率
        fliplr = 0.5,               # 左右翻转概率
        mosaic = 0.5,               # 马赛克增强概率
        mixup = 0.1,                # mixup增强概率
    )
    return results, model

# 训练检测模型和分割模型
def train_model():
    """
    训练分割模型和检测模型，并在训练前后管理标签文件夹的符号链接
    """
    # 1. 设置路径和参数
    data_yaml_path = './detection_data/data.yaml'
    # segmentation_epochs = 150  # 分割模型训练轮次
    detection_epochs = 100     # 检测模型训练轮次

    # 2.1 训练分割模型
    # print("开始训练yolov8n-seg.pt模型...")
    # results, model = train_yolov8n_segmentation(data_yaml_path, epochs=segmentation_epochs)

    # 2.2 训练检测模型
    print("开始训练yolov8n.pt模型...")
    results, model = train_yolov8n_detection(data_yaml_path, epochs=detection_epochs)
   
    # 3. 保存模型
    print("开始保存yolov8n.pt模型...")
    model.save('yolo_model/yolov8n_model.pt')
    print("模型训练、评估和保存完成!")

########################### 机器学习分割模块 ###################################
def ml_segmentation_detection(image):
    """
    使用训练好的YOLOv8n-seg模型对输入图像进行划痕/凹痕的缺陷分割,使用训练好的yolov8n.pt模型进行灰尘/水渍检测

    参数:
        image: 输入图像(ndarray数组)
        
    返回:
        mask_list: 所有检测到的缺陷掩码列表，每个元素是一个二值掩码(ndarray)
        position_list: 对应掩码的位置信息列表，每个元素是[x1, y1, x2, y2]格式的边界框坐标
    """
    # 加载训练好的模型
    seg_model = YOLO('yolo_model/yolov8n_segmentation_model.pt')
    det_model = YOLO('yolo_model/yolov8n_model.pt')
    
    mask_list = []
    position_list = []
    
    # 第一阶段：对输入图像进行划痕/凹痕的缺陷分割
    seg_results = seg_model(image)
    
    # 处理分割结果
    for result in seg_results:
        if hasattr(result, 'masks') and result.masks is not None:
            # 获取对应边界框
            boxes = result.boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class_id]
            
            masks = result.masks
            h, w = image.shape[:2]
            
            # 首先尝试从多边形表示(xy)提取掩码，与general_function.py保持一致
            try:
                polys = masks.xy
                if polys is not None and len(polys) > 0:
                    for i, poly in enumerate(polys):
                        mask = np.zeros((h, w), dtype=np.uint8)
                        pts = np.array(poly, dtype=np.int32)
                        if pts.shape[0] >= 3:
                            cv2.fillPoly(mask, [pts], 1)
                            
                            # 确保掩码格式一致
                            mask = (mask > 0).astype(np.uint8)
                            
                            mask_list.append(mask)
                            if i < len(boxes):
                                position_list.append(boxes[i][:4])
                    continue  # 如果成功从多边形提取，跳过后续方法
            except Exception:
                pass
            
            # 如果多边形提取失败，尝试从掩码数据张量(n, h, w)提取
            try:
                mask_data = masks.data.cpu().numpy()  # 转换为numpy数组
                
                # 遍历每个掩码和对应的边界框
                for i in range(len(mask_data)):
                    mask = mask_data[i]  # 单个掩码
                    
                    # 将掩码缩放到原始图像尺寸
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # 转换为与general_function.py中pred_mask一致的格式：uint8类型，值为0或1
                    mask_resized = (mask_resized > 0).astype(np.uint8)
                    
                    mask_list.append(mask_resized)
                    if i < len(boxes):
                        position_list.append(boxes[i][:4])
            except Exception as e:
                print(f"提取掩码时出错: {e}")
    
    # 第二阶段：对分割结果进行灰尘/水渍检测
    det_results = det_model(image)
    
    # 处理检测结果（没有掩码，需要从边界框生成矩形掩码）
    for result in det_results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class_id]
            
            for box in boxes:
                x1, y1, x2, y2 = box[:4].astype(int)
                
                # 生成矩形掩码
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1
                # 确保掩码格式与分割结果一致
                mask = (mask > 0).astype(np.uint8)
                
                mask_list.append(mask)
                position_list.append(box[:4])
    
    return mask_list, position_list

# 测试验证集的识别效果
def test_ml_validation_set(task='segmentation', vislist=None):
    """
    测试数据验证集的识别效果

    参数:
        task: 任务类型，'segmentation'或'detection'
        vislist: 可视化结果列表，用于合并矩形框检测
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    # 第一阶段：处理划痕与凹痕的缺陷分割
    data_root = os.path.join(repo_root, task + '_data', 'valid')
    images_dir = os.path.join(data_root, 'images')
    labels_dir = os.path.join(data_root, 'labels')
    
    out_dir = os.path.join(repo_root, 'segmentation_results', 'val_yolo_outputs', task + '_visualizations')
    os.makedirs(out_dir, exist_ok=True)
    iou_csv = os.path.join(repo_root, 'segmentation_results', 'val_yolo_outputs', task + '_ious.csv')

    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = sorted(
        [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith(image_extensions)
        ]
    )

    # 分割任务：使用yolov8n-seg模型进行缺陷分割
    if task == 'segmentation':
        model_path = os.path.join(repo_root, 'yolo_model', 'yolov8n_segmentation_model.pt')
        model = YOLO(model_path)
    elif task == 'detection':
        model_path = os.path.join(repo_root, 'yolo_model', 'yolov8n_model.pt')
        model = YOLO(model_path)
    
    vis_temp = []      # 用于存储可视化结果的临时列表，用于后续矩形框检测合并

    with open(iou_csv, 'w', encoding='utf-8') as outf:
        outf.write('image,iou\n')
        for i,img_name in enumerate(tqdm(image_files, desc='Images')):
            img_path = os.path.join(images_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)

            img = cv2.imread(img_path)
            if img is None:
                print('Could not read', img_path)
                continue
            h, w = img.shape[:2]

            # 读取真实掩码
            if task == 'segmentation':
                gt_mask = gf.read_label_polygons(label_path, w, h)
            elif task == 'detection':
                gt_mask = gf.read_label_bboxes(label_path, w, h)

            # run model prediction for this image
            try:
                results = model.predict(img_path, conf=0.25, verbose=False)
            except Exception as e:
                print('Model prediction failed for', img_name, e)
                pred_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                if len(results) == 0:
                    pred_mask = np.zeros((h, w), dtype=np.uint8)
                else:
                    if task == 'segmentation':
                        pred_mask = gf.masks_from_result(results[0], h, w)
                    elif task == 'detection':
                        pred_mask = gf.masks_from_result_boxes(results[0], h, w)

            iou = gf.compute_iou(gt_mask, pred_mask)
            outf.write(f"{img_name},{iou:.6f}\n")

            if vislist is None:
                vis = gf.draw_contours_on_image(img, pred_mask, gt_mask=gt_mask)
                out_path = os.path.join(out_dir, img_name)
                cv2.imwrite(out_path, vis)
                vis_temp.append(vis.copy())
            else:
                vis = gf.draw_contours_on_image(vislist[i], pred_mask, gt_mask=gt_mask)
                out_path = os.path.join(out_dir, img_name)
                cv2.imwrite(out_path, vis)
    
    if task == 'segmentation':
        return vis_temp
    else:
        return None

if __name__ == '__main__':
    # 第一阶段进行划痕/凹痕的缺陷分割
    vis_temp = test_ml_validation_set(task='segmentation')
    # 第二阶段进行灰尘/水渍检测
    test_ml_validation_set(task='detection', vislist=vis_temp)
    # 第三阶段划痕/凹痕缺陷分割的整体指标
    # model = YOLO('yolo_model/yolov8n_segmentation_model.pt')
    # yaml_path = os.path.join('segmentation_data', 'data.yaml')
    # evaluate_model(model, yaml_path)
