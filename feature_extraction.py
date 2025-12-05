import cv2
import os
import pickle
import numpy as np
from skimage import filters as filters_module
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import general_function as gf

# 从YOLO标签中提取缺陷区域图像块及类别（多边形轮廓掩码方面泛化成矩形进行处理）
def extract_defect_patches(image_path, label_path):
    """
    从原始图像和YOLO标签中提取缺陷区域图像块及类别。
    Args:
        image_path: 原始图像路径。
        label_path: 对应的YOLO标签文件路径(.txt).
    Returns:
        patches: 提取的缺陷图像块列表。
        labels: 对应的类别ID列表。
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    h, w = image.shape[:2]
    patches = []
    labels = []
    
    # 读取标签文件
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"标签文件不存在: {label_path}")
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 解析标签行
        parts = list(map(float, line.split()))
        class_id = int(parts[0])
        
        # 提取坐标信息
        coords = parts[1:]
        
        if len(coords) == 4:
            # 目标检测格式: class_id x_center y_center width height
            x_center, y_center, width, height = coords
            
            # 转换为像素坐标
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            
            # 计算边界框
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # 确保边界框在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 提取图像块
            patch = image[y1:y2, x1:x2]
            patches.append(patch)
            labels.append(class_id)
            
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # 实例分割格式: class_id x1 y1 x2 y2 ... xn yn
            # 转换为像素坐标
            polygon = np.array(coords, dtype=np.float32).reshape(-1, 2)
            polygon[:, 0] *= w
            polygon[:, 1] *= h
            polygon = polygon.astype(np.int32)
            
            # 创建掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            
            # 计算边界框（包含所有点的最小矩形）
            x1, y1 = np.min(polygon, axis=0)
            x2, y2 = np.max(polygon, axis=0)
            
            # 确保边界框在图像范围内
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            
            # 提取图像块和对应的掩码区域
            patch = image[y1:y2, x1:x2]
            patch_mask = mask[y1:y2, x1:x2]
            
            # 将掩码应用到图像块
            patch = cv2.bitwise_and(patch, patch, mask=patch_mask)
            
            patches.append(patch)
            labels.append(class_id)
    
    return patches, labels

# 提取图像块并保存到文件夹中做中间产物（作为一种验证）
def save_defect_patches():
    """
    从原始图像和YOLO标签中提取缺陷区域图像块并保存到指定文件夹
    """
    # 定义保存路径
    save_dir = './extracted/patches'
    os.makedirs(save_dir, exist_ok=True)

    # 遍历训练集文件夹
    train_dir = './data/train'
    images_dir = os.path.join(train_dir, 'images')
    labels_dir = os.path.join(train_dir, 'labels')
    for image,label in zip(os.listdir(images_dir),os.listdir(labels_dir)):
        image_path = os.path.join(images_dir, image)
        label_path = os.path.join(labels_dir, label.replace('.jpg', '.txt'))
        patches, labels = extract_defect_patches(image_path, label_path)
        # 保存图像块
        for i, (patch, label) in enumerate(zip(patches, labels)):
            patch_path = os.path.join(save_dir, f"{image.split('.')[0]}_patch_{i}_{label}.jpg")
            cv2.imwrite(patch_path, patch)

# Gabor滤波器组实现
class GaborFilterBank:
    """Gabor滤波器组 - 用于提取多尺度多方向的纹理特征"""
    
    def __init__(self, 
                 num_orientations: int = 8,
                 num_scales: int = 5,
                 kernel_size: int = 31,
                 frequencies: List[float] = None,
                 sigma_x: float = 3.0,
                 sigma_y: float = 3.0):
        """
        初始化Gabor滤波器组
        
        参数:
        num_orientations: 方向数量 (0到π之间均匀分布)
        num_scales: 尺度/频率数量
        kernel_size: 滤波器核大小 (奇数)
        frequencies: 频率列表，如果不指定则使用默认值
        sigma_x: x方向的标准差
        sigma_y: y方向的标准差
        """
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        # 保持kernel_size为奇数
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # 设置频率范围（从低频到高频）
        if frequencies is None:
            self.frequencies = [0.1, 0.2, 0.3, 0.4, 0.5][:num_scales]
        else:
            self.frequencies = frequencies[:num_scales]
            
        # 生成所有可能的组合
        self.orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)
        
        # 创建滤波器组
        self.filters = self._create_gabor_filter_bank()
        
        print(f"已创建Gabor滤波器组: {len(self.filters)}个滤波器")
        print(f"方向: {num_orientations}个 (0到π)")
        print(f"尺度: {num_scales}个 (频率: {self.frequencies})")
    
    def _create_gabor_filter_bank(self) -> List[np.ndarray]:
        """
        创建Gabor滤波器组
        
        返回:
        filters: Gabor滤波器列表
        """
        filters = []
        
        # 遍历所有方向和频率组合
        for theta in self.orientations:
            for frequency in self.frequencies:
                # 创建Gabor核
                kernel = cv2.getGaborKernel(
                    (self.kernel_size, self.kernel_size),
                    sigma=3.0,          # 标准差，控制Gabor函数的宽度，值越大，滤波器响应越宽,同时应用于x和y方向
                    theta=theta,     # Gabor 滤波器的主要方向，以弧度为单位，当 theta=0 时，滤波器对水平方向的纹理敏感；当 theta=π/2 时，对垂直方向的纹理敏感
                    lambd=1.0 / frequency,  # 波长 = 1/频率
                    gamma=0.5,  # 纵横比,控制Gabor函数的椭圆率，调整滤波器在 x 和 y 方向上的扩展比例，当gamma=1时为圆形
                    psi=0,  # 相位偏移,Gabor 函数中余弦/正弦波的相位偏移,控制滤波器对边缘的响应类型（亮边或暗边）, psi=0 或 psi=π 时，滤波器对边缘对称响应
                    ktype=cv2.CV_32F  # 32位浮点型
                )
                
                # 归一化滤波器（避免过大的响应值）
                kernel = kernel / (1.5 * np.sum(np.abs(kernel)))
                
                filters.append(kernel)
        
        return filters
    
    def visualize_filters(self, save_path: Optional[str] = None):
        """
        可视化Gabor滤波器组
        
        参数:
        save_path: 保存图像的路径（可选）
        """
        fig, axes = plt.subplots(self.num_scales, self.num_orientations, 
                                figsize=(self.num_orientations * 2, 
                                        self.num_scales * 2))
        
        if self.num_scales == 1:
            axes = axes.reshape(1, -1)
        elif self.num_orientations == 1:
            axes = axes.reshape(-1, 1)
        
        for i, kernel in enumerate(self.filters):
            scale_idx = i // self.num_orientations
            orientation_idx = i % self.num_orientations
            
            ax = axes[scale_idx, orientation_idx]
            ax.imshow(kernel, cmap='gray')  
            ax.set_title(f'θ={self.orientations[orientation_idx]:.2f}\n'
                        f'f={self.frequencies[scale_idx]:.2f}', 
                        fontsize=8)
            ax.axis('off')
        
        plt.suptitle('Gabor Filter Bank', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"滤波器可视化已保存到: {save_path}")
        
        plt.show()
    
    def apply_filters(self, image: np.ndarray) -> List[np.ndarray]:
        """
        对图像应用所有Gabor滤波器
        
        参数:
        image: 输入图像（灰度图）
        
        返回:
        responses: 滤波响应列表
        """
        responses = []
        
        for kernel in self.filters:
            # 应用滤波器
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(filtered)
        
        return responses
    
    def get_filter_params(self) -> List[Dict]:
        """
        获取每个滤波器的参数信息
        
        返回:
        params: 滤波器参数列表
        """
        params = []
        
        for i, kernel in enumerate(self.filters):
            scale_idx = i // self.num_orientations
            orientation_idx = i % self.num_orientations
            
            params.append({
                'index': i,
                'scale': scale_idx,
                'orientation': orientation_idx,
                'frequency': self.frequencies[scale_idx],
                'theta': self.orientations[orientation_idx],
                'kernel_shape': kernel.shape
            })
        
        return params

# 纹理特征提取类
class TextureFeatureExtractor:
    """纹理特征提取器 - 专门针对掩码区域提取Gabor纹理特征"""
    
    def __init__(self, 
                 gabor_bank: Optional[GaborFilterBank] = None,
                 use_enhanced_features: bool = True):
        """
        初始化纹理特征提取器
        
        参数:
        gabor_bank: Gabor滤波器组(如为None则创建默认)
        use_enhanced_features: 是否使用增强特征（更多统计量）
        """
        if gabor_bank is None:
            self.gabor_bank = GaborFilterBank(
                num_orientations=8,   # 八个方向
                num_scales=5,         # 五个不同频率
                kernel_size=31
            )
        else:
            self.gabor_bank = gabor_bank
            
        self.use_enhanced_features = use_enhanced_features
        
        # 特征名称（用于后续分析）
        self.feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """
        生成特征名称列表
        
        返回:
        feature_names: 特征名称列表
        """
        feature_names = []
        
        # 基础统计特征
        base_stats = ['mean', 'std', 'energy', 'entropy']         # 平均值 、标准差 、能量 、熵
        
        # 增强特征
        if self.use_enhanced_features:
            base_stats.extend(['skewness', 'kurtosis', 'max', 'min', 'median'])         # 偏度 、峰度 、最大值 、最小值 、中值
        
        # 为每个滤波器响应生成特征名
        num_filters = len(self.gabor_bank.filters)
        
        for i in range(num_filters):
            scale_idx = i // self.gabor_bank.num_orientations
            orientation_idx = i % self.gabor_bank.num_orientations
            
            for stat in base_stats:
                feature_names.append(
                    f'gabor_s{scale_idx}_o{orientation_idx}_{stat}'
                )
        
        # 添加全局特征
        feature_names.extend([
            'area_pixels',  # 掩码区域像素数
            'area_ratio',   # 掩码区域占整个图像的比例
            'aspect_ratio', # 边界框纵横比
            'solidity'     # 实心度（面积/凸包面积）
        ])
        
        return feature_names
    
    def extract_features_from_image(self, 
                                   image: np.ndarray, 
                                   mask: np.ndarray) -> np.ndarray:
        """
        从单张图像中提取纹理特征
        
        参数:
        image: 原始RGB图像
        mask: 二值掩码(0=背景,255=前景)
        
        返回:
        features: 特征向量(一维数组)
        """
        # 1. 转换为灰度图像
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # 2. 确保掩码是二值的
        if mask.max() > 1:
            _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        else:
            binary_mask = mask.copy()
        
        # 3. 只考虑掩码区域（将背景设为0）
        masked_image = gray_image * binary_mask
        
        # 4. 检查掩码区域是否有效（如果这张图里面只有灰尘这种瑕疵，那么掩码大概率是小于10的）
        mask_pixels = np.sum(binary_mask)
        '''
        if mask_pixels < 10:  # 掩码区域太小
            print(f"警告: 掩码区域太小 ({mask_pixels} 像素)")
            # 返回零向量或处理异常
            return np.zeros(len(self.feature_names))
        '''

        # 5. 应用Gabor滤波器组
        responses = self.gabor_bank.apply_filters(masked_image)
        
        # 6. 提取每个滤波器响应的统计特征
        features = []
        
        for response in responses:
            # 只考虑掩码区域的响应值
            masked_response = response[binary_mask == 1]
            
            if len(masked_response) > 0:
                # 基础统计特征
                mean_val = np.mean(masked_response)
                std_val = np.std(masked_response)
                
                # 能量（均方值）
                energy_val = np.mean(masked_response ** 2)
                
                # 熵（需要将响应值离散化）
                hist, _ = np.histogram(masked_response, bins=32)
                hist = hist / hist.sum()
                hist = hist[hist > 0]  # 移除0值
                entropy_val = -np.sum(hist * np.log2(hist))
                
                features.extend([mean_val, std_val, energy_val, entropy_val])
                
                # 增强统计特征
                if self.use_enhanced_features:
                    # 偏度（三阶矩）
                    skewness_val = np.mean(((masked_response - mean_val) / std_val) ** 3)
                    
                    # 峰度（四阶矩）
                    kurtosis_val = np.mean(((masked_response - mean_val) / std_val) ** 4) - 3
                    
                    # 最大值、最小值、中位数
                    max_val = np.max(masked_response)
                    min_val = np.min(masked_response)
                    median_val = np.median(masked_response)
                    
                    features.extend([skewness_val, kurtosis_val, 
                                    max_val, min_val, median_val])
            else:
                # 如果掩码区域没有有效响应，使用0填充
                if self.use_enhanced_features:
                    features.extend([0] * 9)  # 9个特征
                else:
                    features.extend([0] * 4)  # 4个基础特征
        
        # 7. 提取形状特征（从掩码）
        shape_features = self._extract_shape_features(binary_mask)
        features.extend(shape_features)
        
        return np.array(features)
    
    def _extract_shape_features(self, mask: np.ndarray) -> List[float]:
        """
        从掩码中提取形状特征
        
        参数:
        mask: 二值掩码
        
        返回:
        shape_features: 形状特征列表
        """
        # 确保mask是uint8类型
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return [0, 0, 0, 0, 0]
        
        # 使用最大的轮廓
        main_contour = max(contours, key=cv2.contourArea)
        
        # 1. 掩码区域像素数
        area_pixels = np.sum(mask)
        
        # 2. 掩码区域占总图像的比例
        total_pixels = mask.shape[0] * mask.shape[1]
        area_ratio = area_pixels / total_pixels if total_pixels > 0 else 0
        
        # 3. 边界框纵横比
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # 4. 实心度（面积/凸包面积）
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_pixels / hull_area if hull_area > 0 else 1
        
        return [area_pixels, area_ratio, aspect_ratio, solidity]
    
    def extract_features_batch(self, 
                              image_paths: List[str], 
                              label_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        批量提取特征
        
        参数:
        image_paths: 图像路径列表
        label_paths: YOLOv8标签文件路径列表
        
        返回:
        features_matrix: 特征矩阵 (n_samples, n_features)
        valid_samples: 成功处理的样本列表
        """
        all_features = []
        valid_samples = []
        
        print(f"开始批量提取特征，共 {len(image_paths)} 个样本，每个样本若干个掩码")
        
        for i, (img_path, label_path) in enumerate(tqdm(zip(image_paths, label_paths), 
                                                      total=len(image_paths),
                                                      desc="提取特征")):
            try:
                # 1. 加载图像
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"警告: 无法加载图像 {img_path}")
                    continue
                
                label_lines = []       # 暂存掩码信息行

                # 2. 从标签文件中读取掩码信息
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    # 过滤出包含掩码信息的行
                    label_lines = [line for line in lines if line.strip() and line.split()[0] in ['0', '1', '2', '3']]
                
                # 3. 循环处理每个掩码信息行
                img_h, img_w = image.shape[:2]         # 先高度后宽度
                for i, line in enumerate(label_lines):
                    # 3.1 解析单个掩码信息返回相应的掩码
                    mask = gf.polygon_to_mask(line, img_w, img_h)
                    # 3.2. 提取特征
                    features = self.extract_features_from_image(image, mask)
                    # 3.3. 添加到结果列表
                    all_features.append(features)
                    valid_samples.append(f"{img_path}_{i}")
                
            except Exception as e:
                print(f"错误: 处理 {img_path} 时发生异常: {str(e)}")
                continue
        
        if len(all_features) == 0:
            raise ValueError("未能成功提取任何特征")
        
        # 转换为numpy数组
        features_matrix = np.array(all_features)
        
        print(f"\n特征提取完成")
        print(f"特征矩阵形状: {features_matrix.shape}")
        print(f"特征维度: {features_matrix.shape[1]}")
        
        return features_matrix, valid_samples
    
    def extract_features_with_labels(self,
                                    images_dir: str,
                                    labels_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        从YOLOv8格式的图像和标签文件夹提取特征和标签
        
        参数:
        images_dir: 训练集图片文件夹路径
        labels_dir: YOLOv8格式的标签文件夹路径
        
        返回:
        X: 特征矩阵
        y: 标签向量
        feature_names: 特征名称
        valid_samples: 有效样本名称列表
        """
        # 验证目录存在
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        
        if not images_path.exists():
            raise ValueError(f"图像目录不存在: {images_path}")
        
        if not labels_path.exists():
            raise ValueError(f"标签目录不存在: {labels_path}")
        
        # 获取所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        label_files = []
        
        for img_file in images_path.iterdir():
            if img_file.suffix.lower() in image_extensions:
                # 查找对应的YOLOv8标签文件（将图像扩展名改为.txt）
                label_file = labels_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    image_files.append(str(img_file))
                    label_files.append(str(label_file))
                else:
                    print(f"警告: 未找到 {img_file.name} 的标签文件")
        
        print(f"找到 {len(image_files)} 个图像-标签对")
        
        # 提取特征
        X, valid_samples = self.extract_features_batch(image_files, label_files)
        
        # 提取YOLOv8格式标签
        y = []
        valid_image_names = []
        
        for sample_info in valid_samples:
            # 从valid_samples中提取图像路径和标签行索引
            # valid_samples格式: "image_path_i"
            parts = sample_info.split('_')
            line_idx = int(parts[-1])
            img_path = '_'.join(parts[:-1])
            sample_name = Path(img_path).stem
            
            label_file = labels_path / f"{sample_name}.txt"
            
            if label_file.exists():
                # 读取YOLOv8标签文件
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    # 过滤出有效的标签行（类别ID为0-3）
                    valid_label_lines = [line for line in lines if line.strip() and line.split()[0] in ['0', '1', '2', '3']]
                    
                    if line_idx < len(valid_label_lines):
                        # 获取对应的标签行
                        line = valid_label_lines[line_idx]
                        class_id = int(line.strip().split()[0])
                        y.append(class_id)
                        valid_image_names.append(f"{sample_name}_{line_idx}")
            else:
                print(f"警告: 未找到 {sample_name} 的标签文件")
                y.append(-1)
                valid_image_names.append(sample_name)
        
        y = np.array(y)
        
        # 移除未知标签的样本
        valid_indices = y != -1
        X = X[valid_indices]
        y = y[valid_indices]
        valid_image_names = [name for i, name in enumerate(valid_image_names) 
                           if valid_indices[i]]
        
        print(f"有效特征样本数: {len(X)}")
        print(f"有效标签样本数: {len(y)}")
        
        return X, y, valid_image_names
    
    def save_features(self, 
                     features: np.ndarray, 
                     labels: Optional[np.ndarray] = None,
                     feature_names: Optional[List[str]] = None,
                     sample_names: Optional[List[str]] = None,
                     output_path: str = 'texture_features.csv'):
        """
        保存提取的特征到CSV文件
        
        参数:
        features: 特征矩阵
        labels: 标签向量（可选）
        feature_names: 特征名称（可选）
        sample_names: 样本名称（可选）
        output_path: 输出文件路径
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        # 创建DataFrame
        df = pd.DataFrame(features, columns=feature_names)
        
        # 添加样本名称
        if sample_names is not None:
            df.insert(0, 'sample_name', sample_names)
        
        # 添加标签
        if labels is not None:
            df['label'] = labels
        
        # 保存到CSV
        df.to_csv(output_path, index=False)
        print(f"特征已保存到: {output_path}")
        print(f"数据形状: {df.shape}")
        
        return df
    
    def visualize_feature_distribution(self, 
                                      features: np.ndarray,
                                      labels: np.ndarray,
                                      feature_indices: List[int] = None,
                                      n_features_to_show: int = 12):
        """
        可视化特征分布
        
        参数:
        features: 特征矩阵
        labels: 标签向量
        feature_indices: 要可视化的特征索引列表
        n_features_to_show: 最多显示的特征数量
        """
        if feature_indices is None:
            # 选择前n个特征
            n_features = min(n_features_to_show, features.shape[1])
            feature_indices = list(range(n_features))
        
        n_features = len(feature_indices)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.ravel() if n_rows > 1 else [axes]
        
        unique_labels = np.unique(labels)
        
        for i, feat_idx in enumerate(feature_indices):
            if i >= len(axes):
                break
                
            ax = axes[i]
            feature_values = features[:, feat_idx]
            
            # 为每个标签绘制直方图
            for label in unique_labels:
                label_mask = labels == label
                ax.hist(feature_values[label_mask], 
                       alpha=0.5, 
                       bins=20, 
                       label=f'Class {label}',
                       density=True)
            
            ax.set_title(f'Feature {feat_idx}: {self.feature_names[feat_idx][:30]}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f'Texture Feature Distribution', fontsize=16)
        plt.tight_layout()
        plt.show()

# 使用Gabor滤波器对图像块做特征的提取（主体），主要用在数据集上
def extract_gabor_features(task: str = 'train', n_components=None):
    """
    滤波器特征提取的核心实现
    """
    # 1. 创建Gabor滤波器数组
    gabor_bank = GaborFilterBank(
        num_orientations=12,          # 12个方向
        num_scales=5,                 # 5个尺度
        kernel_size=21,               # 21x21的核大小
        frequencies=[0.1, 0.2, 0.3, 0.4, 0.5]  # 频率列表
    )

    # 2.可视化Gabor滤波器
    # gabor_bank.visualize_filters()

    # 3.创建特征提取器
    feature_extractor = TextureFeatureExtractor(
        gabor_bank=gabor_bank,
        use_enhanced_features=True
    )

    print(f"特征数量: {len(feature_extractor.feature_names)}")
    print("\n前20个特征名称:")
    for i, name in enumerate(feature_extractor.feature_names[:20]):
        print(f"  {i:3d}: {name}")
    
    # 4.批量数据集的特征提取
    # 遍历数据集文件夹
    task_dir = f'./data/{task}'
    images_dir = os.path.join(task_dir, 'images')
    labels_dir = os.path.join(task_dir, 'labels')
    # 读取所有图像和标签文件，并得到检测后的特征向量与标签向量
    X, y, valid_image_names = feature_extractor.extract_features_with_labels(
        images_dir=images_dir,
        labels_dir=labels_dir
    )
    
    # 5. 对提取到的特征做PCA降维
    if task == 'train':
        X_pca, n_components = train_PCA(X,y)
    else:
        X_pca = apply_PCA(X)  # 现在不需要传递n_components，因为会从保存的参数加载
    
    # 6. 保存降维后的特征
    # 6.1 保存PCA参数
    feature_extractor.save_features(
        features=X,
        labels=y,
        sample_names=valid_image_names,
        output_path=f'./extracted/{task}_gabor_features.csv'
    )
    # 6.3 保存降维后的特征向量与标签信息
    pca_feature_names = [f'pca_component_{i+1}' for i in range(X_pca.shape[1])]
    feature_extractor.save_features(
        features=X_pca,
        labels=y,
        sample_names=valid_image_names,
        feature_names=pca_feature_names,
        output_path=f'./extracted/{task}_gabor_pca_features.csv'
    )

    if task == 'train':
        return X_pca, y, valid_image_names, n_components
    else:
        return X_pca, y, valid_image_names

# training PCA on training set
def train_PCA(X,y):
    """
    Train PCA on training set.
    Args:
        X: Training features.
    Returns:
        pca: Trained PCA model.
    """

    # 确保输入是2D数组
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 计算累积方差并找到最小维数
    print("\n执行PCA降维...")
    pca = PCA()
    pca.fit(X_scaled)
    
    # 计算累积方差
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # 找到达到97.5%累积方差的最小维数（向上取整）
    n_components = np.argmax(cumulative_variance >= 0.975) + 1
    
    print(f"特征原始维度: {X.shape[1]}")
    print(f"达到97.5%累积方差所需最小维数: {n_components}")
    print(f"解释的总方差: {cumulative_variance[n_components-1]:.4f} (97.5%阈值)")
    
    # 执行降维
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 可视化累积方差和降维后的特征分布
    # 累积方差图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', linewidth=2)
    plt.axhline(y=0.975, color='r', linestyle='--', label='97.5% cumulative variance threshold')
    plt.axvline(x=n_components, color='g', linestyle='--', label=f'Number of Components: {n_components}')
    plt.xlabel('PCA Components')
    plt.ylabel('Cumulative Variance')
    plt.title('PCA Cumulative Variance Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./extracted/train_pca_cumulative_variance.png', dpi=300)
    plt.close()
    print(f"训练集累积方差图已保存至: ./extracted/train_pca_cumulative_variance.png")
    
    # 降维后的特征分布（前2个或3个主成分）
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Class Label')
        # 附上每个主成分的解释方差比例和名称
        plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.4f})')
        plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.4f})')
        plt.title('PCA Feature Distribution (First 2 Principal Components)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./extracted/train_pca_features_2d.png', dpi=300)
        plt.close()
        print(f"2D特征分布图已保存至: ./extracted/train_pca_features_2d.png")
    
    # 保存降维后的特征
    # 保存PCA参数
    pca_params = {
        'scaler': scaler,
        'pca': pca,
        'n_components': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cumulative_variance
    }
    with open(f'./extracted/train_pca_params.pkl', 'wb') as f:
        pickle.dump(pca_params, f)
    print(f"训练集PCA参数已保存至: ./extracted/train_pca_params.pkl")

    return X_pca, n_components

# apply PCA on test set
def apply_PCA(X, n_components=None):
    """
    Apply PCA on test set using pre-trained PCA parameters.
    Args:
        X: Test features (1D array for single sample, 2D array for multiple samples).
        n_components: Number of PCA components (optional, will use trained value if not provided).
    Returns:
        X_pca: PCA-transformed test features.
    """
    
    # 检查输入是否为空
    if X is None or len(X) == 0:
        return None
    
    # 确保输入是2D数组
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    # 加载训练好的PCA参数
    try:
        with open('./extracted/train_pca_params.pkl', 'rb') as f:
            pca_params = pickle.load(f)
        
        scaler = pca_params['scaler']
        pca = pca_params['pca']
        
        # 如果提供了n_components，则使用它，否则使用训练好的n_components
        if n_components is not None:
            pca.n_components = n_components
        
        # 标准化特征（使用训练好的scaler，不要重新拟合）
        X_scaled = scaler.transform(X)
        
        # 执行降维（使用训练好的PCA模型，不要重新拟合）
        X_pca = pca.transform(X_scaled)
        
        return X_pca
    except FileNotFoundError:
        print("错误:找不到训练好的PCA参数文件。请先运行训练集的特征提取。")
        raise
    except Exception as e:
        print(f"应用PCA时出错：{e}")
        raise

# 训练分类器并保存模型
def train_classifier(X, y, model_path='./extracted/trained_classifier.pkl'):
    """
    训练分类器并保存模型
    
    Args:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签向量 (n_samples,)
        model_path: 模型保存路径
        
    Returns:
        model: 训练好的分类器模型
        accuracy: 训练集准确率
    """
    print("\n开始训练分类器...")
    
    # 验证特征矩阵和标签向量的行数一致性
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"特征矩阵行数({X.shape[0]})与标签向量行数({y.shape[0]})不一致!")
    
    print(f"输入特征矩阵形状: {X.shape}")
    print(f"输入标签向量形状: {y.shape}")
    print(f"样本数量: {X.shape[0]}")
    
    # 使用SVM分类器，参数经过调优
    model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        random_state=42
    )
    
    # 训练模型
    model.fit(X, y)
    
    # 评估模型 - 预测结果与输入特征矩阵行数一致
    y_pred = model.predict(X)
    print(f"\n预测标签向量形状: {y_pred.shape}")
    print(f"预测标签向量: {y_pred}")
    
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\n分类器训练完成")
    print(f"训练集准确率: {accuracy:.4f}")
    print(f"分类报告:")
    print(classification_report(y, y_pred))

    # 写入分类结果文件中
    os.makedirs('./classification_results', exist_ok=True)
    with open('./classification_results/classification_report.txt', 'a') as f:
        f.write(f"Training DataSet Accuracy: {accuracy:.4f}\n")
        # 写入真实标签
        f.write(f"True labels: {y}\n")
        # 写入预测标签
        f.write(f"Predicted labels: {y_pred}\n")
        f.write(f"classification_report:\n")
        f.write(classification_report(y, y_pred))
        f.write("\n")
    
    # 保存模型
    model_path = './classification_results/trained_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n分类器模型已保存至: {model_path}")
    
    return model, accuracy

# 使用训练好的分类器进行预测
def predict_with_classifier(feature_vector, true_label=None, model_path='./classification_results/trained_classifier.pkl'):
    """
    使用训练好的分类器进行预测，支持单个或批量特征向量
    
    Args:
        feature_vector: 输入特征向量 (n_features,) 或特征向量列表 (n_samples, n_features)
        true_label: 真实标签或真实标签列表（可选，用于计算准确率和生成分类报告）
        model_path: 模型文件路径
        
    Returns:
        predicted_labels: 预测的标签或标签列表
        confidences: 预测的置信度或置信度列表（如果模型支持）
    """
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 确保输入是二维数组
    is_single_sample = len(feature_vector.shape) == 1
    if is_single_sample:
        feature_vector = feature_vector.reshape(1, -1)
    
    # 进行预测
    predicted_labels = model.predict(feature_vector)
    print(f"\n预测标签向量形状: {predicted_labels.shape}")
    print(f"预测标签向量: {predicted_labels}")
    
    # 尝试获取置信度（部分模型支持）
    confidences = []
    if hasattr(model, 'predict_proba'):
        confidences = model.predict_proba(feature_vector).max(axis=1)
    elif hasattr(model, 'decision_function'):
        # 对于没有概率输出的SVM，使用决策函数值作为置信度
        decision_values = model.decision_function(feature_vector)
        if decision_values.ndim == 2:
            # 多分类情况
            confidences = np.abs(decision_values).max(axis=1)
        else:
            # 二分类情况
            confidences = np.abs(decision_values)
    
    # 处理单个样本的情况
    if is_single_sample:
        predicted_labels = predicted_labels[0]
        confidences = confidences[0] if confidences else None
    
    # 生成分类报告（如果提供了真实标签）
    report = None
    if true_label is not None:
        # 将真实标签转换为numpy数组
        true_labels = np.array(true_label).reshape(-1)
        
        # 确保预测标签和真实标签的形状匹配
        if is_single_sample:
            predicted_labels_for_report = np.array([predicted_labels])
        else:
            predicted_labels_for_report = predicted_labels
        
        # 生成分类报告
        report = classification_report(true_labels, predicted_labels_for_report)
        
        # 计算准确率
        accuracy = accuracy_score(true_labels, predicted_labels_for_report)
        
        # 将分类报告保存到文件
        os.makedirs('./classification_results', exist_ok=True)
        with open('./classification_results/prediction_report.txt', 'a') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            # 写入真实标签的长度和全部内容
            f.write(f"True labels length: {len(true_labels)}\t")
            f.write(f"True labels: {true_labels.tolist()}\n")
            # 写入预测标签的长度和全部内容
            f.write(f"Predicted labels length: {len(predicted_labels_for_report)}\t")
            f.write(f"Predicted labels: {predicted_labels_for_report.tolist()}\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n")
        
        print(f"Prediction accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
    
    return predicted_labels, confidences

if __name__ == '__main__':
    # 提取训练集的Gabor特征，X为降维后的特征向量，y为标签，valid_image_names为图像文件名，n_components为PCA降维后的维度
    # X, y, valid_image_names, n_components = extract_gabor_features(task='train')
    # 训练分类器
    # classification_model, accuracy = train_classifier(X, y)
    # 提取测试集的Gabor特征
    n_components = 89
    X_test, y_test, valid_image_names_test = extract_gabor_features(task='valid', n_components=n_components)
    # 对测试集进行预测
    predicted_labels, confidences = predict_with_classifier(X_test, y_test)

