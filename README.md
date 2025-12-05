# 汽车缺陷检测系统

## 项目概述

本项目是一个基于计算机视觉和机器学习技术的汽车缺陷检测系统，主要用于自动检测汽车表面的四种缺陷类型：划痕（scratch）、凹痕（runs）、污垢（dirt）和水渍（water marks）。系统采用了先进的图像增强、深度学习分割和传统机器学习分类相结合的技术方案，实现了高效、准确的缺陷检测与分类。

## 技术栈

- **编程语言**: Python 3.8+
- **核心库**:
  - OpenCV: 图像处理与计算机视觉
  - NumPy: 数值计算
  - PIL/Pillow: 图像读取与保存
  - Scikit-learn: 机器学习算法
  - Ultralytics YOLOv8: 深度学习目标检测与分割
  - Albumentations: 图像增强
  - Matplotlib: 数据可视化

## 目录结构

```
car_defects_detect/
├── data/                    # 数据集目录
│   ├── train/              # 训练集
│   ├── valid/              # 验证集
│   └── test/               # 测试集
├── yolo_model/              # YOLOv8模型文件
├── enhance_results/        # 图像增强结果
├── segmentation_results/   # 缺陷分割结果
├── binary_results/         # 二值化结果
├── extracted/              # 特征提取结果
├── test_final_results/     # 最终测试结果
│   ├── pictures/           # 可视化结果图像
│   └── classification_results.txt # 分类结果文本
├── main.py                 # 主程序入口
├── regional_segmentation.py # 缺陷分割与检测模块
├── feature_extraction.py   # 特征提取与分类模块
├── image_enhancement.py    # 图像增强模块
├── general_function.py     # 通用函数模块
├── histogram_equalization.py # 直方图均衡化模块
└── README.md               # 项目说明文档
```

## 功能模块

### 1. 图像增强

使用CLAHE（Contrast Limited Adaptive Histogram Equalization）方法对输入图像进行增强，提高缺陷区域与背景的对比度，为后续的缺陷分割提供更好的图像质量。

**实现文件**: `image_enhancement.py`

### 2. 缺陷分割与检测

采用YOLOv8模型进行缺陷分割和检测，包括两个核心模型：
- **分割模型（YOLOv8n-seg）**: 用于分割划痕和凹痕等缺陷区域，生成缺陷掩码
- **检测模型（YOLOv8n）**: 用于检测污垢和水渍等表面缺陷

**实现文件**: `regional_segmentation.py`

### 3. 特征提取

使用Gabor滤波器组提取缺陷区域的纹理特征，包含12个方向和5个尺度的滤波器，能够有效捕获不同类型缺陷的纹理信息。

**实现文件**: `feature_extraction.py`

### 4. 缺陷分类

使用SVM（Support Vector Machine）分类器对提取的特征进行分类，识别缺陷类型。系统还采用PCA（Principal Component Analysis）进行特征降维，提高分类效率。

**实现文件**: `feature_extraction.py`

### 5. 结果可视化与评估

将检测到的缺陷在原始图像上标记出来，并计算分类准确率。系统会保存可视化结果和分类报告。

**实现文件**: `general_function.py` 和 `main.py`

## 安装说明

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd car_defects_detect
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
   (注：如果没有requirements.txt文件，请手动安装上述核心库)

3. **下载YOLOv8模型**
   - YOLOv8n-seg.pt (分割模型)
   - YOLOv8n.pt (检测模型)
   将下载的模型文件放入 `yolo_model/` 目录

## 使用方法

### 1. 准备测试数据

将需要测试的汽车表面图像放入 `data/test/images/` 目录，并确保对应的标签文件（如果有）在 `data/test/labels/` 目录中。

### 2. 运行测试

执行主程序：

```bash
python main.py
```

### 3. 查看结果

- **可视化结果**: 保存于 `test_final_results/pictures/` 目录，显示原始图像及检测到的缺陷标记
- **分类报告**: 保存于 `test_final_results/classification_results.txt` 文件，包含每张图像的缺陷数量、分类准确率等信息

## 缺陷类型说明

系统能够检测和分类四种汽车表面缺陷：

| 类别ID | 缺陷类型 | 描述 |
|-------|----------|------|
| 0     | dirt     | 汽车表面的灰尘或污垢 |
| 1     | runs     | 汽车表面的凹痕或压痕 |
| 2     | scratch  | 汽车表面的划痕 |
| 3     | water marks | 汽车表面的水渍 |

## 工作流程

1. **图像输入**: 读取汽车表面图像
2. **图像增强**: 应用CLAHE增强图像对比度
3. **缺陷分割**: 使用YOLOv8分割模型提取缺陷区域掩码
4. **缺陷检测**: 使用YOLOv8检测模型定位缺陷位置
5. **特征提取**: 对缺陷区域提取Gabor纹理特征
6. **特征降维**: 使用PCA对特征进行降维
7. **缺陷分类**: 使用SVM分类器识别缺陷类型
8. **结果输出**: 保存可视化结果和分类报告

## 性能评估

系统性能主要通过分类准确率进行评估，平均准确率约为53%（基于当前测试集）。准确率计算采用了考虑向量相似性的方法，能够更准确地评估多缺陷场景下的分类性能。

## 未来改进方向

1. **模型优化**: 进一步优化YOLOv8模型，提高缺陷分割和检测的准确率
2. **特征工程**: 探索更多类型的特征提取方法，如深度特征
3. **分类器改进**: 尝试使用更先进的分类算法，如深度学习分类器
4. **实时检测**: 优化算法，实现实时缺陷检测
5. **多光源适应**: 增强系统对不同光照条件的适应性

## 许可证

本项目仅供研究和学习使用。

## 联系方式

如有问题或建议，请联系项目开发团队。