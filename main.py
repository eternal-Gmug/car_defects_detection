import image_enhancement as ie
import regional_segmentation as rs
import histogram_equalization as heq
import albumentations as A
import time
import os
from PIL import Image
import numpy as np
import cv2

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
    image_path = os.path.join("pictures", image_files[6])
    image_name = os.path.basename(image_path)
    # src = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # 使用PIL对象打开图像，目的为了与后面的直方图均衡化相对应
    src = Image.open(image_path)
    
    if src is None:
        print(f"无法加载图像: {image_path}")
        return -1
    
    homo_result = None
    CLAHE_result_1 = None   # 使用albumentations的CLAHE
    CLAHE_result_2 = None   # 自主实现的CLAHE
    HeQ = heq.ImageContraster()
    img_array = np.array(src)
    
    # 创建albumentations的CLAHE变换
    # clip_limit控制对比度限制，tile_grid_size控制图像分块大小
    clahe_transform = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
    
    # 检查图像通道数
    if len(img_array.shape) == 2:
        # 灰度图像处理
        homo_result = ie.homomorphic_filter(img_array, 1.5, 0.5, 1.0, 30.0)
        # 对于灰度图，需要扩展为3通道以便albumentations处理
        img_3channel = np.expand_dims(img_array, axis=-1)
        img_3channel = np.repeat(img_3channel, 3, axis=-1)
        # 应用CLAHE
        result = clahe_transform(image=img_3channel)
        # 转回灰度图
        CLAHE_result_1 = result['image'][:,:,0]
        # 自主实现的CLAHE
        CLAHE_result_2 = HeQ.enhance_contrast(img_array, method='CLAHE')
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # 彩色图像处理（逐通道）
        # PIL读取的是RGB，需要转换为BGR供OpenCV使用
        '''
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # 使用OpenCV的split方法分离通道
        b, g, r = cv2.split(img_bgr)
        # 对每个通道应用同态滤波
        result_b = ie.homomorphic_filter(b, 1.5, 0.5, 1.0, 30.0)
        result_g = ie.homomorphic_filter(g, 1.5, 0.5, 1.0, 30.0)
        result_r = ie.homomorphic_filter(r, 1.5, 0.5, 1.0, 30.0)
        # 合并通道
        homo_result = cv2.merge([result_b, result_g, result_r])
        '''
        # 应用albumentations的CLAHE到RGB图像
        result = clahe_transform(image=img_array)
        CLAHE_result_1 = result['image']
        # 自主实现的CLAHE
        # CLAHE_result_2 = HeQ.enhance_contrast(img_array, method='CLAHE')
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # RGBA四通道图像处理
        print("处理RGBA四通道图像...")
        # 分离RGB和Alpha通道
        img_rgb = img_array[:,:,:3]
        alpha = img_array[:,:,3]
        '''
        # 对RGB通道应用同态滤波
        # 先转换为BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        b, g, r = cv2.split(img_bgr)
        b_enhanced = ie.homomorphic_filter(b, 1.5, 0.5, 1.0, 30.0)
        g_enhanced = ie.homomorphic_filter(g, 1.5, 0.5, 1.0, 30.0)
        r_enhanced = ie.homomorphic_filter(r, 1.5, 0.5, 1.0, 30.0)
        
        # 合并通道
        homo_result_rgb = cv2.merge([b_enhanced, g_enhanced, r_enhanced])
        # 转回RGB并添加Alpha通道
        homo_result_rgb_rgb = cv2.cvtColor(homo_result_rgb, cv2.COLOR_BGR2RGB)
        homo_result = np.dstack((homo_result_rgb_rgb, alpha))
        '''
        # 应用albumentations的CLAHE到RGB部分
        result = clahe_transform(image=img_rgb)
        clahe_rgb = result['image']
        # 添加回Alpha通道
        CLAHE_result_1 = np.dstack((clahe_rgb, alpha))
        # 自主实现的CLAHE
        # CLAHE_result_2 = HeQ.enhance_contrast(img_array, method='CLAHE')
    else:
        print("不支持的图像通道数")
        return -1
    
    if homo_result is not None:
        # 将OpenCV的结果转换为PIL Image对象
        homo_result = cv2.cvtColor(homo_result, cv2.COLOR_BGR2RGB)
        # homo_result = Image.fromarray(homo_result)
    
    # 评估图像质量
    # ie.evaluate_image(src, homo_result, image_name)
    # ie.evaluate_image(src, CLAHE_result_1, image_name)
    # ie.evaluate_image(src, CLAHE_result_2, image_name)

    # 保存处理前后的图像
    # ie.save_image(src, homo_result, image_path)
    # 间隔1秒
    # time.sleep(1)
    # ie.save_image(src, CLAHE_result_1, image_path)
    # 间隔1秒
    # time.sleep(1)
    # ie.save_image(src, CLAHE_result_2, image_path)

    ########################### 缺陷分割 ###################################
    # 对CLAHE_result_1进行缺陷分割
    rs.traditional_segmentation(CLAHE_result_1, image_name)

if __name__ == "__main__":
    main()
