
'''
- 读取MNIST数据集，并展示其中的一张图片，看看文件详细格式。
- 空白区域什么数据，数字区域什么数字
'''
import os
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline


def load_mnist_image(image_index, dataset="train"):
    if dataset == "train":
        dataset = train_dataset
    elif dataset == "test":
        dataset = test_dataset
    else:
        raise ValueError("dataset must be 'train' or 'test'")
    
    sample_image, sample_label = dataset[image_index]
    
    return sample_image, sample_label

def test_load_mnist_image():
    sample_index = 0
    sample_image, sample_label = load_mnist_image(sample_index, dataset="train")

    print("image length: " + len(sample_image))
    print("image label: " + len(sample_label))

    # 保存第一张图片到文件中
    plt.imsave("sample_image.png", sample_image.squeeze(), cmap='gray')

    print(f"Sample label: {sample_label}")
    plt.imshow(sample_image.squeeze(), cmap='gray')
    plt.title(f"Label: {sample_label}")
    plt.axis('off')
    plt.show()

def test_mnist():
    data_path = './data'
    transform = transforms.ToTensor()

    global train_dataset, test_dataset  # 声明为全局变量
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    test_load_mnist_image()

import numpy as np

def smooth_curve(data, smoothness=3, window_size=None):
    """
    使用高斯滤波平滑曲线
    
    参数：
    data - 输入数据数组（numpy数组）
    smoothness - 平滑度参数（1-5，越大越平滑）
    window_size - 滤波窗口大小（奇数，自动计算时约为6*smoothness + 1）
    
    返回：
    平滑后的numpy数组
    """
    # 自动计算窗口大小（确保为奇数）
    if window_size is None:
        window_size = 6 * smoothness + 1
    window_size = max(3, int(window_size) // 2 * 2 + 1)  # 保证最小为3的奇数
    
    # 根据smoothness计算标准差（经验公式）
    sigma = smoothness * 0.6
    
    # 生成高斯核
    x = np.arange(-window_size//2 + 1, window_size//2 + 1)
    kernel = np.exp(-x**2/(2*sigma**2))
    kernel = kernel / kernel.sum()
    
    # 处理边界（通过镜像填充）
    padded_data = np.pad(data, (window_size//2, window_size//2), mode='reflect')
    
    # 执行卷积
    smoothed = np.convolve(padded_data, kernel, mode='valid')
    
    return smoothed[:len(data)]  # 保证输出长度与输入一致

def simple_smooth(data, window=5):
    """移动平均法（更快速但平滑效果较弱）"""
    window = max(3, int(window) // 2 * 2 + 1)
    return np.convolve(data, np.ones(window)/window, mode='same')
def stock_ma_to_image2(ma_data, target_size=(64, 64)):
    """
    将股票20日均线数据转换为类似MNIST的64*64灰度图像
    参数：
    ma_data : 输入的20日均线数据数组
    target_size : 目标图像尺寸，默认(64, 64)
    返回：
    PIL.Image对象
    """
    # 输入验证
    if not isinstance(ma_data, (list, np.ndarray)):
        raise TypeError("输入必须是列表或numpy数组")
    
    ma_data = np.asarray(ma_data, dtype=np.float64)
    if ma_data.size == 0:
        raise ValueError("输入数组不能为空")
    
    # 数据清洗：处理NaN和无穷大
    ma_data = ma_data[~np.isnan(ma_data)]
    if ma_data.size == 0:
        raise ValueError("有效数据为空（全部为NaN）")
    
    # 所有数据不能为0
    if np.all(ma_data == 0):
        raise ValueError("所有数据不能为0")
    
    #####
    #ms_data = smooth_curve(ma_data, smoothness=5, window_size=21)
    ms_data = simple_smooth(ma_data, window=11)
    '''smoothed_data = []
    for i in range(len(ma_data)):
        if i == 0:
            smoothed_data.append(ma_data[i])
        elif i== len(ma_data) - 1:
            smoothed_data.append(ma_data[i])
        else:
            smoothed_data.append((ma_data[i-1] + ma_data[i] + ma_data[i+1]) / 3)
    ma_data = np.asarray(smoothed_data, dtype=np.float64)
    '''
    #####
    
    # 第一个数据标准化为100，后面的按比例缩放
    scaled_data = np.asarray(ma_data * 100.0 / ma_data[0], dtype=np.float64)
    
    # 计算数据特征
    data_length = len(scaled_data)
    if data_length < 2:
        raise ValueError("有效数据长度不足（至少需要2个数据点）")
    
    # 创建全零画布
    target_width, target_height = target_size
    canvas = np.zeros(target_size, dtype=np.uint8)
    
    # 计算Y轴缩放因子，使数据居中
    min_val, max_val = np.min(scaled_data), np.max(scaled_data)
    data_range = max_val - min_val
    
    # 使用三次样条插值生成更多点
    x_original = np.linspace(0, 1, data_length)
    x_interpolated = np.linspace(0, 1, data_length * 10)  # 10倍插值点
    cs = CubicSpline(x_original, scaled_data)
    interpolated_data = cs(x_interpolated)
    
    # 将数据映射到图像高度，并居中
    y_positions = (target_height - 1) - ((interpolated_data - min_val) / data_range * target_height).astype(int)
    
    # 计算X轴坐标，确保数据居中
    x_coords = np.linspace(0, target_width-1, len(x_interpolated)).astype(int)
    
    # 使用抗锯齿的线条绘制
    img = Image.new('L', target_size, 0)
    draw = ImageDraw.Draw(img)
    
    # 绘制平滑的线条
    for i in range(1, len(x_coords)):
        # 使用抗锯齿线条
        draw.line([
            (x_coords[i-1], y_positions[i-1]), 
            (x_coords[i], y_positions[i])
        ], fill=255, width=1)
    
    # 转换回numpy数组
    canvas = np.array(img)

    return Image.fromarray(canvas, 'L')

# 这个函数实现了如下功能：输入一条曲线数据，将该曲线做平滑处理，然后转换为灰度图像，最后保存到文件中。
# 函数名：stock_ma_to_image
# 输入参数：
#   curve_data：曲线数据，类型为list，每个元素为一个浮点数。
#   save_path：保存图像的路径，类型为str。
# 返回值：无
# 函数功能：
#   1. 将曲线数据做平滑处理，得到平滑后的曲线数据。
#   2. 将平滑后的曲线数据转换为灰度图像，图像的像素值范围为0-255。
#   3. 将图像保存到指定路径下。
# 函数要求：
#   1. 曲线数据的长度至少为2。
#   2. 保存的图像为灰度图像，像素值范围为0-255。
#   3. 保存的图像尺寸为64*64。
#   4. 保存的图像格式为png。
#   5. 保存的图像路径为save_path。
#   6. 保存的图像文件名为“curve.png”。
def stock_ma_to_image(curve_data, save_path, target_size=(64, 64)):
    if len(curve_data) < 2:
        raise ValueError("curve_data must have at least 2 elements")
    
    smoothed_data = []
    for i in range(len(curve_data)):
        if i == 0:
            smoothed_data.append(curve_data[i])
        elif i== len(curve_data) - 1:
            smoothed_data.append(curve_data[i])
        else:
            smoothed_data.append((curve_data[i-1] + curve_data[i] + curve_data[i+1]) / 3)
    min_val = min(smoothed_data)
    max_val = max(smoothed_data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in smoothed_data]
    # 处理可能的NaN值并确保数据有效性
    valid_data = [x for x in normalized_data if not np.isnan(x)]
    if not valid_data:
        raise ValueError("No valid data available to generate image")
    gray_values = [int(np.clip(x * 255, 0, 255)) for x in valid_data]

    # 创建一个64x64的灰度图像
    img = Image.new('L', target_size, 0)
    # 将数据点绘制到图像上，使用灰度值作为像素值
    # 调整图像生成逻辑
    height, width = target_size
    for i in range(min(len(gray_values), height*width)):
        x = i % width
        y = i // width
        img.putpixel((x, y), gray_values[i])
    # 保存图像到指定路径
    img.save(os.path.join(save_path, 'curve.png'))
    # 显示图像
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
    return img

def read_KData_and_make_ma(file_path, ma_period=[10, 20, 50, 100]):
    # 读取文件（适配逗号分隔含空格格式）
    df = pd.read_csv(file_path,
                   encoding='gbk',
                   sep=r'\s*,\s*',  # 匹配逗号及周围空格
                   header=None,
                   skiprows=3,  # 跳过前3行
                   skipfooter=1,  # 跳过最后1行
                   engine='python',  # 必须指定engine以支持skipfooter
                   names=['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额'],
                   dtype={'日期': str, 
                          '开盘': float, 
                          '最高': float, 
                          '最低': float, 
                          '收盘': float, 
                          '成交量': 'Int64', 
                          '成交额': float},
                   na_values=['--', 'NaN', 'N/A'],
                   on_bad_lines='warn')  # 跳过错误行并警告
    
    # 日期作为索引
    df.set_index('日期', inplace=True)

    # 转换数据类型
    df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')
    
    # 只取日期和收盘价
    # 检查列名是否匹配
    print("实际列名:", df.columns.tolist())
    df = df[['收盘', '成交量']].copy()  # 暂时移除日期列
    
    # 处理缺失值并计算移动平均
    df['收盘'] = df['收盘'].ffill()  # 前向填充缺失值
    for period in ma_period:
        df['ma' + str(period)] = df['收盘'].rolling(window=period, min_periods=1).mean()

    return df

if __name__ == '__main__':
    # 示例用法
    path = r"C:/new_tdx/T0002/export_20230527_后复权/SH#600017.txt"
    df = read_KData_and_make_ma(path, ma_period=[10, 20, 50, 100])
    curve_data = df['ma20'].values.tolist()  # 示例数据，可以替换为实际的曲线数据
    img = stock_ma_to_image2(curve_data[100:300], target_size=(200, 200))
    img.save(os.path.join('./Data/Temp', 'curve.png'))
    #plt.show()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    # 打印数据框的前几行以检查结果：
    print(df.head())
