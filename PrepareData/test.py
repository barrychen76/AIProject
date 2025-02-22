import numpy as np
from PIL import Image


def smart_scale_to_image(array, target_size=(64, 64)):
    """
    智能缩放一维数组到指定尺寸的灰度图像
    特征：
    - 自动保持原始数据比例
    - 智能居中填充
    - 自适应亮度范围
    """
    data = np.array(array)
    if len(data) < 2:
        return Image.new('L', target_size, 0)

    # 计算原始数据特征
    min_val, max_val = data.min(), data.max()
    data_range = max_val - min_val if max_val != min_val else 1
    data_length = len(data)

    # 计算原始宽高比（假设x轴为数据索引，y轴为数据值）
    original_aspect = data_range / data_length

    # 确定目标画布尺寸
    target_width, target_height = target_size
    target_aspect = target_width / target_height

    # 计算缩放尺寸（保持原始比例）
    if original_aspect > target_aspect:
        scaled_width = target_width
        scaled_height = int(target_width / original_aspect)
    else:
        scaled_height = target_height
        scaled_width = int(target_height * original_aspect)

    # 增加插值密度并使用三次样条插值
    from scipy.interpolate import UnivariateSpline
    x_original = np.linspace(0, 1, data_length)
    x_scaled = np.linspace(0, 1, scaled_width * 4)  # 4倍采样点
    spline = UnivariateSpline(x_original, data, s=0)
    interpolated = spline(x_scaled)

    # 计算Y轴坐标（翻转坐标系）
    y_normalized = (interpolated - min_val) / data_range
    y_positions = ((1 - y_normalized) * (scaled_height - 1)).astype(int)

    # 使用线段连接绘制（带抗锯齿）
    from PIL import ImageDraw
    scaled_img = Image.new('L', (scaled_width, scaled_height), 0)
    draw = ImageDraw.Draw(scaled_img)
    
    # 转换坐标为实际像素位置
    x_coords = np.linspace(0, scaled_width-1, len(x_scaled))
    points = list(zip(x_coords, y_positions))
    
    # 绘制渐变宽度线段
    prev_point = points[0]
    for point in points[1:]:
        # 主线段（宽度2抗锯齿）
        draw.line([prev_point, point], fill=255, width=2)
        # 辅助渐变（宽度1半透明）
        draw.line([prev_point, point], fill=200, width=1)
        prev_point = point
    
    scaled_img = np.array(scaled_img)  # 转换回numpy数组

    # 创建最终画布并居中
    canvas = np.zeros(target_size, dtype=np.uint8)
    pad_left = (target_width - scaled_width) // 2
    pad_top = (target_height - scaled_height) // 2

    # 边界保护
    valid_width = min(scaled_width, target_width - pad_left)
    valid_height = min(scaled_height, target_height - pad_top)

    canvas[pad_top:pad_top + valid_height,
    pad_left:pad_left + valid_width] = scaled_img[:valid_height, :valid_width]

    return Image.fromarray(canvas, 'L')

def load_data(file_path):
    """
    从文件中加载数据
    :param file_path: 数据文件路径
    :return: 包含数据的列表
    """
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]
    return data

'''
# 使用示例
if __name__ == "__main__":
    # 生成测试数据（正弦波 + 噪声）
    x = np.linspace(0, 4 * np.pi, 100)
    test_data = (np.sin(x) * 50 + 100).tolist()

    # 生成图像
    img = smart_scale_to_image(test_data)

    # 保存结果
    img.save("smart_scaled.png")
    img.show()  # 显示图像（需要GUI支持）

    # 从文件加载数据
    #file_path = 'data.txt'  # 假设数据文件名为 data.txt
    #loaded_data = load_data(file_path)

    # 使用加载的数据生成图像
    #loaded_img = smart_scale_to_image(loaded_data)
    #loaded_img.save("loaded_smart_scaled.png")
    #loaded_img.show()  # 显示图像（需要GUI支持）
'''




import os
import json

def count_dates_in_json(directory):
    """
    统计目录下所有 JSON 文件中的日期数量。
    
    参数:
    directory: 目录路径，包含 JSON 文件。
    
    返回:
    count: 日期总数。
    """

    files_date_count = 0  # 初始化文件计数器

    # 遍历目录下的文件
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # 只处理 JSON 文件
            file_path = os.path.join(directory, filename)

            # 读取 JSON 文件
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                date_count = 0  # 初始化日期计数器

                # 遍历 JSON 数据中的日期
                for item in data.get("traning_data", []):
                    date_count += len(item.get("date", []))  # 累加日期数量

                files_date_count += date_count
                print(f"文件 {filename} 中的日期数量为: {date_count}")

    return files_date_count

# 示例用法
directory = "./Data/short_term"  # 替换为您的目录路径
file_date_count = count_dates_in_json(directory)
print(f"目录 {directory} 中的日期总数为: {file_date_count}")
