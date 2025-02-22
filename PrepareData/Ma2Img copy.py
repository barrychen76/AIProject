import numpy as np
import pandas as pd
import json
import os
from PIL import Image
from scipy.interpolate import UnivariateSpline


def smart_scale_to_image(array, target_size=(64, 64)):
    """
    智能缩放一维数组到指定尺寸的灰度图像
    特征：
    - 自动保持原始数据比例
    - 智能居中填充
    - 自适应亮度范围
    """
    print(f"\n[DEBUG] 输入数据统计: len={len(array)}, min={np.min(array):.2f}, max={np.max(array):.2f}")
    
    # 规范化输入数据
    orig_data = np.array(array, dtype=np.float64)
    if orig_data.size == 0:
        raise ValueError("输入数据不能为空数组")
        
    # 将第一个数据标准化为100并按比例缩放
    if orig_data[0] == 0:
        # 自动调整初始值为最小非零值
        non_zero_values = orig_data[orig_data != 0]
        if len(non_zero_values) == 0:
            raise ValueError("数据中所有值都为零")
        scale_factor = 100 / non_zero_values[0]
    else:
        scale_factor = 100 / orig_data[0]
    data = orig_data * scale_factor
    
    if len(data) < 2:
        return Image.new('L', target_size, 0)

    # 计算原始数据特征
    min_val, max_val = data.min(), data.max()
    data_range = max_val - min_val if max_val != min_val else 1
    data_length = len(data)

    # 计算原始宽高比
    original_aspect = data_length / data_range

    # 确定目标画布尺寸
    target_width, target_height = target_size
    target_aspect = target_width / target_height

    # 计算缩放尺寸（保持原始比例）
    if original_aspect > target_aspect:
        scaled_height = target_height
        scaled_width = int(target_height * original_aspect)
    else:
        scaled_width = target_width
        scaled_height = int(target_width / original_aspect)

    # 使用三次样条插值
    x_original = np.linspace(0, 1, data_length)
    x_scaled = np.linspace(0, 1, scaled_width * 4)  # 4倍采样点
    spline = UnivariateSpline(x_original, data, s=0)
    interpolated = spline(x_scaled)

    # 计算Y轴坐标（翻转坐标系）
    y_normalized = (interpolated - min_val) / data_range
    y_positions = ((1 - y_normalized) * (scaled_height - 1)).astype(int)

    # 使用线段连接绘制
    from PIL import ImageDraw
    scaled_img = Image.new('L', (scaled_width, scaled_height), 0)
    draw = ImageDraw.Draw(scaled_img)
    
    # 转换坐标为实际像素位置
    x_coords = np.linspace(0, scaled_width-1, len(x_scaled))
    points = list(zip(x_coords, y_positions))
    
    # 绘制渐变宽度线段
    prev_point = points[0]
    for point in points[1:]:
        draw.line([prev_point, point], fill=255, width=2)
        draw.line([prev_point, point], fill=200, width=1)
        prev_point = point
    
    scaled_img = np.array(scaled_img)

    # 创建最终画布并居中
    canvas = np.zeros(target_size, dtype=np.uint8)
    pad_left = max((target_width - scaled_width) // 2, 0)
    pad_top = max((target_height - scaled_height) // 2, 0)

    # 计算实际可填充区域
    valid_width = min(scaled_width, target_width - pad_left)
    valid_height = min(scaled_height, target_height - pad_top)
    
    # 确保源图像区域不越界
    src_height = min(valid_height, scaled_img.shape[0])
    src_width = min(valid_width, scaled_img.shape[1])
    
    # 执行画布填充
    canvas[pad_top:pad_top + src_height,
           pad_left:pad_left + src_width] = scaled_img[:src_height, :src_width]

    return Image.fromarray(canvas, 'L')


def read_and_process_file(file_path, dts, ma_period=20):
    # 读取文件（适配逗号分隔含空格格式）
    df = pd.read_csv(file_path,
                   encoding='gbk',
                   sep=r'\s*,\s*',  # 匹配逗号及周围空格
                   header=None,
                   skiprows=3,  # 跳过前3行
                   skipfooter=1,  # 跳过最后1行
                   engine='python',  # 必须指定engine以支持skipfooter
                   names=['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额'],
                   dtype={'日期': str, '开盘': float, '最高': float, '最低': float, '收盘': float, '成交量': 'Int64', '成交额': float},
                   na_values=['--', 'NaN', 'N/A'],
                   on_bad_lines='warn')  # 跳过错误行并警告
    
    # 转换数据类型
    df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')
    
    # 只取日期和收盘价
    df = df[['日期', '收盘']].copy()
    
    # 计算ma_period日均线
    df['ma' + str(ma_period)] = df['收盘'].rolling(window=ma_period).mean()
    df = df[ma_period:]

    # 所有日期的数据，都读出来
    dfs = pd.DataFrame()
    for dt in dts:
        df_t = df[df['日期'] <= dt]
        df_t = df_t[len(df_t)-ma_period:]
        if len(df_t) == ma_period:
            dfs.loc[len(dfs)] = df_t

    return dfs


def load_json(json_file_path):
    """
    从JSON文件中加载数据
    :param json_file_path: JSON文件路径
    :return: 包含数据的字典
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if not data:
        raise ValueError("JSON文件为空")
    
    base_path = r"C:\new_tdx\T0002\export_20230527_后复权"

    for item in data['traning_data']:
        code = item['code']
        pre_fix = 'SH' if code.startswith('6') else 'SZ'

        file_name = pre_fix + code + '.txt'
        item['file_path'] = os.path.join(base_path, file_name)
                
    return data


# 使用示例
if __name__ == "__main__":
    try:
        data = load_json("Test/data.json")
        print("成功加载JSON文件:")

        for item in data['traning_data']:
            print(item)
            dfs = read_and_process_file(item['file_path'], item['date'])

            for i, f in enumerate(dfs):
                img = smart_scale_to_image(f['ma20'].values, target_size=(64, 64))
                img.save(f'Data/{item["code"]}-{i}.png')

        # 使用相对路径并添加文件存在检查
        path = r"C:/new_tdx/T0002/export_20230527_后复权/SH#600017.txt"
        df = read_and_process_file(path)
        print("成功加载数据文件:")
        print(df[:100])

        img = smart_scale_to_image(df['MA20'].values[:1800], target_size=(3600, 1800))
        img.save("Test/smart_scaled_2.png")
        
    except FileNotFoundError:
        print("数据文件未找到，使用测试数据...")
        # 生成测试数据（正弦波 + 噪声）
        x = np.linspace(0, 4 * np.pi, 100)
        test_data = (np.sin(x) * 50 + 100).tolist()
        img = smart_scale_to_image(test_data)
        img.save("Test/smart_scaled_demo.png")
        print("示例图像已保存到 Test/smart_scaled_demo.png")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
