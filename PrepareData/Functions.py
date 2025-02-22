import os.path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline
from config import Config

class PrepareTrainingData:
    def __init__(self):
        pass

    # 函数名：
    # 功能：将若干numpy数组，转换成类似MNIST数据集的格式，并存储到文件中
    #       1，若干numpy数组，每个数组代表一幅图片的数据
    #       2，遍历这些数组，将所有数据相连接成一个大数组
    #       3，把这个大数组存储到一个文件中
    # 参数：若干numpy数组，每个数组代表一幅图片的数据
    # 返回：文件名
    @staticmethod
    def save_data_to_file(imgs_array, output_file_name):
        # 验证输入
        if not isinstance(imgs_array, (list, np.ndarray)):
            raise TypeError("输入必须是列表或numpy数组")
        
        try:
            data = np.concatenate(imgs_array, axis=0)
            np.save(output_file_name, data)
            return True
        except Exception as e:
            raise ValueError("输入数组必须是图像数据")

    # 函数名：stock_ma_to_image
    # 功能：将股票20日均线数据转换为类似MNIST的64*64灰度图像
    # 参数：ma_data : 输入的20日均线数据数组
    #       target_size : 目标图像尺寸，默认(64, 64)
    # 返回：PIL.Image对象
    @staticmethod
    def stock_ma_to_image(ma_data, target_size=(64, 64)):
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

    # 函数名：random_jitter
    # 功能：输入一个数组，复制n份，并对每份新复制的数组每个值，随机上下浮动3%，精度保留2位小数
    # 用途：用于数据增强，增加数据数量，提高模型泛化能力
    # 参数：array - 输入数组
    #       n - 复制份数 默认为20份
    # 返回：jittered - 复制后的数组列表
    @staticmethod
    def random_jitter(array, copy_number=20):
        jittered = []
        for _ in range(copy_number):
            jittered.append(np.round(array * (1 + np.random.uniform(-0.03, 0.03)), 2))
        return jittered
    

# 读取数据文件，并计算指定均线，返回K线数据和均线数据
# 返回值：DataFrame对象，包含K线数据（日期、收盘价、成交量）和均线数据
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
    df = df[['日期', '收盘', '成交量']].copy()
    
    # 计算ma_period日均线
    for period in ma_period:
        df['ma' + str(period)] = df['收盘'].rolling(window=period).mean()

    return df

# 处理单个日期的成交量数据
def prepare_one_short_term_data(code, dt, ma_period=[10, 20, 50, 100]):

    data_path = os.path.join(Config.PRICE_DATA_PATH, f"{Config.stock_file_path(code)}.txt")

    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return
    
    df = read_KData_and_make_ma(data_path, dt, ma_period=ma_period)
    if df is None:
        print(f"数据文件无效: {data_path}")

    imgs = []
    # 生成图片
    for period in ma_period:
        # 转换成array
        ma_data = df['ma' + str(period)].values[:period]
        img = PrepareTrainingData.stock_ma_to_image(ma_data, target_size=(64, 64))
        imgs.append(img)

    return imgs

# 处理单个日期的成交量数据
def prepare_one_ma_data(code, dt, ma_period=[10, 20, 50, 100]):

    data_path = os.path.join(Config.PRICE_DATA_PATH, f"{Config.stock_file_path(code)}.txt")

    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return
    
    df = read_KData_and_make_ma(data_path, ma_period=ma_period)
    if df is None:
        print(f"数据文件无效: {data_path}")


    df = df[df['日期'] <= dt]

    imgs = []
    # 生成图片
    for period in ma_period:
        # 转换成array
        ma_data = df['ma' + str(period)].values[:period]
        img = PrepareTrainingData.stock_ma_to_image(ma_data, target_size=(64, 64))
        imgs.append(img)

    return imgs

def PrepareMaTrainingData(folder_path, output_folder):
    # 遍历folder_path下所有json文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取json文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                for item in data['traning_data']:
                    # 处理每个item
                    name = item['name']
                    code = item['code']
                    dts = item['date']
                    for dt in dts:
                        handle_data(code, dt)
            

def PrepareShortTermTrainingData(folder_path, output_folder):

    output_data = []
    label_data = []

    # 遍历folder_path下所有json文件
    for filename in os.listdir(folder_path):
        label = 1
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取json文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                for item in data['traning_data']:
                    # 处理每个item
                    name = item['name']
                    code = item['code']
                    dts = item['date']
                    for dt in dts:
                        imgs, labels = prepare_one_short_term_data(code, dt)
                        output_data.extend(imgs)
                        label_data.extend(labels)

    #data = np.concatenate(imgs, axis=0)
    data_file_path = os.path.join(output_folder, 'short_term_data.dat')
    np.save(data_file_path, data)

    label_file_path = os.path.join(output_folder, 'short_term_label.dat')
    np.save(label_file_path, label_data)

def Main():
    try:
        # 使用相对路径并添加文件存在检查
        path = r"C:/new_tdx/T0002/export_20230527_后复权/SH#600017.txt"
        df = read_KData_and_make_ma(path)
        print("成功加载数据文件:")
        print(df[:100])  

        img = stock_ma_to_image(df['ma20'].values[:200], target_size=(64, 64))
        #保存图片，如果文件存在则覆盖
        img.save("Test/smart_scaled_2.png")
        
    except FileNotFoundError:
        print("数据文件未找到，使用测试数据...")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    Main()