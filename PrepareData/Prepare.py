import os
import json
import pandas as pd
import numpy as np
from Indicators import Indicators
import Functions as func
from config import Config

class Prepare:
    # 私有变量：形态类型总数
    _kline_shape_count = 6

    def prepare_short_term_data(self):
        # short-term 数据由若干索引json文件，和对应的数据文件组成
        json_path = "./data/short_term/"
        
        # Initialize a list to store data chunks
        data_chunks = [[] for _ in range(6)]
        labels = []
        label_value = 0
        print(f"初始化数据块和标签: data_chunks={len(data_chunks)}, labels={len(labels)}")

        json_file_count = 0
        # 遍历json目录，读取所有json文件，并将数据存入self.data中
        for json_file in os.listdir(json_path):
            if json_file.endswith('.json'):
                json_file_count += 1
                
                with open(os.path.join(json_path, json_file), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    label_value = int(json_file[0])
                    print(f"读取的JSON文件: {json_file}, 标签值: {label_value}")
                    for item in data['training_data']:
                        name = item['name']
                        code = item['code']
                        dts = item['date']
                        for dt in dts:
                            df = self._prepare_one_short_term_data(code, name, dt)
                            if df is not None:
                                data_chunks[label_value].append(df)
                            else:
                                print(f"数据不足，跳过: {json_file}")

        if (json_file_count != self._kline_shape_count):
            print(f"K线形态文件数量不正确：{json_file_count} - {self._kline_shape_count}")
            return None

        # 数据读取完毕，不足的数据，需要复制一些数据
        # 每种类型，最多5000份，合计3万份测试数据
        # 其中80%用于训练，20%用于测试
        for i in range(self._kline_shape_count):
            self._augment_data(data_chunks[i], target_count=5000)

        # 随机打乱数据集，并随机上下浮动3%
        for i in range(self._kline_shape_count):
            np.random.shuffle(data_chunks[i])
            data_chunks[i] = [df * (1 + np.random.uniform(-0.03, 0.03)) for df in data_chunks[i]]

        # 将数据集转换为 MNIST 格式的数据集，并保存到文件
        df_all = pd.DataFrame()
        for i in range(self._kline_shape_count):
            df_all = pd.concat([df_all, pd.DataFrame(data_chunks[i])], ignore_index=True)
            df_all['label'] = i

        print(f"数据集形状: {df_all.shape}")
        self._create_mnist_like_dataset(df_all,
                                        ['volume', 'yin_long_ma_06', 'yang_yin06_cha_ma'],
                                        30000,
                                        'short_term_data.dat',
                                        'short_term_label.dat')

    def _augment_data(self, data_list, target_count):
        """
        增强数据集，使其达到目标数量
        :param data_list: 数据列表
        :param target_count: 目标数量
        """
        current_count = len(data_list)
        if current_count >= target_count:
            return

        copy_count = target_count - current_count
        for _ in range(copy_count):
            random_index = np.random.randint(0, current_count)
            df = data_list[random_index]
            data_list.append(df)

    # 读取一只股票，一个时间的数据
    # 返回一个DataFrame，包含如下数据：
    # 1，成交量形态数据
    # 2，三个量能数据，yin_long_ma_06、yang_yin06_cha、yang_yin06_cha_ma
    def _prepare_one_short_term_data(self, code, name, dt, ma_period=[10, 20, 50, 100]):
        # 拼出数据文件路径
        data_path = os.path.join(Config.PRICE_DATA_PATH, f"{Config.stock_file_path(code)}.txt")
        print(f"数据文件路径: {data_path}")
        if not os.path.exists(data_path):
            print(f"数据文件不存在: {data_path}")
            return None
        
        # 读取数据文件
        df = func.read_KData_and_make_ma(data_path, dt, ma_period=ma_period)
        if df is None:
            print(f"数据文件无效: {data_path}")
            return None
        print(f"数据文件读取结果: {df.shape}")
        
        # 计算阳线量能
        df = Indicators.calc_bearish_volume(df)

        # 每个数据包含如下两类数据，都要转换成int8类型：
        # 1，成交量形态数据
        # 2，三个量能数据，yin_long_ma_06、yang_yin06_cha、yang_yin06_cha_ma
        # 取60日的数据，也就是说，总数据量
        self._normalize_and_convert(df['volume'], inplace=True)
        self._normalize_and_convert(df['yin_long_ma_06'], inplace=True)
        self._normalize_and_convert(df['yang_yin06_cha'], inplace=True)
        self._normalize_and_convert(df['yang_yin06_cha_ma'], inplace=True)

        return df
        
    # 将 Pandas Series 规范化，以最后一个数据为100为基点，其余点位按比例缩放，并转换为 int8 类型
    def _normalize_and_convert(self, vol_series, inplace=False):
        # 将 Pandas Series 转换为 numpy 数组
        arr = vol_series.to_numpy(dtype=float)
        
        # 规范化数组，使最后一个值为 100
        scale_factor = 100 / arr[-1]
        normalized_arr = arr * scale_factor
        
        # 转换为 int8 类型
        int8_arr = np.int8(normalized_arr)

        # 如果 inplace 为 True，则将结果写回到 vol_series
        if inplace:
            vol_series[:] = int8_arr
            return None
        else:
            return int8_arr

    '''
    将 DataFrame 中的指定字段转换为类似 MNIST 数据集格式的文件。
    参数:
        df: 输入的 DataFrame，包含多个字段。
        fields: 需要使用的字段列表（长度为 3）。
        num_records: 测试数据的数量（例如 100）。
        data_file: 保存数据的文件名。
        label_file: 保存标签的文件名。
    '''
    def _create_mnist_like_dataset(self, df, fields, num_records, data_file, label_file):
        # 检查字段数量
        if len(fields) != 3:
            raise ValueError("字段列表必须包含 3 个字段")

        # 初始化数据和标签数组
        data = np.empty((num_records, 60, 3), dtype=np.int8)
        labels = np.empty(num_records, dtype=np.int8)

        # 填充数据和标签
        for i in range(num_records):
            # 每条数据是 60x3 的数组
            data[i] = df[fields].iloc[i * 60:(i + 1) * 60].values
            # 假设标签是数组索引（可根据需求修改）
            labels[i] = i

        # 保存数据到文件
        with open(data_file, 'wb') as f:
            np.save(f, data)

        # 保存标签到文件
        with open(label_file, 'wb') as f:
            np.save(f, labels)

if __name__ == '__main__':
    # 创建一个PrepareTrainingData对象
    Prepare().prepare_short_term_data()