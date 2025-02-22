class Indicators:
    def __init__(self):
        pass

    # 函数名：calc_bearish_volume
    # 功能：阳线量能指标
    # 参数：data - 输入数据
    # 返回：包含指标的数据DataFrame
    @staticmethod
    def calc_bearish_volume(self, data):
        n = 15
        ma_n = 5

        # 标识哪些是阳线，哪些是阴线
        data['yang_candlestick'] = (data['Close'] > data['Open']).astype(int)
        
        # 计算n日阳线和阴线的成交量总和
        data['yang_sum'] = data['value'].rolling(window=n).apply(lambda x: x[data['yang_candlestick'][x.index]==1].sum(), raw=False)
        data['yin_sum'] = data['value'].rolling(window=n).apply(lambda x: x[data['yang_candlestick'][x.index]==0].sum(), raw=False)

        # 计算n日成交量均值
        data['yang_sum'] = data['yang_sum'] / n / 10000
        data['yin_sum'] = data['yin_sum'] / n / 10000

        # 计算n日成交量均值的移动平均线
        data['yang_sum_ma'] = data['yang_sum'].ewm(span=ma_n, adjust=False).mean()
        data['yang_sum_ma_2'] = data['yang_sum_ma'].ewm(span=ma_n, adjust=False).mean()
        data['yin_sum_ma'] = data['yin_sum'].rolling(window=ma_n).mean()

        # 计算长期（100日）均线的6折线，用于计算
        data['yin_long_ma_06'] = data['yin_sum_ma'].rolling(window=100).mean() * 0.6    # 留下

        #data['aaa'] = data['yin_sum_ma'] - data['yang_sum_ma']
        #data['bbb'] = (data['yin_sum_ma'] - data['yang_sum_ma_2']).rolling(window=5).mean()

        # 计算阳线总和与阴线总和6折的差值
        # 由于经历了一段时间下跌，阳线总和势必会比阴线总和小，所以阳线只需要恢复到阴线总和的6折即可
        data['yang_yin06_cha'] = data['yang_sum_ma'] - data['yin_long_ma_06']           # 留下
        data['yang_yin06_cha_ma'] = data['yang_yin06_cha'].rolling(window=5).mean()     # 留下

        # 移除无用列
        data.drop(['yang_candlestick'], axis=1, inplace=True)
        data.drop(['yang_sum'], axis=1, inplace=True)
        data.drop(['yin_sum'], axis=1, inplace=True)
        data.drop(['yang_sum_ma'], axis=1, inplace=True)
        data.drop(['yang_sum_ma_2'], axis=1, inplace=True)
        data.drop(['yin_sum_ma'], axis=1, inplace=True)

        return data
    