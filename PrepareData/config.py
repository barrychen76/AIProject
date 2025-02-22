class Config:
    # 数据路径配置
    PRICE_DATA_PATH = r"C:\new_tdx\T0002\export_20230527_后复权"
    #JSON_BASE_PATH = "Data/"
    
    # 图像生成配置
    TARGET_SIZE = (64, 64)  # 默认图像尺寸
    SPLINE_SMOOTHING = 0    # 样条曲线平滑参数
    LINE_WIDTH = 2          # 主线条宽度
    
    # 均线参数配置
    MA_PERIODS = [20, 50, 100]  # 分析的均线周期
    DEFAULT_MA_PERIOD = 20      # 默认均线周期
    
    # 数据增强配置
    JITTER_RANGE = 0.03    # 随机浮动范围±3%
    JITTER_COPIES = 20     # 每个样本生成的副本数
    
    # 文件路径模板
    @staticmethod
    def stock_file_path(code: str) -> str:
        prefix = 'SH' if code.startswith('6') else 'SZ'
        return f"{prefix}{code}.txt"
    
    # 输出路径配置
    @staticmethod 
    def output_image_path(code: str, index: int) -> str:
        return f"Data/{code}-{index}.png"

if __name__ == "__main__":
    # 配置验证测试
    print(f"默认MA周期: {Config.DEFAULT_MA_PERIOD}")
    print(f"SH600000路径: {Config.stock_file_path('600000')}")