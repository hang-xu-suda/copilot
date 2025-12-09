"""
反向求解器配置文件（完整版）
"""

# ═══════════════════════════════════════════════════════════════════
# 数据配置
# ═══════════════════════════════════════════════════════════════════

DATA_PATH = 'base_data/spotar_data.pkl.gz'

# ═══════════════════════════════════════════════════════════════════
# 反向求解器算法参数
# ═══════════════════════════════════════════════════════════════════

# α离散近似参数
REVERSE_L1 = 50  # 路径分布离散化级别（推荐30-100）
REVERSE_L2 = 10  # 每个时间片采样的旅行时间数（推荐5-20）

# 求解参数
REVERSE_MAX_LABELS = 100000  # 最大标签数
REVERSE_ALPHA_DEFAULT = 0.95  # 默认可靠性要求

# 输出控制
REVERSE_VERBOSE = False  # 是否输出详细信息
REVERSE_PRINT_INTERVAL = 100  # 打印间隔

# 容差参数
SLOT_TOLERANCE = 5  # 时间片匹配容差（推荐5-10）

# ═══════════════════════════════════════════════════════════════════
# 默认问题设置
# ═══════════════════════════════════════════════════════════════════

# 默认目标到达时间
DEFAULT_ARRIVAL_HOUR = 9
DEFAULT_ARRIVAL_MINUTE = 0

# ═══════════════════════════════════════════════════════════════════
# 敏感性分析配置
# ═══════════════════════════════════════════════════════════════════

# α敏感性分析测试值
ALPHA_SENSITIVITY_VALUES = [0.85, 0.90, 0.95, 0.99]

# 时间预算分析测试时间点（小时.分钟）
TIME_BUDGET_TEST_TIMES = [
    (8, 0),   # 08:00
    (8, 30),  # 08:30
    (9, 0),   # 09:00
    (9, 30),  # 09:30
    (10, 0),  # 10:00
]

# ═══════════════════════════════════════════════════════════════════
# 预设配置方案
# ═══════════════════════════════════════════════════════════════════

# 快速模式（用于测试）
FAST_MODE = {
    'L1': 30,
    'L2': 5,
    'max_labels': 10000,
    'description': '快速测试模式'
}

# 标准模式（推荐）
STANDARD_MODE = {
    'L1': 50,
    'L2': 10,
    'max_labels': 50000,
    'description': '标准求解模式'
}

# 精确模式（高精度）
ACCURATE_MODE = {
    'L1': 100,
    'L2': 20,
    'max_labels': 100000,
    'description': '高精度求解模式'
}

# 默认使用标准模式
DEFAULT_MODE = STANDARD_MODE

# ═══════════════════════════════════════════════════════════════════
# 测试模式参数
# ═══════════════════════════════════════════════════════════════════

TEST_L1 = 30
TEST_L2 = 5
TEST_MAX_LABELS = 10000
TEST_ALPHA = 0.95
TEST_VERBOSE = False

# ═══════════════════════════════════════════════════════════════════
# 可视化参数
# ═══════════════════════════════════════════════════════════════════

FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
SHOW_PLOTS = True

# 图表颜色
COLORS = {
    'departure': '#90EE90',      # 浅绿色
    'arrival': '#FFB6C1',        # 浅粉色
    'reserved': '#FFFF99',       # 浅黄色
    'cdf': '#2E8B57',            # 海绿色
    'reliability': '#DC143C',    # 深红色
    'expected': '#4169E1'        # 皇家蓝
}

# ═══════════════════════════════════════════════════════════════════
# 性能优化参数
# ═══════════════════════════════════════════════════════════════════

# 缓存配置
ENABLE_SLOT_CACHE = True  # 启用时间片缓存
CACHE_SIZE_LIMIT = 10000  # 缓存大小限制

# 内存管理
MAX_MEMORY_MB = 4096
ENABLE_MEMORY_CHECK = False

# ═══════════════════════════════════════════════════════════════════
# 日志配置
# ═══════════════════════════════════════════════════════════════════

LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
SAVE_LOG = False
LOG_FILE = 'reverse_solver.log'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ═══════════════════════════════════════════════════════════════════
# 验证参数
# ═══════════════════════════════════════════════════════════════════

# 验证开关
ENABLE_TIME_CONSISTENCY_CHECK = True
ENABLE_RANGE_CHECK = True
ENABLE_PROBABILITY_CHECK = False

# 调试输出
DEBUG_CONVOLUTION = False
DEBUG_DOMINANCE = False
DEBUG_WEIGHT_CALCULATION = False  # 调试概率权重计算

# ═══════════════════════════════════════════════════════════════════
# 其他配置
# ═══════════════════════════════════════════════════════════════════

RANDOM_SEED = 42

# 时间单位
TIME_UNIT = 0.1  # 分钟
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24

# 时间格式
TIME_FORMAT = '%H:%M'
DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# ═══════════════════════════════════════════════════════════════════
# 实验配置
# ═══════════════════════════════════════════════════════════════════

EXPERIMENT_CONFIGS = {
    'baseline': {
        'L1': 50,
        'L2': 10,
        'max_labels': 50000,
        'description': '基线配置'
    },
    'high_precision': {
        'L1': 100,
        'L2': 20,
        'max_labels': 100000,
        'description': '高精度配置'
    },
    'fast': {
        'L1': 30,
        'L2': 5,
        'max_labels': 10000,
        'description': '快速配置'
    },
    'ultra_fast': {
        'L1': 20,
        'L2': 3,
        'max_labels': 5000,
        'description': '超快速配置'
    }
}

# ═══════════════════════════════════════════════════════════════════
# 配置验证函数
# ═══════════════════════════════════════════════════════════════════

def validate_config():
    """验证配置参数的合理性"""
    errors = []
    warnings = []
    
    # 验证L1
    if REVERSE_L1 < 10:
        errors.append(f"L1={REVERSE_L1} 太小，最小推荐值为10")
    elif REVERSE_L1 > 200:
        warnings.append(f"L1={REVERSE_L1} 较大，可能影响性能")
    
    # 验证L2
    if REVERSE_L2 < 2:
        errors.append(f"L2={REVERSE_L2} 太小，最小推荐值为2")
    elif REVERSE_L2 > 50:
        warnings.append(f"L2={REVERSE_L2} 较大，可能影响性能")
    
    # 验证alpha
    if not (0 < REVERSE_ALPHA_DEFAULT < 1):
        errors.append(f"ALPHA={REVERSE_ALPHA_DEFAULT} 必须在(0, 1)范围内")
    
    # 验证max_labels
    if REVERSE_MAX_LABELS < 1000:
        warnings.append(f"MAX_LABELS={REVERSE_MAX_LABELS} 较小，可能找不到解")
    
    # 打印结果
    if errors:
        print("❌ 配置错误:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    if warnings:
        print("⚠ 配置警告:")
        for warn in warnings:
            print(f"  - {warn}")
    
    if not errors and not warnings:
        print("✓ 配置验证通过")
    
    return len(errors) == 0


def get_mode_config(mode_name: str = 'standard'):
    """
    获取指定模式的配置
    
    Args:
        mode_name: 模式名称 ('fast', 'standard', 'accurate')
        
    Returns:
        配置字典
    """
    modes = {
        'fast': FAST_MODE,
        'standard': STANDARD_MODE,
        'accurate': ACCURATE_MODE
    }
    
    return modes.get(mode_name, STANDARD_MODE)


def print_config_summary():
    """打印配置摘要"""
    print(f"\n{'='*70}")
    print(f"配置摘要")
    print(f"{'='*70}")
    print(f"  数据路径: {DATA_PATH}")
    print(f"  算法参数:")
    print(f"    - L1: {REVERSE_L1}")
    print(f"    - L2: {REVERSE_L2}")
    print(f"    - 默认α: {REVERSE_ALPHA_DEFAULT}")
    print(f"    - 最大标签数: {REVERSE_MAX_LABELS:,}")
    print(f"  输出控制:")
    print(f"    - 详细输出: {REVERSE_VERBOSE}")
    print(f"    - 打印间隔: {REVERSE_PRINT_INTERVAL}")
    print(f"  缓存: {'开启' if ENABLE_SLOT_CACHE else '关闭'}")
    print(f"  日志: {'开启' if SAVE_LOG else '关闭'}")
    print(f"{'='*70}\n")


# 自动验证配置
if __name__ == "__main__":
    print("验证配置文件...")
    validate_config()
    print()
    print_config_summary()