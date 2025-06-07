# 屏幕配置
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
LOG_PANEL_HEIGHT = 260

# 安全区域配置
SAFETY_ITEMS = [
    {'behavior': '已识别【人员】', 'recognized': False, 'time': ''},
    {'behavior': '已识别【违禁物品】', 'recognized': False, 'time': ''},
    {'behavior': '已识别【人员】进入安全区域', 'recognized': False, 'time': ''},
    {'behavior': '已识别【违禁物品】进入安全区域', 'recognized': False, 'time': ''},
    {'behavior': '已识别【人员】即将进入安全区域', 'recognized': False, 'time': ''},
    {'behavior': '已识别【违禁物品】即将进入安全区域', 'recognized': False, 'time': ''}
]

# 日志配置
MAX_LOG_DISPLAY = 50
LOG_DIR = './logs'
LOG_FILE = 'log.txt'

# 服务器配置
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000 