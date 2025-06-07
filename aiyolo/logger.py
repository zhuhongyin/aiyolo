import os
from datetime import datetime
from collections import deque
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import LOG_DIR, LOG_FILE, MAX_LOG_DISPLAY, SAFETY_ITEMS

def put_chinese_text(img, text, position, text_color=(0, 0, 0), text_size=20):
    """使用PIL绘制中文文字"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("simhei.ttf", text_size, encoding="utf-8")
    draw.text(position, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def create_log_panel(width, height):
    """创建日志面板"""
    panel = np.ones((height, width, 3), dtype=np.uint8) * 240
    headers = ["行为分析", "已识别", "识别时间"]
    col_width = width // 3
    for i, header in enumerate(headers):
        x = i * col_width + 10
        panel = put_chinese_text(panel, header, (x, 20), text_size=20)
    return panel

def update_log_panel(panel):
    """更新日志面板内容"""
    col_width = panel.shape[1] // 3
    for i, item in enumerate(SAFETY_ITEMS):
        y = 50 + i * 36
        panel = put_chinese_text(panel, item['behavior'], (20, y))
        recognized_text = "是" if item['recognized'] else "否"
        panel = put_chinese_text(panel, recognized_text, (col_width + 20, y))
        panel = put_chinese_text(panel, item['time'], (2*col_width + 20, y))
    return panel

def write_log_to_file(log_entry):
    """将日志条目写入文件"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_path = os.path.join(LOG_DIR, LOG_FILE)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{log_entry['time']}] {log_entry['behavior']} - 已识别: {log_entry['recognized']}\n")

class LogManager:
    def __init__(self):
        self.logs = deque(maxlen=MAX_LOG_DISPLAY)
    
    def add_log(self, behavior, recognized):
        """添加新的日志条目"""
        log_entry = {
            'behavior': behavior,
            'recognized': recognized,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.logs.append(log_entry)
        write_log_to_file(log_entry) 