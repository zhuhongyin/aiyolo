import cv2
import numpy as np
from yolo.yolo_infer import YOLODetector
from tracking.tracker import SimpleTracker
from utils import draw_bbox, draw_trajectory, is_in_safe_zone, calculate_distance
import os
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def put_chinese_text(img, text, position, text_color=(0, 0, 0), text_size=20):
    """
    使用PIL绘制中文文字
    """
    # 创建PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 加载中文字体
    font = ImageFont.truetype("simhei.ttf", text_size, encoding="utf-8")
    
    # 绘制文字
    draw.text(position, text, font=font, fill=text_color)
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def create_log_panel(width, height):
    """
    创建日志面板（下半部分固定高度260px）
    """
    # 创建浅灰色背景 (RGB: 240, 240, 240)
    panel = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # 绘制表头（取消标题，列宽为屏幕1/3）
    headers = ["行为分析", "已识别", "识别时间"]
    col_width = width // 3  # 每列宽度为屏幕宽度的1/3
    for i, header in enumerate(headers):
        x = i * col_width + 10  # 根据1/3宽度计算位置
        panel = put_chinese_text(panel, header, (x, 20), text_size=20)  # 调整y坐标适应无标题
    
    return panel

BACKGROUND_COLOR = 240
MAX_LOG_DISPLAY = 50

def update_log_panel(panel):
    col_width = panel.shape[1] // 3
    for i, item in enumerate(SAFETY_ITEMS):
        y = 50 + i * 36
        panel = put_chinese_text(panel, item['behavior'], (20, y))
        recognized_text = "是" if item['recognized'] else "否"
        panel = put_chinese_text(panel, recognized_text, 
                               (col_width + 20, y))
        panel = put_chinese_text(panel, item['time'], 
                               (2*col_width + 20, y))
    return panel

def write_log_to_file(log_entry):
    """
    将日志条目写入文件
    Args:
        log_entry: 包含行为、识别状态和时间的日志字典
    """
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'log.txt')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{log_entry['time']}] {log_entry['behavior']} - 已识别: {log_entry['recognized']}\n")

def main():
    # 初始化摄像头
    # cap = cv2.VideoCapture("rtsp://admin:dtct123456@10.10.140.144")
    global cap
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            # 此时检测器和跟踪器尚未初始化，无需清理
            return
    except Exception as e:
        print(f"摄像头初始化错误: {e}")
        cleanup_resources()
        return

    # 设置缓冲区大小, 越小延迟越低, 但可能会丢失帧
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
    # 禁用自动白平衡,提高检测的稳定性
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    # 设置视频编码, 需要保存视频或通过网络传输视频时使用
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'X264'))

    # 检查模型文件是否存在
    model_path = 'yolo11n.pt'
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请手动下载模型文件例如yolov8n.pt、yolo11n.pt并放置在项目根目录：")
        return

    # 初始化检测器和跟踪器
    detector = YOLODetector(model_path)
    tracker = SimpleTracker()

    # 获取屏幕分辨率
    screen_width = 1920  # 可以根据实际屏幕调整
    screen_height = 1080

    # 计算左右区域宽度
    left_width = int(screen_width * 0.7)
    right_width = screen_width - left_width

    # 定义安全区域（示例：图像中心区域）
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    safe_zone = [
        frame_width // 3,
        frame_height // 3,
        2 * frame_width // 3,
        2 * frame_height // 3
    ]

    # 用于记录已经报警的目标及最后报警时间（时间窗口去重）
    alerted_objects = set()
    predicted_alerted_objects = set()
    last_alert_times = {}  # 格式：{id: 最后报警时间戳}
    last_predicted_alert_times = {}  # 格式：{id: 最后预测报警时间戳}
    
    # 初始化日志列表（使用deque限制最大长度）
    from collections import deque
    logs = deque(maxlen=50)  # 最多保存50条日志

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break

        # 目标检测
        detections = detector.detect(frame)

        # 目标跟踪
        tracked_objects = tracker.update(frame, detections)

        # 创建日志面板 col_width = panel.shape[1] // 3
        log_panel = create_log_panel(screen_width, 260)  # 固定宽度为580px

        # 绘制安全区域
        cv2.rectangle(frame, (safe_zone[0], safe_zone[1]), 
                     (safe_zone[2], safe_zone[3]),
                     (0, 0, 255), 2, cv2.LINE_AA)

        # 处理每个跟踪目标
        for obj in tracked_objects:
            # 绘制当前边界框和标签
            label = f"{obj['class_name']} ID:{obj['id']}"
            frame = draw_bbox(frame, obj['bbox'], label)

            # 提取类别名称（用于区分人员/违禁物品）
            class_name = obj['class_name']
            # is_person = class_name == "人员"
            is_person = class_name == "person"
            
            # 更新固定行为条目
            behavior_keys = [
                '已识别【人员】',
                '已识别【违禁物品】',
                '已识别【人员】进入安全区域',
                '已识别【违禁物品】进入安全区域',
                '已识别【人员】即将进入安全区域',
                '已识别【违禁物品】即将进入安全区域'
            ]
            
            # 基础识别事件（已识别【人员】/【违禁物品】）
            base_behavior = '已识别【人员】' if is_person else '已识别【违禁物品】'
            for item in SAFETY_ITEMS:
                if item['behavior'] == base_behavior:
                    item['recognized'] = True
                    item['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 检查预测位置是否在安全区域内（即将进入）
            predicted_in_zone = is_in_safe_zone(obj['future_bbox'], safe_zone)
            behavior = ('已识别【人员】即将进入安全区域' if is_person 
                       else '已识别【违禁物品】即将进入安全区域')
            for item in SAFETY_ITEMS:
                if item['behavior'] == behavior:
                    item['recognized'] = predicted_in_zone
                    if predicted_in_zone:
                        item['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 检查当前是否在安全区域内（已进入）
            current_in_zone = is_in_safe_zone(obj['bbox'], safe_zone)
            behavior = ('已识别【人员】进入安全区域' if is_person 
                       else '已识别【违禁物品】进入安全区域')
            for item in SAFETY_ITEMS:
                if item['behavior'] == behavior:
                    item['recognized'] = current_in_zone
                    if current_in_zone:
                        item['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 更新日志面板
        log_panel = update_log_panel(log_panel)

        # 调整主画面大小（上半部分高度为屏幕高度-260px）
        video_height = screen_height - 260
        frame = cv2.resize(frame, (screen_width, video_height))  # 宽度自适应屏幕

        # 调整日志面板大小（下半部分固定高度260px，宽度与视频区域一致）
        log_panel = cv2.resize(log_panel, (screen_width, 260))

        # 垂直拼接主画面和日志面板
        combined_frame = np.vstack((frame, log_panel))

        # 显示结果
        # 使用FrameManager单例存储当前帧
        frame_manager.current_frame = combined_frame

        # 持续处理视频流， 在视频处理循环中，即使不需要检测按键，也需要调用这个函数来保持窗口响应。
        # 等待键盘输入 ：暂停程序执行1毫秒，等待用户按键输入
        cv2.waitKey(1) 

    # 服务器关闭时会自动调用cleanup_resources()

class FrameManager:
    _instance = None
    _current_frame = None
    _cap = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FrameManager, cls).__new__(cls)
        return cls._instance
    
    @property
    def current_frame(self):
        return self._current_frame
    
    @current_frame.setter
    def current_frame(self, value):
        self._current_frame = value
        
    @property
    def cap(self):
        return self._cap
    
    @cap.setter
    def cap(self, value):
        self._cap = value

# 创建单例实例
frame_manager = FrameManager()
SAFETY_ITEMS = [
    {'behavior': '已识别【人员】', 'recognized': False, 'time': ''},
    {'behavior': '已识别【违禁物品】', 'recognized': False, 'time': ''},
    {'behavior': '已识别【人员】进入安全区域', 'recognized': False, 'time': ''},
    {'behavior': '已识别【违禁物品】进入安全区域', 'recognized': False, 'time': ''},
    {'behavior': '已识别【人员】即将进入安全区域', 'recognized': False, 'time': ''},
    {'behavior': '已识别【违禁物品】即将进入安全区域', 'recognized': False, 'time': ''}
]

def cleanup_resources():
    """释放所有资源"""
    print("清理资源...")
    if frame_manager.cap is not None:
        frame_manager.cap.release()
    cv2.destroyAllWindows()
    frame_manager.current_frame = None  # 添加此行

if __name__ == '__main__':
    # 设置信号处理
    import signal
    def handle_signal(signum, frame):
        cleanup_resources()
        # 等待IO操作完成（可选）
        time.sleep(0.5)  
        exit(0)
    
    signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # kill命令

    # 启动Flask服务器前检查端口
    import socket
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    if is_port_in_use(5000):
        print("警告：端口5000已被占用，尝试释放...")
        # 这里可以添加平台相关的端口释放命令（如Windows用netstat查找PID并终止）

    # 启动Flask服务器
    from flask import Flask, Response
    app = Flask(__name__)

    def generate_frames():
        while True:
            # 使用FrameManager获取当前帧
            current_frame = frame_manager.current_frame
            if current_frame is not None and not isinstance(current_frame, type(None)):
                try:
                    ret, buffer = cv2.imencode('.jpg', current_frame)
                    if ret:  # 只有编码成功时才生成帧
                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                except Exception as e:
                    print(f"帧编码错误: {e}")
                    time.sleep(0.1)  # 防止CPU过载
            else:
                time.sleep(0.1)

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    # 启动视频处理线程
    import threading
    video_thread = threading.Thread(target=main)
    video_thread.daemon = True
    video_thread.start()

    # 启动Flask服务器
    app.run(host='0.0.0.0', port=5000, threaded=True)
