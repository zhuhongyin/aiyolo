import cv2
import numpy as np
from yolo.yolo_infer import YOLODetector
from tracking.tracker import SimpleTracker
from utils import draw_bbox, draw_trajectory, is_in_safe_zone, calculate_distance
import os
import time
from datetime import datetime
import signal
import socket
import threading
from video_processor import VideoProcessor
from server import VideoServer
from logger import LogManager

# 以下函数已移动到logger.py模块
# def put_chinese_text(img, text, position, text_color=(0, 0, 0), text_size=20):
# def create_log_panel(width, height):
# def update_log_panel(panel):
# def write_log_to_file(log_entry):

# 以下常量已移动到config.py模块
# BACKGROUND_COLOR = 240
# MAX_LOG_DISPLAY = 50

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

def cleanup_resources():
    """释放所有资源"""
    print("清理资源...")
    if frame_manager.cap is not None:
        frame_manager.cap.release()
    cv2.destroyAllWindows()
    frame_manager.current_frame = None

def handle_signal(signum, frame):
    """处理信号"""
    cleanup_resources()
    time.sleep(0.5)
    exit(0)

def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def main():
    # 检查模型文件
    model_path = 'yolo11n.pt'
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    # 初始化组件
    detector = YOLODetector(model_path)
    tracker = SimpleTracker()
    video_processor = VideoProcessor(detector, tracker)
    log_manager = LogManager()
    
    # 初始化摄像头
    cap = video_processor.initialize_camera()
    if cap is None:
        return
    
    frame_manager.cap = cap
    
    # 设置安全区域
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_processor.set_safe_zone(frame_width, frame_height)
    
    # 主循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break
            
        # 处理帧
        processed_frame = video_processor.process_frame(frame)
        frame_manager.current_frame = processed_frame
        
        # 等待键盘输入
        cv2.waitKey(1)

if __name__ == '__main__':
    # 设置信号处理
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # 检查端口
    if is_port_in_use(5000):
        print("警告：端口5000已被占用，尝试释放...")
    
    # 创建单例实例
    frame_manager = FrameManager()
    
    # 启动视频处理线程
    video_thread = threading.Thread(target=main)
    video_thread.daemon = True
    video_thread.start()
    
    # 启动Flask服务器
    server = VideoServer(frame_manager)
    server.run()
