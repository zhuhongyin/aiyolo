import cv2
import numpy as np
from datetime import datetime
from config import SCREEN_WIDTH, SCREEN_HEIGHT, LOG_PANEL_HEIGHT, SAFETY_ITEMS
from logger import create_log_panel, update_log_panel
from utils import draw_bbox, is_in_safe_zone

class VideoProcessor:
    def __init__(self, detector, tracker):
        self.detector = detector
        self.tracker = tracker
        self.frame_manager = None
        self.safe_zone = None
        self.alerted_objects = set()
        self.predicted_alerted_objects = set()
        self.last_alert_times = {}
        self.last_predicted_alert_times = {}

    def initialize_camera(self):
        """初始化摄像头"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("无法打开摄像头")
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            return cap
        except Exception as e:
            print(f"摄像头初始化错误: {e}")
            return None

    def set_safe_zone(self, frame_width, frame_height):
        """设置安全区域"""
        self.safe_zone = [
            frame_width // 3,
            frame_height // 3,
            2 * frame_width // 3,
            2 * frame_height // 3
        ]

    def process_frame(self, frame):
        """处理单个视频帧"""
        # 目标检测
        detections = self.detector.detect(frame)
        
        # 目标跟踪
        tracked_objects = self.tracker.update(frame, detections)
        
        # 创建日志面板
        log_panel = create_log_panel(SCREEN_WIDTH, LOG_PANEL_HEIGHT)
        
        # 绘制安全区域
        cv2.rectangle(frame, (self.safe_zone[0], self.safe_zone[1]), 
                     (self.safe_zone[2], self.safe_zone[3]),
                     (0, 0, 255), 2, cv2.LINE_AA)
        
        # 处理每个跟踪目标
        for obj in tracked_objects:
            self._process_tracked_object(frame, obj)
        
        # 更新日志面板
        log_panel = update_log_panel(log_panel)
        
        # 调整主画面大小
        video_height = SCREEN_HEIGHT - LOG_PANEL_HEIGHT
        frame = cv2.resize(frame, (SCREEN_WIDTH, video_height))
        
        # 调整日志面板大小
        log_panel = cv2.resize(log_panel, (SCREEN_WIDTH, LOG_PANEL_HEIGHT))
        
        # 垂直拼接主画面和日志面板
        return np.vstack((frame, log_panel))

    def _process_tracked_object(self, frame, obj):
        """处理单个跟踪目标"""
        # 绘制当前边界框和标签
        label = f"{obj['class_name']} ID:{obj['id']}"
        frame = draw_bbox(frame, obj['bbox'], label)
        
        # 提取类别名称
        class_name = obj['class_name']
        is_person = class_name == "person"
        
        # 检查预测位置是否在安全区域内
        predicted_in_zone = is_in_safe_zone(obj['future_bbox'], self.safe_zone)
        current_in_zone = is_in_safe_zone(obj['bbox'], self.safe_zone)
        
        # 更新安全状态
        self._update_safety_status(is_person, predicted_in_zone, current_in_zone)

    def _update_safety_status(self, is_person, predicted_in_zone, current_in_zone):
        """更新安全状态"""
        # 基础识别事件（已识别【人员】/【违禁物品】）
        base_behavior = '已识别【人员】' if is_person else '已识别【违禁物品】'
        for item in SAFETY_ITEMS:
            if item['behavior'] == base_behavior:
                item['recognized'] = True
                item['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 检查预测位置是否在安全区域内（即将进入）
        behavior = ('已识别【人员】即将进入安全区域' if is_person 
                   else '已识别【违禁物品】即将进入安全区域')
        for item in SAFETY_ITEMS:
            if item['behavior'] == behavior:
                item['recognized'] = predicted_in_zone
                if predicted_in_zone:
                    item['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 检查当前是否在安全区域内（已进入）
        behavior = ('已识别【人员】进入安全区域' if is_person 
                   else '已识别【违禁物品】进入安全区域')
        for item in SAFETY_ITEMS:
            if item['behavior'] == behavior:
                item['recognized'] = current_in_zone
                if current_in_zone:
                    item['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 