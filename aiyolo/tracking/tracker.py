import cv2
import numpy as np
from collections import deque
import time

class SimpleTracker:
    def __init__(self, max_track_length=30, max_missed=55):
        """
        初始化跟踪器
        Args:
            max_track_length: 轨迹最大长度
            max_missed: 最大允许未匹配帧数
        """
        self.tracks = {}  # 存储每个目标的轨迹
        self.next_id = 0
        self.max_track_length = max_track_length
        self.last_time = time.time()
        self.max_distance = 50  # 最大匹配距离
        self.max_missed = max_missed  # 最大未匹配帧数

    def calculate_center(self, bbox):
        """
        计算边界框中心点
        """
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def calculate_distance(self, center1, center2):
        """
        计算两点之间的欧氏距离
        """
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def calculate_velocity(self, trajectory):
        """
        计算目标速度
        Args:
            trajectory: 轨迹列表
        Returns:
            velocity: 速度向量 (vx, vy)
        """
        if len(trajectory) < 2:
            return (0, 0)
        
        # 获取最近的两个位置
        current = trajectory[-1]
        previous = trajectory[-2]
        
        # 计算中心点
        current_center = self.calculate_center(current)
        previous_center = self.calculate_center(previous)
        
        # 计算速度（像素/秒）
        dt = time.time() - self.last_time
        if dt == 0:
            return (0, 0)
            
        vx = (current_center[0] - previous_center[0]) / dt
        vy = (current_center[1] - previous_center[1]) / dt
        
        return (vx, vy)

    def predict_future_position(self, current_bbox, velocity, seconds=5):
        """
        预测未来位置
        Args:
            current_bbox: 当前边界框
            velocity: 速度向量
            seconds: 预测时间（秒）
        Returns:
            future_bbox: 预测的边界框
        """
        # 计算当前中心点
        center_x = (current_bbox[0] + current_bbox[2]) // 2
        center_y = (current_bbox[1] + current_bbox[3]) // 2
        
        # 计算预测位置
        future_x = center_x + velocity[0] * seconds
        future_y = center_y + velocity[1] * seconds
        
        # 计算边界框宽度和高度
        width = current_bbox[2] - current_bbox[0]
        height = current_bbox[3] - current_bbox[1]
        
        # 返回预测的边界框
        return (
            int(future_x - width/2),
            int(future_y - height/2),
            int(future_x + width/2),
            int(future_y + height/2)
        )

    def update(self, frame, detections):
        """
        更新跟踪器状态
        Args:
            frame: 当前帧
            detections: YOLO检测结果
        Returns:
            tracked_objects: 跟踪结果列表
        """
        tracked_objects = []
        current_time = time.time()
        # 标记所有轨迹为未匹配
        for track in self.tracks.values():
            track['matched'] = False
        # 计算所有现有轨迹的中心点
        existing_centers = {
            track_id: self.calculate_center(track_info['current_bbox'])
            for track_id, track_info in self.tracks.items()
        }
        # 计算所有新检测的中心点
        detection_centers = [
            self.calculate_center(det['bbox'])
            for det in detections
        ]
        # 匹配检测结果和现有轨迹
        for i, det in enumerate(detections):
            det_center = detection_centers[i]
            det_class = det['class_name']
            best_match = None
            best_distance = float('inf')
            for track_id, track_center in existing_centers.items():
                track_info = self.tracks[track_id]
                if track_info['matched']:
                    continue
                # 类别不同不匹配
                if track_info['class_name'] != det_class:
                    continue
                distance = self.calculate_distance(det_center, track_center)
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_match = track_id
            if best_match is not None:
                # 更新现有轨迹
                track_info = self.tracks[best_match]
                track_info['current_bbox'] = det['bbox']
                track_info['history'].append(det['bbox'])
                velocity = self.calculate_velocity(list(track_info['history']))
                track_info['velocity'] = velocity
                future_bbox = self.predict_future_position(det['bbox'], velocity)
                track_info['missed'] = 0  # 匹配上，missed清零
                track_info['matched'] = True
                tracked_objects.append({
                    'id': best_match,
                    'class_name': track_info['class_name'],
                    'bbox': det['bbox'],
                    'velocity': velocity,
                    'future_bbox': future_bbox
                })
            else:
                # 创建新轨迹
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'class_name': det['class_name'],
                    'history': deque(maxlen=self.max_track_length),
                    'current_bbox': det['bbox'],
                    'velocity': (0, 0),
                    'missed': 0,
                    'matched': True
                }
                self.tracks[track_id]['history'].append(det['bbox'])
                tracked_objects.append({
                    'id': track_id,
                    'class_name': det['class_name'],
                    'bbox': det['bbox'],
                    'velocity': (0, 0),
                    'future_bbox': det['bbox']
                })
        # 更新未匹配轨迹的missed计数
        to_delete = []
        for track_id, track_info in self.tracks.items():
            if not track_info['matched']:
                track_info['missed'] = track_info.get('missed', 0) + 1
                if track_info['missed'] > self.max_missed:
                    to_delete.append(track_id)
        for track_id in to_delete:
            del self.tracks[track_id]
        self.last_time = current_time
        return tracked_objects

    def get_trajectory(self, track_id):
        """
        获取指定ID的轨迹
        Args:
            track_id: 轨迹ID
        Returns:
            trajectory: 轨迹列表
        """
        if track_id in self.tracks:
            return list(self.tracks[track_id]['history'])
        return []