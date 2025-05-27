import numpy as np
import cv2
from collections import deque, OrderedDict
import time
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    """
    用于边界框跟踪的卡尔曼滤波器
    """
    count = 0
    
    def __init__(self, bbox):
        """
        创建跟踪器
        """
        # 确保边界框坐标顺序正确
        x1, y1, x2, y2 = bbox
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 初始化卡尔曼滤波器参数
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0], 
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        
        # 初始化状态
        self.kf.statePost = np.array([
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            0, 0, 0
        ], dtype=np.float32).reshape(-1, 1)
        
        self.kf.statePre = self.kf.statePost.copy()
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        """
        使用检测框更新跟踪器
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # 确保边界框坐标顺序正确
        x1, y1, x2, y2 = bbox
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        width = x2 - x1
        height = y2 - y1
        
        # 验证尺寸有效性
        if width <= 0 or height <= 0:
            print(f"Invalid bbox dimensions: width={width}, height={height}")
            return
        
        try:
            # 构造测量值并验证形状
            measurement = np.array([
                x1,
                y1,
                width,
                height
            ], dtype=np.float32).reshape(-1, 1)
            
            if measurement.shape != (4, 1):
                print(f"Invalid measurement shape: {measurement.shape}")
                return
            
            # 验证测量值是否为有限数值
            if not np.all(np.isfinite(measurement)):
                print(f"Invalid measurement values: {measurement}")
                return
            
            # 确保状态协方差矩阵正确初始化
            if not hasattr(self.kf, 'errorCovPost'):
                self.kf.errorCovPost = np.eye(7, dtype=np.float32)
            
            self.kf.correct(measurement)
        except Exception as e:
            print(f"Error in Kalman filter correction: {e}")
            print(f"Measurement: {measurement}")
            print(f"StatePost shape: {self.kf.statePost.shape}")
            print(f"MeasurementMatrix shape: {self.kf.measurementMatrix.shape}")
            return

    def predict(self):
        """
        执行预测步骤
        """
        if (self.kf.statePre[6] + self.kf.statePre[2]) <= 0:
            self.kf.statePre[6] *= 0
        
        predicted_state = self.kf.predict()
        pred_bbox = np.array([
            predicted_state[0, 0],
            predicted_state[1, 0],
            predicted_state[0, 0] + predicted_state[2, 0],
            predicted_state[1, 0] + predicted_state[3, 0]
        ])
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(pred_bbox)
        return pred_bbox
    
    def get_state(self):
        """
        获取当前边界框
        """
        state = self.kf.statePost
        return np.array([
            state[0, 0],
            state[1, 0],
            state[0, 0] + state[2, 0],
            state[1, 0] + state[3, 0]
        ])

class DeepSORTTracker:
    """
    基于DeepSORT算法的跟踪器
    """
    def __init__(self, max_iou_distance=0.7, max_age=30, n_init=3):
        """
        初始化跟踪器
        """
        self.trackers = []
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age  # Increase max age to prevent premature deletion
        self.n_init = n_init
        self.tracks = {}
        self.next_id = 0
        self.last_time = time.time()
        # Add edge buffer threshold (percentage of frame width/height)
        self.edge_buffer = 0.05  # 5% of frame dimensions
    
    def calculate_iou(self, bbox1, bbox2):
        """
        计算两个边界框的交并比(IoU)
        """
        # 解包边界框坐标
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # 计算交集区域
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # 计算交集面积
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # 计算每个边界框的面积
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        # 计算并集面积
        union_area = area1 + area2 - inter_area

        # 计算IoU
        return inter_area / union_area if union_area > 0 else 0
    
    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        将检测框关联到跟踪器
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), list(range(len(trackers)))
        
        # 计算IOU矩阵
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk_bbox in enumerate(trackers):
                iou_matrix[d, t] = self.calculate_iou(det, trk_bbox)
        
        # 使用匈牙利算法进行匹配
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.asarray(matched_indices).T
        
        # 保存匹配结果
        matched_pairs = []
        unmatched_detections = []
        
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
            else:
                matched_pairs.append(m.reshape(1, 2))
        
        if len(matched_pairs) == 0:
            matched_pairs = np.empty((0, 2), dtype=int)
        else:
            matched_pairs = np.concatenate(matched_pairs, axis=0)
        
        unmatched_trackers = np.array([
            t for t in range(len(trackers)) 
            if t not in matched_pairs[:, 1]
        ])
        
        return matched_pairs, unmatched_detections, unmatched_trackers
    
    def update(self, frame, detections):
        """
        更新跟踪器状态
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Filter out detections too close to edges
        valid_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Check if bbox center is within frame bounds minus buffer
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if (center_x > frame_width * self.edge_buffer and 
                center_x < frame_width * (1 - self.edge_buffer) and
                center_y > frame_height * self.edge_buffer and 
                center_y < frame_height * (1 - self.edge_buffer)):
                valid_detections.append(det)
        
        # Only use valid detections for tracking
        detections = valid_detections
        
        # 获取当前跟踪器列表
        active_trackers = []
        for track_id, tracker in list(self.tracks.items()):
            pred_bbox = tracker.predict()
            if any(pred_bbox):  # 如果跟踪器有效
                active_trackers.append((track_id, pred_bbox))
        
        # 提取边界框用于匹配
        tracker_bboxes = [bbox for _, bbox in active_trackers]
        detection_bboxes = [det['bbox'] for det in detections]
        
        # 执行检测框与跟踪器的关联
        association_result = self.associate_detections_to_trackers(
            detection_bboxes,
            tracker_bboxes,
            iou_threshold=0.3
        )
        
        # 处理关联结果
        if len(association_result) == 3:
            matched_pairs, unmatched_dets, _ = association_result
        else:
            matched_pairs = np.empty((0, 2), dtype=int)
            unmatched_dets = list(range(len(detections)))
        
        # 更新匹配的跟踪器
        for m in matched_pairs:
            det_index, trk_index = m[0], m[1]
            track_id, _ = active_trackers[trk_index]
            
            # 获取并验证检测框
            bbox = detections[det_index]['bbox']
            x1, y1, x2, y2 = bbox
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 确保坐标有效且尺寸合理
            if x2 - x1 > 0 and y2 - y1 > 0:  # 确保尺寸有效
                # 使用规范化后的坐标更新跟踪器
                self.tracks[track_id].update([x1, y1, x2, y2])
                self.tracks[track_id].hits += 1
                self.tracks[track_id].time_since_update = 0
        
        # 移除过期的跟踪器
        for t, (track_id, _) in enumerate(active_trackers):
            if self.tracks[track_id].time_since_update > self.max_age:
                del self.tracks[track_id]
        
        # 创建新跟踪器
        for d in unmatched_dets:
            bbox = detections[d]["bbox"]
            x1, y1, x2, y2 = bbox
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 确保坐标有效且尺寸合理
            if x2 - x1 > 0 and y2 - y1 > 0:  # 确保尺寸有效
                # 使用规范化后的坐标创建新跟踪器
                self.tracks[self.next_id] = KalmanBoxTracker([x1, y1, x2, y2])
                self.tracks[self.next_id].hits = 0
                self.tracks[self.next_id].hit_streak = 0
                self.next_id += 1
        
        # 准备返回结果
        tracked_objects = []
        for track_id, tracker in self.tracks.items():
            bbox = tracker.get_state()
            tracked_objects.append({
                'id': track_id,
                'class_name': detections[0]['class_name'] if detections else 'unknown',
                'bbox': tuple(map(int, bbox)),
                'velocity': (0, 0),
                'future_bbox': tuple(map(int, bbox))
            })
        
        return tracked_objects