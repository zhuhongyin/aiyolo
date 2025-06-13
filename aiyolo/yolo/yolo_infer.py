from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolo11n.pt'):
        """
        初始化YOLO检测器
        Args:
            model_path: YOLO模型路径，默认使用yolo12n.pt
        """
        self.model = YOLO(model_path)
        # 定义目标类别
        self.classes = {
           0: 'person',    # 人
           39: 'bottle',   # 瓶子
           66: 'keyboard', # 键盘
           33: 'suitcase' # 行李箱
        }

    def detect(self, frame):
        """
        对输入帧进行目标检测与跟踪（启用YOLO内置跟踪）
        Args:
            frame: 输入图像帧
        Returns:
            detections: 检测结果列表（包含跟踪ID、类别、置信度、边界框）
        """
        # 启用跟踪并设置置信度阈值（conf=0.5）
        # 初始化前帧置信度缓存（首次调用时）
        if not hasattr(self, 'prev_confidences'):
            self.prev_confidences = None

        # 执行YOLO跟踪推理
        results = self.model.track(frame, verbose=False, conf=0.3, tracker="bytetrack.yaml")  # 使用ByteTrack跟踪算法
        detections = []

        # 获取当前帧置信度（用于后续缓存）
        current_confidences = [float(box.conf[0]) for r in results for box in r.boxes if int(box.cls[0]) in self.classes]

        # 叠加前帧置信度（解决识别框闪烁）
        if self.prev_confidences is not None:
            # 按索引匹配前后帧置信度（简单实现），权重0.3平衡平滑性
            min_len = min(len(current_confidences), len(self.prev_confidences))
            for i in range(min_len):
                current_confidences[i] += self.prev_confidences[i] * 0.5  # 可调整权重系数（建议0.05~0.5）

        # 更新前帧置信度缓存（供下一帧使用）
        self.prev_confidences = current_confidences.copy()
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in self.classes:  # 只处理关心的类别
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    track_id = int(box.id[0]) if box.id is not None else -1  # 获取YOLO生成的跟踪ID
                    detections.append({
                        'cls': cls,
                        'class_name': self.classes[cls],
                        'conf': conf,
                        'bbox': xyxy,
                        'track_id': track_id  # 新增跟踪ID字段
                    })
        
        return detections