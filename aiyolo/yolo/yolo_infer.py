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
        #    0: 'person',    # 人
           39: 'bottle',   # 瓶子
           66: 'keyboard', # 键盘
        #    33: 'suitcase' # 行李箱
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
        results = self.model.track(frame, verbose=False, conf=0.6, tracker="bytetrack.yaml")  # 使用ByteTrack跟踪算法
        detections = []
        
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