from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolo12n.pt'):
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
        #    33: 'suitcase' # 行李箱
        }

    def detect(self, frame):
        """
        对输入帧进行目标检测
        Args:
            frame: 输入图像帧
        Returns:
            detections: 检测结果列表，每个元素包含类别、置信度和边界框
        """
        # 设置verbose=False来关闭性能信息输出
        results = self.model(frame, verbose=False)
        detections = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in self.classes:  # 只处理我们关心的类别
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append({
                        'cls': cls,
                        'class_name': self.classes[cls],
                        'conf': conf,
                        'bbox': xyxy
                    })
        
        return detections 