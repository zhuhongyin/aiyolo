import cv2
import numpy as np

def is_in_safe_zone(bbox, safe_zone):
    """
    判断目标是否在安全区域内
    Args:
        bbox: 目标边界框 [x1, y1, x2, y2]
        safe_zone: 安全区域 [x1, y1, x2, y2]
    Returns:
        bool: 是否在安全区域内
    """
    x, y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    return (safe_zone[0] <= x <= safe_zone[2] and 
            safe_zone[1] <= y <= safe_zone[3])

def calculate_distance(bbox1, bbox2):
    """
    计算两个边界框中心点之间的距离
    Args:
        bbox1: 第一个边界框 [x1, y1, x2, y2]
        bbox2: 第二个边界框 [x1, y1, x2, y2]
    Returns:
        float: 像素距离
    """
    center1 = ((bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2)
    center2 = ((bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def draw_trajectory(frame, trajectory, color=(0, 255, 0)):
    """
    在图像上绘制轨迹
    Args:
        frame: 输入图像
        trajectory: 轨迹点列表
        color: 轨迹颜色
    Returns:
        frame: 绘制轨迹后的图像
    """
    if len(trajectory) < 2:
        return frame
    
    points = np.array(trajectory, dtype=np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(frame, [points], False, color, 2)
    return frame

def draw_bbox(frame, bbox, label, color=(0, 255, 0)):
    """
    在图像上绘制边界框和标签
    Args:
        frame: 输入图像
        bbox: 边界框 [x1, y1, x2, y2]
        label: 标签文本
        color: 边界框颜色
    Returns:
        frame: 绘制边界框后的图像
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame 