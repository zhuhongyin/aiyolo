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

def update_log_panel(panel, logs, max_logs=MAX_LOG_DISPLAY):
    """
    更新日志面板
    Args:
        panel: 日志面板图像
        logs: 日志列表
        max_logs: 最大显示日志条数，默认50条
    """
    if not logs:
        return panel  # 提前返回空日志情况

    # 设置背景
    panel[:] = BACKGROUND_COLOR

    # 绘制标题
    # panel = put_chinese_text(panel, "行为分析日志", (10, 10), text_size=24)

    # 设置固定列宽为屏幕1/3
    # col_width = panel.shape[1] // 3  # 使用面板实际宽度计算列宽
    # col_width = 340  # 固定列宽为340px
    col_width = 640  # 固定列宽为640px
    col_width2 = 1280  # 固定2列宽为1280px
    
    # 绘制表头（与create_log_panel列标题对齐）
    headers = ["行为分析", "已识别", "识别时间"]
    for i, header in enumerate(headers):
        x = i * col_width + 10  # 根据1/3宽度计算位置
        panel = put_chinese_text(panel, header, (x, 20), text_size=18)  # 调整y坐标适应无标题

    # 显示最新日志（倒序显示），动态计算行高
        visible_logs = logs[-max_logs:]
        text_height = 20  # 根据text_size=16调整，预留4px边距
        for i, log in enumerate(reversed(visible_logs)):
            y = 90 + i * (text_height + 8)  # 行高=文字高度+8px间距
            try:
                # 取消截断限制，通过调整列宽保证显示
                behavior = log['behavior']
                recognized = log['recognized']
                time_str = log['time']
            except KeyError as e:
                print(f"Missing key in log entry: {e}")
                continue

            # 调整列宽匹配文字显示需求
            panel = put_chinese_text(panel, behavior, (10, y), text_size=16)
            panel = put_chinese_text(panel, recognized, (col_width , y), text_size=16)  # 行为分析列宽扩展至340px
            panel = put_chinese_text(panel, time_str, (col_width2 , y), text_size=16)  # 识别时间列左移
    
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
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
            
            # 基础识别事件（已识别【人员】/【违禁物品】）
            base_behavior = f"已识别【{class_name}】" if class_name in ["人员", "违禁物品"] else ""
            if base_behavior:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logs.append({
                    'behavior': base_behavior,
                    'recognized': "是",
                    'time': current_time
                })

            # 检查预测位置是否在安全区域内（即将进入）
            if is_in_safe_zone(obj['future_bbox'], safe_zone):
                current_timestamp = time.time()
                last_predicted_alert = last_predicted_alert_times.get(obj['id'], 0)
                if obj['id'] not in predicted_alerted_objects or (current_timestamp - last_predicted_alert) > 180:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = {
                        'behavior': f"已识别【{class_name}】即将进入安全区域",
                        'recognized': "是",
                        'time': current_time
                    }
                    logs.append(log_entry)
                    print(f"预警：{label} 预计5秒内将进入安全区域！")
                    predicted_alerted_objects.add(obj['id'])
                    last_predicted_alert_times[obj['id']] = current_timestamp
                    write_log_to_file(log_entry)
            else:
                if obj['id'] in predicted_alerted_objects:
                    predicted_alerted_objects.remove(obj['id'])

            # 检查当前是否在安全区域内（已进入）
            if is_in_safe_zone(obj['bbox'], safe_zone):
                current_timestamp = time.time()
                last_alert = last_alert_times.get(obj['id'], 0)
                if obj['id'] not in alerted_objects or (current_timestamp - last_alert) > 180:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = {
                        'behavior': f"已识别【{class_name}】进入安全区域",
                        'recognized': "是",
                        'time': current_time
                    }
                    logs.append(log_entry)
                    print(f"警告：{label} 进入安全区域！")
                    alerted_objects.add(obj['id'])
                    last_alert_times[obj['id']] = current_timestamp
                    write_log_to_file(log_entry)
            else:
                if obj['id'] in alerted_objects:
                    alerted_objects.remove(obj['id'])

        # 更新日志面板
        log_panel = update_log_panel(log_panel, list(logs))  # 修复：保存返回值以更新面板

        # 调整主画面大小（上半部分高度为屏幕高度-260px）
        video_height = screen_height - 260
        frame = cv2.resize(frame, (screen_width, video_height))  # 宽度自适应屏幕

        # 调整日志面板大小（下半部分固定高度260px，宽度与视频区域一致）
        log_panel = cv2.resize(log_panel, (screen_width, 260))

        # 垂直拼接主画面和日志面板
        combined_frame = np.vstack((frame, log_panel))

        # 显示结果
        # 将帧存入全局变量供Flask使用
        global current_frame
        current_frame = combined_frame

        # 按ESC退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # 按ESC退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# 全局变量存储当前帧
current_frame = None

if __name__ == '__main__':
    # 启动Flask服务器
    from flask import Flask, Response
    app = Flask(__name__)

    def generate_frames():
        while True:
            if current_frame is not None:
                # 编码为JPEG格式
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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