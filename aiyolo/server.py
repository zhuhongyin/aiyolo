import cv2
import time
from flask import Flask, Response
from config import SERVER_HOST, SERVER_PORT

# 基于 Flask 的视频流服务器，主要功能是将实时视频帧通过 HTTP 接口传输给客户端（例如网页浏览器）。
# 它使用了 MJPEG（Motion JPEG）协议来实现视频流的推送。
class VideoServer:
    def __init__(self, frame_manager):
        self.app = Flask(__name__)
        self.frame_manager = frame_manager
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._generate_frames(), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')

    def _generate_frames(self):
        """生成视频流"""
        while True:
            current_frame = self.frame_manager.current_frame
            if current_frame is not None and not isinstance(current_frame, type(None)):
                try:
                    ret, buffer = cv2.imencode('.jpg', current_frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
                except Exception as e:
                    print(f"帧编码错误: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def run(self):
        """启动服务器"""
        self.app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True) 