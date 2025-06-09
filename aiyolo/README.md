# AIYOLO 智能视频监控系统

## 项目概述
AIYOLO是一个基于YOLO目标检测的智能视频监控系统，能够实时检测和跟踪视频中的人员和违禁物品，并在目标进入或即将进入安全区域时发出警报。系统支持实时视频流显示, 和预警信息展示。

## 系统架构
系统采用模块化设计，主要包含以下核心模块：
- 视频处理模块：负责视频流的获取和处理
- 目标检测模块：基于YOLO的目标检测和跟踪
- 日志管理模块：处理系统预警时间和状态显示
- Web服务器模块：提供视频流访问接口
- 工具模块：提供通用功能支持·

## 模块说明

### 1. 视频处理模块 (video_processor.py)
负责视频流的处理和目标跟踪的核心模块。

#### 主要功能：
- 摄像头初始化和配置
- 视频帧处理
- 目标检测和跟踪
- 安全区域监控
- 状态更新和日志记录

#### 核心类：
- `VideoProcessor`：视频处理主类
  - 初始化摄像头
  - 设置安全区域
  - 处理视频帧
  - 更新安全状态

### 2. 目标检测模块 (yolo/yolo_infer.py)
基于YOLO的目标检测实现。

#### 主要功能：
- YOLO模型加载和初始化
- 目标检测和跟踪
- 目标类别管理
- 置信度处理

#### 核心类：
- `YOLODetector`：YOLO检测器类
  - 模型初始化
  - 目标检测
  - 跟踪ID管理

### 3. 日志管理模块 (logger.py)
处理系统日志和状态显示。

#### 主要功能：
- 日志面板创建和更新
- 日志文件记录
- 中文文字渲染
- 状态显示管理

#### 核心类：
- `LogManager`：日志管理类
  - 日志记录
  - 状态更新
  - 面板显示

### 4. Web服务器模块 (server.py)
提供视频流访问接口。

#### 主要功能：
- Flask服务器配置
- 视频流生成
- 路由管理
- 实时视频传输

#### 核心类：
- `VideoServer`：视频服务器类
  - 服务器初始化
  - 视频流生成
  - 路由设置

### 5. 工具模块 (utils.py)
提供通用功能支持。

#### 主要功能：
- 边界框绘制
- 轨迹绘制
- 安全区域检测
- 距离计算

#### 核心函数：
- `draw_bbox`：绘制边界框
- `draw_trajectory`：绘制轨迹
- `is_in_safe_zone`：安全区域检测
- `calculate_distance`：距离计算

### 6. 配置模块 (config.py)
系统配置管理。

#### 主要配置项：
- 屏幕配置
- 安全区域配置
- 日志配置
- 服务器配置

### 7. 主程序模块 (main.py)
系统入口和核心流程控制。

#### 主要功能：
- 组件初始化
- 信号处理
- 线程管理
- 资源清理

#### 核心类：
- `FrameManager`：帧管理单例类
  - 当前帧管理
  - 摄像头管理

## 系统流程
1. 系统启动
   - 加载配置
   - 初始化组件
   - 启动视频处理线程
   - 启动Web服务器

2. 视频处理
   - 获取视频帧
   - 目标检测和跟踪
   - 安全区域监控
   - 状态更新

3. 日志记录
   - 更新日志面板
   - 记录日志文件
   - 显示状态信息

4. 视频流传输
   - 生成视频流
   - 通过Web接口传输
   - 客户端显示

## 依赖项
- OpenCV
- NumPy
- Pillow
- Flask
- Ultralytics (YOLO)
- 其他依赖见requirements.txt

## 使用说明
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行系统：
```bash
python main.py
```

3. 访问视频流：
- 打开浏览器访问：`http://localhost:5000/video_feed`

## 注意事项
1. 确保已安装所有必要的依赖
2. 确保有可用的摄像头设备
3. 确保模型文件存在于正确位置
4. 系统需要Python 3.7+环境 


## 核心算法

### 1. 目标检测算法
系统采用YOLO（You Only Look Once）目标检测算法，具体实现：
- 使用YOLO11作为基础检测模型
- 支持多类别目标检测，包括人员和违禁物品
- 使用置信度阈值（0.3）过滤低置信度检测结果
- 采用置信度平滑处理，减少检测框闪烁
- 支持实时目标检测和跟踪

### 2. 目标跟踪算法
系统使用自定义的SimpleTracker进行目标跟踪，主要特点：

#### 核心参数：
- 最大轨迹长度：60帧
- 最大未匹配帧数：55帧
- 最大匹配距离：50像素
- 轨迹匹配权重：距离(0.3) + IOU(0.7)

#### 跟踪流程：
1. 目标匹配
   - 计算检测框与现有轨迹的中心点距离
   - 计算检测框与现有轨迹的IOU
   - 综合距离和IOU得分进行最优匹配
   - 考虑目标类别一致性

2. 轨迹管理
   - 维护目标ID和类别信息
   - 使用deque存储历史轨迹（最大60帧）
   - 计算目标运动速度
   - 预测目标未来位置

3. 状态更新
   - 更新匹配成功的轨迹
   - 创建新检测目标的轨迹
   - 处理未匹配轨迹的missed计数
   - 删除长时间未匹配的轨迹

#### 关键算法实现：
```python
# 速度计算
def calculate_velocity(self, trajectory):
    if len(trajectory) < 2:
        return (0, 0)
    current = trajectory[-1]
    previous = trajectory[-2]
    current_center = self.calculate_center(current)
    previous_center = self.calculate_center(previous)
    dt = time.time() - self.last_time
    vx = (current_center[0] - previous_center[0]) / dt
    vy = (current_center[1] - previous_center[1]) / dt
    return (vx, vy)

# 未来位置预测
def predict_future_position(self, current_bbox, velocity, seconds=5):
    center_x = (current_bbox[0] + current_bbox[2]) // 2
    center_y = (current_bbox[1] + current_bbox[3]) // 2
    future_x = center_x + velocity[0] * seconds
    future_y = center_y + velocity[1] * seconds
    width = current_bbox[2] - current_bbox[0]
    height = current_bbox[3] - current_bbox[1]
    return (
        int(future_x - width/2),
        int(future_y - height/2),
        int(future_x + width/2),
        int(future_y + height/2)
    )
```

### 3. 安全区域预警算法
系统实现了基于轨迹预测的安全区域预警机制：

#### 预警流程：
1. 轨迹预测
   - 基于历史轨迹计算目标速度
   - 预测目标5秒后的位置
   - 保持目标尺寸不变，仅预测中心点位置

2. 预警判断
   - 检查目标当前位置是否在安全区域内
   - 检查预测位置是否在安全区域内
   - 根据检测结果更新预警状态

3. 预警处理
   - 更新日志面板显示
   - 记录预警信息到日志文件
   - 触发相应的处理流程

#### 预警逻辑：
```python
# 安全区域判断
def is_in_safe_zone(bbox, safe_zone):
    x, y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    return (safe_zone[0] <= x <= safe_zone[2] and 
            safe_zone[1] <= y <= safe_zone[3])

# 预警状态更新
if current_in_zone:
    update_status("已进入安全区域")
elif predicted_in_zone:
    update_status("即将进入安全区域")
```

## 性能优化
1. 检测优化
   - 使用置信度平滑减少误检
   - 优化检测阈值提高准确率
   - 支持模型量化提升速度

2. 跟踪优化
   - 使用运动预测减少跟踪丢失
   - 优化ID分配策略
   - 处理目标遮挡情况

3. 预警优化
   - 多级预警机制
   - 预警去重处理
   - 预警时间窗口控制