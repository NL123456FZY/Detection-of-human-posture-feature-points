import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class PoseDetector:
    def __init__(self):
        try:
            print("正在初始化姿态检测器...")
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.prev_positions = None  # 用于存储上一帧的关键点位置
            self.fall_detected = False
            self.fall_threshold = 0.5  # 跌倒检测阈值
            self.fall_count = 0  # 添加跌倒计数器
            self.fall_duration = 0  # 添加跌倒持续时间
            self.status_text = "正常"  # 添加状态文本
            self.status_color = (0, 255, 0)  # 默认绿色
            
            # 修改字体路径为系统字体的完整路径
            self.font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows系统黑体字体
            # 如果上面的路径不存在，可以尝试以下备选字体：
            # self.font_path = "C:/Windows/Fonts/msyh.ttc"   # 微软雅黑
            # self.font_path = "C:/Windows/Fonts/simsun.ttc" # 宋体
            self.font_size = 30
            self.small_font_size = 24
            
            # 测试字体文件是否存在
            if not os.path.exists(self.font_path):
                print(f"警告：找不到字体文件 {self.font_path}")
                # 尝试其他字体
                backup_fonts = [
                    "C:/Windows/Fonts/simhei.ttf",
                    "C:/Windows/Fonts/msyh.ttc",
                    "C:/Windows/Fonts/simsun.ttc",
                    "C:/Windows/Fonts/simkai.ttf"
                ]
                for font in backup_fonts:
                    if os.path.exists(font):
                        self.font_path = font
                        print(f"使用备选字体：{font}")
                        break
                else:
                    raise Exception("未找到可用的中文字体文件")
            
            print("姿态检测器初始化成功！")
            
            # 添加跌倒检测相关的参数
            self.fall_history = []  # 存储最近几帧的状态
            self.history_size = 10  # 历史状态窗口大小
            self.fall_confidence = 0  # 跌倒置信度
            self.min_confidence = 0.7  # 最小置信度阈值
            self.stable_frames = 0  # 稳定帧计数
            self.min_stable_frames = 5  # 最小稳定帧数
            
            # 添加警告框闪烁相关的参数
            self.warning_alpha = 0.4  # 警告框透明度
            self.blink_speed = 30    # 闪烁速度（帧数）
            self.line_length = 30    # 角标长度
            self.line_thickness = 4  # 线条粗细
        except Exception as e:
            print(f"初始化失败，错误信息：{str(e)}")
            raise

    def find_pose(self, img, draw=True):
        """检测并绘制人体姿态关键点"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        
        if self.results.pose_landmarks:
            if draw:
                # 绘制骨骼连接线
                self.mp_draw.draw_landmarks(
                    img, 
                    self.results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    # 修改关键点的样式
                    self.mp_draw.DrawingSpec(
                        color=(0, 0, 255),  # 红色关键点
                        thickness=4,
                        circle_radius=4
                    ),
                    # 修改连接线的样式
                    self.mp_draw.DrawingSpec(
                        color=(0, 255, 0),  # 绿色连接线
                        thickness=2,
                        circle_radius=2
                    )
                )
                
                # 添加关键点标注
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # 绘制更大的关键点圆圈
                    cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)
                    
                    # 添加关键点编号（可选）
                    # cv2.putText(img, str(id), (cx-10, cy-10), 
                    #           cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                    
                    # 标注重要关键点
                    important_points = {
                        0: "头部",
                        11: "左肩",
                        12: "右肩",
                        23: "左髋",
                        24: "右髋",
                        25: "左膝",
                        26: "右膝",
                        27: "左踝",
                        28: "右踝"
                    }
                    
                    if id in important_points:
                        # 为重要关键点添加标注
                        img = self._cv2AddChineseText(
                            img,
                            important_points[id],
                            (cx + 10, cy - 10),
                            (255, 255, 255),
                            20  # 较小的字体大小
                        )
                        # 为重要关键点绘制特殊标记
                        cv2.circle(img, (cx, cy), 8, (0, 255, 255), cv2.FILLED)
                        cv2.circle(img, (cx, cy), 10, (0, 255, 255), 2)
        
        return img
    
    def find_position(self, img):
        """获取所有关键点的位置"""
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        return lm_list

    def detect_fall(self, img):
        """检测跌倒行为"""
        if not self.results.pose_landmarks:
            self.fall_count = 0
            self.status_text = "未检测到人体"
            self.status_color = (0, 255, 255)
            self._draw_status(img)
            return False, img

        # 获取关键点和图像尺寸
        landmarks = self.results.pose_landmarks.landmark
        h, w, _ = img.shape
        
        # 1. 计算更多的身体特征
        # 躯干角度
        shoulder_mid_x = (landmarks[11].x + landmarks[12].x) / 2
        shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2
        hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2
        hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2
        
        trunk_angle = abs(np.degrees(np.arctan2(
            shoulder_mid_x - hip_mid_x,
            shoulder_mid_y - hip_mid_y
        )))
        
        # 身体高度比例（使用多个关键点）
        head_y = landmarks[0].y
        ankle_y = (landmarks[27].y + landmarks[28].y) / 2
        body_height = abs(head_y - ankle_y) * h
        height_ratio = body_height / h
        
        # 头部位置相对于臀部的高度
        head_hip_height = landmarks[0].y - hip_mid_y
        
        # 计算膝盖位置
        knee_y = (landmarks[25].y + landmarks[26].y) / 2
        knee_hip_ratio = abs(knee_y - hip_mid_y) / abs(ankle_y - hip_mid_y)
        
        # 2. 计算速度和加速度
        current_pos = np.array([shoulder_mid_x, shoulder_mid_y])
        velocity = 0
        if self.prev_positions is not None:
            velocity = np.linalg.norm(current_pos - self.prev_positions)
        self.prev_positions = current_pos

        # 3. 评估跌倒可能性（调整权重）
        fall_score = 0
        
        # 躯干角度判断（更严格的条件）
        if trunk_angle > 65:  # 更大的角度阈值
            fall_score += 0.35
        elif trunk_angle > 45:
            fall_score += 0.15
            
        # 身体高度判断（更精确的比例）
        if height_ratio < 0.35:  # 更严格的高度比例
            fall_score += 0.35
        elif height_ratio < 0.45:
            fall_score += 0.15
            
        # 头部位置判断
        if head_hip_height > 0.1:  # 头部明显低于臀部
            fall_score += 0.2
            
        # 膝盖位置判断（新增）
        if knee_hip_ratio < 0.3:  # 膝盖异常弯曲
            fall_score += 0.1
            
        # 突然运动判断（调整阈值）
        if velocity > 0.15:  # 更大的速度阈值
            fall_score += 0.1
            
        # 4. 更新历史记录（增加平滑处理）
        self.fall_history.append(fall_score)
        if len(self.fall_history) > self.history_size:
            self.fall_history.pop(0)
            
        # 使用加权平均计算置信度（最近的帧权重更大）
        weights = np.linspace(0.5, 1.0, len(self.fall_history))
        self.fall_confidence = np.average(self.fall_history, weights=weights)
        
        # 5. 判断是否跌倒（更严格的条件）
        is_falling = False
        if self.fall_confidence > self.min_confidence:
            self.stable_frames += 1
            if self.stable_frames >= self.min_stable_frames:
                is_falling = True
                self.fall_count += 1
                self.fall_duration = round(self.fall_count / 30, 1)
                self.status_text = "异常！检测到跌倒！"
                self.status_color = (0, 0, 255)
        else:
            # 快速恢复正常状态
            if self.fall_confidence < 0.3:  # 添加快速恢复条件
                self.stable_frames = 0
                self.fall_count = 0
                self.fall_duration = 0
                self.status_text = "正常"
                self.status_color = (0, 255, 0)

        # 显示调试信息
        self._draw_status(img)
        if is_falling:
            self._draw_fall_warning(img)
            
        debug_info = (f"置信度: {self.fall_confidence:.2f} "
                     f"角度: {trunk_angle:.1f}° "
                     f"高度比: {height_ratio:.2f}")
        img = self._cv2AddChineseText(img, debug_info, (10, 70), 
                                    (255, 255, 255), self.small_font_size)
        
        return is_falling, img

    def _cv2AddChineseText(self, img, text, position, textColor=(255, 255, 255), textSize=30):
        """
        在图片上添加中文文字
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 加载字体
        fontStyle = ImageFont.truetype(self.font_path, textSize, encoding="utf-8")
        # 绘制文字
        draw.text(position, text, textColor, font=fontStyle)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def _draw_status(self, img):
        """绘制状态信息"""
        h, w = img.shape[:2]
        panel_height = 80
        panel_y = h - panel_height  # 将面板移到底部
        
        # 创建半透明的深色背景
        overlay = img[panel_y:h, 0:w].copy()
        cv2.rectangle(img, (0, panel_y), (w, h), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.2, img[panel_y:h, 0:w], 0.8, 0, img[panel_y:h, 0:w])

        # 显示状态信息
        img = self._cv2AddChineseText(img, f"状态: {self.status_text}", (10, panel_y + 10), 
                                    self.status_color, self.font_size)

    def _draw_fall_warning(self, img):
        """绘制跌倒警告"""
        # 获取图像尺寸
        h, w = img.shape[:2]
        
        # 添加半透明的红色警告背景
        overlay = img.copy()
        
        # 扩大警告框的尺寸，使其更醒目
        warning_x1, warning_y1 = 10, 100
        warning_x2, warning_y2 = w-10, 200
        
        # 绘制半透明红色背景
        cv2.rectangle(overlay, 
                     (warning_x1, warning_y1), 
                     (warning_x2, warning_y2), 
                     (0, 0, 255), -1)
        # 调整透明度（0.4表示40%的红色）
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # 修改警告文字
        img = self._cv2AddChineseText(img, 
                                    "⚠️检测到跌倒行为！", 
                                    (warning_x1 + 10, warning_y1 + 10), 
                                    (255, 255, 255), 
                                    self.font_size)
        
        # 添加持续时间信息
        img = self._cv2AddChineseText(img, 
                                    f"跌倒持续时间: {self.fall_duration}秒", 
                                    (warning_x1 + 10, warning_y1 + 50), 
                                    (255, 255, 255), 
                                    self.small_font_size)
        
        # 添加��烁边框效果
        if self.fall_count % 30 < 15:  # 每秒闪烁一次
            # 主边框
            cv2.rectangle(img, 
                         (warning_x1-5, warning_y1-5), 
                         (warning_x2+5, warning_y2+5), 
                         (0, 0, 255), 3)
            
            # 添加双边框效果
            cv2.rectangle(img, 
                         (warning_x1-8, warning_y1-8), 
                         (warning_x2+8, warning_y2+8), 
                         (255, 255, 255), 2)  # 白色外边框
            
            # 添加四角标记
            line_length = 30  # 角标长度
            thickness = 4     # 线条粗细
            color = (0, 0, 255)  # 红色
            
            # 左上角
            cv2.line(img, (warning_x1-5, warning_y1-5), (warning_x1-5+line_length, warning_y1-5), color, thickness)
            cv2.line(img, (warning_x1-5, warning_y1-5), (warning_x1-5, warning_y1-5+line_length), color, thickness)
            
            # 右上角
            cv2.line(img, (warning_x2+5, warning_y1-5), (warning_x2+5-line_length, warning_y1-5), color, thickness)
            cv2.line(img, (warning_x2+5, warning_y1-5), (warning_x2+5, warning_y1-5+line_length), color, thickness)
            
            # 左下角
            cv2.line(img, (warning_x1-5, warning_y2+5), (warning_x1-5+line_length, warning_y2+5), color, thickness)
            cv2.line(img, (warning_x1-5, warning_y2+5), (warning_x1-5, warning_y2+5-line_length), color, thickness)
            
            # 右下角
            cv2.line(img, (warning_x2+5, warning_y2+5), (warning_x2+5-line_length, warning_y2+5), color, thickness)
            cv2.line(img, (warning_x2+5, warning_y2+5), (warning_x2+5, warning_y2+5-line_length), color, thickness)
        
        return img

def main():
    try:
        print("正在启动人体姿态检测系统...")
        print("按 'q' 或 'ESC' 键退出程序")
        
        # 测试摄像头是否可用
        cap = cv2.VideoCapture("D:\桌面\失败的MAN.mp4")
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return
            
        print("摄像头初始化成功！")
        detector = PoseDetector()
        
        while True:
            success, img = cap.read()
            if not success:
                print("无法获取摄像头画面")
                break
                
            # 检测姿态
            img = detector.find_pose(img)
            
            # 获取关键点位置
            lm_list = detector.find_position(img)
            
            # 检测跌倒
            is_falling, img = detector.detect_fall(img)
            if is_falling:
                print(f"检测到跌倒行为！持续时间：{detector.fall_duration}秒")
            
            # 显示画面
            cv2.imshow("人体姿态检测", img)
            
            # 退出检测
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("正在退出程序...")
                break
                
    except Exception as e:
        print(f"程序运行出错：{str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("程序已结束")

if __name__ == "__main__":
    main()