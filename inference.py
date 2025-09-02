import torch
import cv2
import logging
import time
import collections
from typing import Optional, List, Tuple
import numpy as np
from torchvision import transforms


class RealTimeMoViNetDetector:
    """
    实时MoViNet视频流检测器
    适用于RTSP流监控，支持滑动窗口检测，资源优化
    """

    def __init__(self,
                 model_path: str,
                 model_config: dict,
                 window_size: int = 128,
                 window_overlap: int = 64,
                 detection_threshold: float = 0.8,
                 frame_buffer_size: int = 200,
                 device: str = 'cpu',
                 log_level: int = logging.INFO):
        """
        初始化实时检测器

        Args:
            model_path: 模型权重文件路径
            model_config: 模型配置参数
            window_size: 检测窗口大小（帧数）
            window_overlap: 窗口重叠大小（帧数）
            detection_threshold: 违规检测阈值
            frame_buffer_size: 帧缓冲区大小
            device: 运算设备 ('cpu' 或 'cuda')
            log_level: 日志级别
        """
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.detection_threshold = detection_threshold
        self.frame_buffer_size = frame_buffer_size
        self.device = device

        # 初始化日志
        self.logger = self._setup_logger(log_level)

        # 初始化模型
        self.model = self._load_model(model_path, model_config)

        # 初始化帧缓冲区（使用循环缓冲区）
        self.frame_buffer = collections.deque(maxlen=frame_buffer_size)

        # 初始化预处理
        self.transform = self._setup_transform()

        # 检测状态
        self.is_initialized = True
        self.detection_count = 0
        self.last_detection_time = 0

        self.logger.info(f"RealTimeMoViNetDetector initialized with window_size={window_size}, "
                         f"overlap={window_overlap}, threshold={detection_threshold}")

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger('MoViNetDetector')
        logger.setLevel(log_level)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_model(self, model_path: str, model_config: dict):
        """加载模型"""
        try:
            from net.movinet import MoViNet
            from net.cfg import build_movinet_a0_cfg

            self.logger.info("Loading model...")

            # 构建模型配置
            cfg = build_movinet_a0_cfg() if not model_config else model_config

            # 创建模型（因果模式）
            model = MoViNet(
                cfg,
                causal=True,
                pretrained=False,
                num_classes=2,
                conv_type="2plus1d",
                tf_like=True
            )

            # 加载权重
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()

            self.logger.info(f"Model loaded successfully from {model_path}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def _setup_transform(self):
        """设置图像预处理"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """预处理单帧"""
        try:
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 应用变换
            processed_frame = self.transform(frame_rgb)
            return processed_frame
        except Exception as e:
            self.logger.error(f"Frame preprocessing failed: {str(e)}")
            raise

    def _detect_in_window(self, frames: List[torch.Tensor]) -> Tuple[bool, float]:
        """
        在指定窗口中检测违规行为

        Args:
            frames: 窗口内的帧列表

        Returns:
            (是否违规, 置信度)
        """
        try:
            if len(frames) < self.window_size:
                return False, 0.0

            # 构造输入张量
            clip = torch.stack(frames[:self.window_size])
            clip = clip.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
            clip = clip.to(self.device)

            # 模型推理
            with torch.no_grad():
                self.model.clean_activation_buffers()  # 确保状态干净
                output = self.model(clip)
                prob = torch.softmax(output, dim=1)

                # 假设索引1为违规类
                violation_prob = prob[0][1].item()
                is_violation = violation_prob > self.detection_threshold

                return is_violation, violation_prob

        except Exception as e:
            self.logger.error(f"Detection in window failed: {str(e)}")
            return False, 0.0

    def process_frame(self, frame: np.ndarray) -> Optional[dict]:
        """
        处理单帧

        Args:
            frame: 输入帧 (BGR格式)

        Returns:
            检测结果字典或None
        """
        try:
            if not self.is_initialized:
                self.logger.warning("Detector not initialized")
                return None

            # 预处理帧
            processed_frame = self._preprocess_frame(frame)
            self.frame_buffer.append(processed_frame)

            # 检查是否需要进行检测
            detection_result = None
            if len(self.frame_buffer) >= self.window_size:
                # 进行检测
                is_violation, confidence = self._detect_in_window(list(self.frame_buffer))

                if is_violation:
                    self.detection_count += 1
                    self.last_detection_time = time.time()

                    detection_result = {
                        'timestamp': time.time(),
                        'is_violation': True,
                        'confidence': confidence,
                        'frame_count': len(self.frame_buffer),
                        'detection_id': self.detection_count
                    }

                    self.logger.info(f"🚨 Violation detected! Confidence: {confidence:.4f}")

                # 滑动窗口：移除部分帧
                for _ in range(self.window_size - self.window_overlap):
                    if self.frame_buffer:
                        self.frame_buffer.popleft()

            return detection_result

        except Exception as e:
            self.logger.error(f"Frame processing failed: {str(e)}")
            return None

    def process_rtsp_stream(self, rtsp_url: str, reconnect_attempts: int = 5) -> None:
        """
        处理RTSP流

        Args:
            rtsp_url: RTSP流地址
            reconnect_attempts: 重连尝试次数
        """
        cap = None
        attempt = 0

        while attempt < reconnect_attempts:
            try:
                self.logger.info(f"Connecting to RTSP stream: {rtsp_url} (attempt {attempt + 1})")
                cap = cv2.VideoCapture(rtsp_url)

                if not cap.isOpened():
                    raise Exception("Failed to open RTSP stream")

                self.logger.info("RTSP stream connected successfully")
                attempt = 0  # 重置重连计数

                frame_count = 0
                start_time = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("Failed to read frame from RTSP stream")
                        break

                    frame_count += 1

                    # 处理帧
                    detection_result = self.process_frame(frame)

                    # 记录处理统计
                    if frame_count % 100 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        self.logger.info(f"Processed {frame_count} frames, FPS: {fps:.2f}")

                    # 可以在这里添加其他处理逻辑
                    if detection_result:
                        self._handle_detection(detection_result)

            except Exception as e:
                self.logger.error(f"RTSP stream error: {str(e)}")
                attempt += 1

                if attempt < reconnect_attempts:
                    self.logger.info(f"Reconnecting in 5 seconds... ({attempt}/{reconnect_attempts})")
                    time.sleep(5)
                else:
                    self.logger.error("Max reconnection attempts reached")
                    break

            finally:
                if cap:
                    cap.release()
                    self.logger.info("RTSP stream released")

    def _handle_detection(self, detection_result: dict) -> None:
        """
        处理检测结果（可重写以实现自定义报警逻辑）

        Args:
            detection_result: 检测结果
        """
        # 这里可以实现报警逻辑，如：
        # - 发送邮件/短信
        # - 保存违规视频片段
        # - 触发外部API
        self.logger.info(f"Handling detection: {detection_result}")

    def get_status(self) -> dict:
        """获取检测器状态"""
        return {
            'initialized': self.is_initialized,
            'buffer_size': len(self.frame_buffer),
            'detection_count': self.detection_count,
            'last_detection_time': self.last_detection_time,
            'device': self.device
        }

    def reset(self) -> None:
        """重置检测器状态"""
        self.frame_buffer.clear()
        self.detection_count = 0
        self.last_detection_time = 0
        if self.model:
            self.model.clean_activation_buffers()
        self.logger.info("Detector reset completed")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    config = {
        'model_path': 'checkpoints/movinet_best.pth',
        'model_config': None,  # 使用默认配置
        'window_size': 128,
        'window_overlap': 64,  # 64帧（50%重叠确保不漏检）
        'detection_threshold': 0.5,
        'frame_buffer_size': 200,
        'device': 'cuda',  # 或 'cuda' 如果有GPU
        'log_level': logging.INFO
    }

    try:
        # 创建检测器实例
        detector = RealTimeMoViNetDetector(**config)

        # 处理RTSP流
        rtsp_url = "/home/kend/Guanxin/work/workspace/movinet-pytorch/tt2.mp4"
        detector.process_rtsp_stream(rtsp_url)

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
