# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/30 11:16
@Author  : Kend
@FileName: logger
@Software: PyCharm
@modifier:
"""
import logging
import os

def setup_logger(log_dir):
    """设置日志记录器"""
    log_file = os.path.join(log_dir, 'training.log')

    # 创建logger
    logger = logging.getLogger('movinet_trainer')
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

