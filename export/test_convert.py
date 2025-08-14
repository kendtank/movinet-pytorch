# -*- coding: utf-8 -*-
"""
@Time    : 2025/8/1 14:34
@Author  : Kend
@FileName: test_convert
@Software: PyCharm
@modifier:
"""

""" 简单测试 tf-lite 测试 """


import tensorflow as tf
import numpy as np
import os


def create_simple_tflite_model():
    """
    创建一个简单的TensorFlow Lite模型用于测试
    """
    print("创建测试用TensorFlow Lite模型...")

    # 创建一个简单的模型
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3, 16, 224, 224)),
        tf.keras.layers.Conv3D(8, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling3D(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 保存为SavedModel
    model.save('test_tf_model')
    print("测试TensorFlow模型已保存")

    # 转换为TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_saved_model('test_tf_model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open('test_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("测试TensorFlow Lite模型已创建: test_model.tflite")

    # 验证
    interpreter = tf.lite.Interpreter(model_path='test_model.tflite')
    interpreter.allocate_tensors()

    print("测试模型验证通过")


if __name__ == "__main__":
    create_simple_tflite_model()
