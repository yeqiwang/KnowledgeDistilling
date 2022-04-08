"""
构建模型
"""

import tensorflow as tf


def TeacherModel():
    """
    构建教师模型
    :return: 序列化模型
    """
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(10)(x)
    outputs = tf.keras.layers.Softmax()(x)

    return tf.keras.models.Model(inputs, outputs)


def StudentModel(with_softmax=True):
    """
    构建学生模型
    :param: 是否加入softmax
    :return: 序列化模型
    """
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(10)(x)
    if with_softmax:
        x = tf.keras.layers.Softmax()(x)
    outputs = x

    return tf.keras.models.Model(inputs, outputs)