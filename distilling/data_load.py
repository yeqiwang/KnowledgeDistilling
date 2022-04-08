"""
加载数据
"""

import tensorflow as tf
from distilling import train_config as tc


def _get_fashion_mnist():
    """
    加载fashion_mnist数据集
    :return:
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 图像归一化
    train_images = train_images / 255
    test_images = test_images / 255

    # reshape增加维度
    train_images = tf.reshape(train_images, shape=train_images.shape + (1, ))
    test_images = tf.reshape(test_images, shape=test_images.shape + (1, ))

    # 标签one-hot编码
    train_labels = tf.one_hot(train_labels, depth=10)
    test_labels = tf.one_hot(test_labels, depth=10)

    return (train_images, train_labels), (test_images, test_labels)


def get_dataset():
    """
    获取dataset实例
    :return:
    """
    (train_images, train_labels), (test_images, test_labels) = _get_fashion_mnist()
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.cache().shuffle(buffer_size=train_images.shape[0]).batch(tc.DATA_CONFIG['batch_size'])

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.cache().batch(tc.DATA_CONFIG['batch_size'])
    return train_ds, test_ds
