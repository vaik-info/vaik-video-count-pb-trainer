import random
import glob
import os

import tqdm
import numpy as np
import tensorflow as tf


class VideoCountDataset:
    output_signature = None
    classes = None
    frame_num = None
    skip_frame_ratio = None
    input_size = None
    max_sample_num = None
    parsed_dataset = None

    def __new__(cls, tfrecords_dir_path, classes, skip_frame_ratio=(1, 2), input_size=(320, 320, 3), max_sample_num=None):
        cls.classes = classes
        cls.skip_frame_ratio = skip_frame_ratio
        cls.input_size = input_size
        cls.max_sample_num = max_sample_num
        cls.parsed_dataset = cls.__load_tfrecords(tfrecords_dir_path)
        cls.output_signature = (
            (tf.TensorSpec(name=f'video', shape=(None,) + input_size, dtype=tf.uint8),
                                tf.TensorSpec(name=f'count', shape=(len(classes)), dtype=tf.int32),
                                tf.TensorSpec(name=f'length', shape=(), dtype=tf.int32)),
            )


        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=cls.output_signature
        )
        return dataset

    @classmethod
    def _generator(cls):
        step_index = 0
        while True:
            for parsed_data in cls.parsed_dataset:
                step_index += 1
                if cls.max_sample_num is not None and cls.max_sample_num < step_index:
                    return
                video, count = parsed_data
                video_array = np.zeros((video.shape[0], ) + cls.input_size, dtype=np.uint8)
                random_skip_frame_ratio = random.choice(cls.skip_frame_ratio)
                video = video[::random_skip_frame_ratio]
                video = tf.image.resize_with_crop_or_pad(video, max(video.shape[1:3]), max(video.shape[1:3]))
                video = tf.image.resize(video, (cls.input_size[0], cls.input_size[1]))
                video_array[:video.shape[0], :, :, :] = video
                yield ((tf.cast(video_array, tf.uint8), tf.cast(count, tf.int32), tf.convert_to_tensor(video.shape[0], dtype=tf.int32)), )

    @classmethod
    def __parse_tfrecord_fn(cls, example):
        feature_description = {
            'video': tf.io.FixedLenFeature([], tf.string),
            'shape': tf.io.FixedLenFeature([4], tf.int64),
            'count': tf.io.FixedLenFeature([len(cls.classes)], tf.int64)
        }
        example = tf.io.parse_single_example(example, feature_description)
        video = tf.io.parse_tensor(example['video'], out_type=tf.uint8)
        shape = example['shape']
        video = tf.reshape(video, shape)
        class_index = example['count']
        return video, class_index

    @classmethod
    def __load_tfrecords(cls, tfrecords_dir_path):
        tfrecords_path_list = glob.glob(os.path.join(tfrecords_dir_path, '*.tfrecords-*'))
        raw_dataset = tf.data.TFRecordDataset(tfrecords_path_list)
        parsed_dataset = raw_dataset.map(cls.__parse_tfrecord_fn)
        return parsed_dataset