import os
import argparse
from datetime import datetime
import pytz
import tensorflow as tf

from data import video_count_dataset
from model import mobile_net_v2_cam_video_model
from callbacks import save_callback

def train(train_tfrecords_dir_path, test_tfrecords_dir_path, classes_txt_path, epochs, step_size, batch_size, image_size, skip_frame_ratio,
          test_sample_num, output_dir_path):
    with open(classes_txt_path, 'r') as f:
        classes = f.readlines()
    classes = [label.strip() for label in classes]

    # train
    TrainDataset = type(f'TrainDataset', (video_count_dataset.VideoCountDataset,), dict())
    train_dataset = TrainDataset(train_tfrecords_dir_path, classes, skip_frame_ratio=skip_frame_ratio,
                                 input_size=(image_size, image_size, 3))
    train_dataset = train_dataset.padded_batch(batch_size=batch_size, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.int32)))
    # valid
    ValidDataset = type(f'ValidDataset', (video_count_dataset.VideoCountDataset,), dict())
    valid_dataset = ValidDataset(test_tfrecords_dir_path, classes, skip_frame_ratio=skip_frame_ratio,
                                 input_size=(image_size, image_size, 3), max_sample_num=test_sample_num)
    valid_data = video_count_dataset.VideoCountDataset.get_all_data(valid_dataset)

    # train_valid
    TrainValidDataset = type(f'TrainValidDataset', (video_count_dataset.VideoCountDataset,), dict())
    train_valid_dataset = TrainValidDataset(train_tfrecords_dir_path, classes, skip_frame_ratio=skip_frame_ratio,
                                 input_size=(image_size, image_size, 3), max_sample_num=test_sample_num)
    train_valid_data = video_count_dataset.VideoCountDataset.get_all_data(train_valid_dataset)

    # prepare model
    train_model, save_model = mobile_net_v2_cam_video_model.prepare(len(classes), image_size, fine=True)
    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.Huber())

    # prepare callback
    save_model_dir_path = os.path.join(output_dir_path,
                                       f'{datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d-%H-%M-%S")}')
    prefix = f'step-{step_size}_batch-{batch_size}'
    callback = save_callback.SaveCallback(save_model=save_model, save_model_dir_path=save_model_dir_path, prefix=prefix, valid_data=valid_data, train_valid_data=train_valid_data)

    train_model.fit_generator(train_dataset, steps_per_epoch=step_size,
                        epochs=epochs,
                        validation_data=valid_data,
                        callbacks=[callback])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pb')
    parser.add_argument('--train_tfrecords_dir_path', type=str, default='/media/kentaro/dataset/.vaik-mnist-video-count-dataset/train_tfrecords')
    parser.add_argument('--test_tfrecords_dir_path', type=str, default='/media/kentaro/dataset/.vaik-mnist-video-count-dataset/valid_tfrecords')
    parser.add_argument('--classes_txt_path', type=str, default='/media/kentaro/dataset/.vaik-mnist-video-count-dataset/train_tfrecords/classes.txt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--skip_frame_ratio', nargs='+', type=int, default=(1, 2, 4))
    parser.add_argument('--valid_sample_num', type=int, default=100)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-video-count-pb-trainer/output_model')
    args = parser.parse_args()

    args.train_tfrecords_dir_path = os.path.expanduser(args.train_tfrecords_dir_path)
    args.test_tfrecords_dir_path = os.path.expanduser(args.test_tfrecords_dir_path)
    args.classes_txt_path = os.path.expanduser(args.classes_txt_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(args.output_dir_path, exist_ok=True)

    train(args.train_tfrecords_dir_path, args.test_tfrecords_dir_path, args.classes_txt_path, args.epochs, args.step_size,
          args.batch_size, args.image_size, args.skip_frame_ratio,
          args.valid_sample_num, args.output_dir_path)