import os
import argparse
from tqdm import tqdm
from PIL import Image
from data import video_count_dataset

def create_gif(video, output_file_path, duration=5):
    gif_image_list = []
    for video_index in range(video.shape[0]):
        image = video[video_index].numpy()
        gif_image_list.append(Image.fromarray(image))
    gif_image_list[0].save(output_file_path,
                           save_all=True, append_images=gif_image_list[1:], optimize=False, duration=duration, loop=0)


def dump(tfrecords_dir_path, classes_txt_path, sample_num, image_height, image_width, skip_frame_ratio, output_dir_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.makedirs(output_dir_path, exist_ok=True)
    with open(classes_txt_path, 'r') as f:
        classes = f.readlines()
    classes = [label.strip() for label in classes]

    DumpDataset = type(f'DumpVideoClassificationDataset', (video_count_dataset.VideoCountDataset,), dict())
    dump_dataset = DumpDataset(tfrecords_dir_path, classes, skip_frame_ratio=(skip_frame_ratio, ),
                               input_size=(image_height, image_width, 3), max_sample_num=sample_num)

    for index, data in tqdm(enumerate(dump_dataset)):
        video, count, length = data[0]
        count_str = ""
        for class_index, label in enumerate(classes):
            count_str += f'{label}_{count[class_index]}_'
        count_str = count_str[:-1]
        output_file_path = os.path.join(output_dir_path, f'{count_str}_skipframe-{skip_frame_ratio}_length-{length}_{index:04d}.gif')
        create_gif(video, output_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dump')
    parser.add_argument('--tfrecords_dir_path', type=str, default='/media/kentaro/dataset/.vaik-mnist-video-count-dataset/valid_tfrecords')
    parser.add_argument('--classes_txt_path', type=str, default='/media/kentaro/dataset/.vaik-mnist-video-count-dataset/valid_tfrecords/classes.txt')
    parser.add_argument('--sample_num', type=int, default=100)
    parser.add_argument('--image_height', type=int, default=320)
    parser.add_argument('--image_width', type=int, default=320)
    parser.add_argument('--skip_frame_ratio', type=int, default=2)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-video-count-pb-trainer/dump')
    args = parser.parse_args()

    args.tfrecords_dir_path = os.path.expanduser(args.tfrecords_dir_path)
    args.classes_txt_path = os.path.expanduser(args.classes_txt_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    dump(args.tfrecords_dir_path, args.classes_txt_path, args.sample_num, args.image_height, args.image_width, args.skip_frame_ratio, args.output_dir_path)