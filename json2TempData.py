import json
import argparse
import os
import numpy as np


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def norm_kp(h, w, bbox, kps):
    norm_kps = []
    for i, v in enumerate(kps):
        if i % 3 == 0:
            norm_kps.append((v-bbox[0]) / w)
        elif i % 3 == 1:
            norm_kps.append((v-bbox[1]) / h)
        elif i % 3 == 2:
            # norm_kps.append(v)
            if v == 0:
                norm_kps[i-1] = 0
                norm_kps[i-2] = 0
    return norm_kps


def extract_info(json_data, data_txt, frame_length, label_txt, label_num, sample_interval):
    norm_kps = []
    for key, value in json_data.items():

        frame = value["frame"]
        objects = value["objects"]
        img_height = value["img_height"]
        img_width = value["img_width"]
        # norm_kps = []
        frame_list = []
        for obj in objects:
            bbox = obj["bbox"]
            # score = obj["score"]
            # idx = obj["idx"]
            box_h = bbox[3] - bbox[1]
            box_w = bbox[2] - bbox[0]
            kps = obj["kps"]
            norm_kps.append(norm_kp(box_h, box_w, bbox, kps))
            frame_list.append(frame)

    sample_num = len(norm_kps) // sample_interval - 1
    for i in range(sample_num):
        label_idx = 0
        target_kps = norm_kps[i*sample_interval: (i*sample_interval) + frame_length]
        target_kps_flat = np.array(target_kps).flatten()
        # target_str_kps = [lambda x: float(x), ]
        # str_kps = [l.split(",") for l in target_kps]

        with open(data_txt, "a") as writer:
            kps_str = '\t'.join(f"{k:.6f}" for k in target_kps_flat)
            label_idx += 1
            # else:
            writer.write(f"{kps_str}\n")

        with open(label_txt, "a") as lb:
            for idx in range(label_idx):
                lb.write(f"{label_num}\n")


def process_files(input_dir, data_output_dir, label_output_dir, frame_length, label_num, sample_interval):
    for file in os.listdir(input_dir):
        if file.endswith(".json"):
            json_file = os.path.join(input_dir, file)
            basename = os.path.splitext(file)[0]
            data_txt = os.path.join(data_output_dir, f"{basename}.txt")
            label_txt = os.path.join(label_output_dir, f"{basename}_label.txt")

            json_data = read_json_file(json_file)
            extract_info(json_data, data_txt, frame_length, label_txt, label_num, sample_interval)


def main(args):
    input_dir = args.input_dir
    data_output_dir = args.data_output_dir
    label_output_dir = args.label_output_dir
    label_num = args.label_num
    frame_length = args.frame_length
    sample_interval = args.sample_interval

    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    process_files(input_dir, data_output_dir, label_output_dir, frame_length, label_num, sample_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing JSON files')
    parser.add_argument('--data_output_dir', type=str, required=True, help='Path to the output directory for data TXT files')
    parser.add_argument('--label_output_dir', type=str, required=True, help='Path to the output directory for label TXT files')
    parser.add_argument('--label_num', type=int, default=0, help='Label number for the current action')
    parser.add_argument('--frame_length', type=int, default=5, help='Interval for saving frames')
    parser.add_argument('--sample_interval', type=int, default=4, help='Interval for saving frames')

    args = parser.parse_args()
    main(args)
