import cv2
import numpy as np
from pathlib import Path


def read_label_file(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = [list(map(float, line.strip().split())) for line in lines]
    return np.array(labels)


def read_cls_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        if content.endswith('\n'):
            content = content[:-2]
    return content.split('\n')


def draw_boxes(img, labels, class_names):
    h, w, _ = img.shape
    for label in labels:
        cls, x_center, y_center, box_w, box_h, *keypoints = label
        cls = int(cls)

        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = class_names[cls]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 2), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw keypoints
        if len(keypoints) % 3 == 0:
            keypoint_radius = 3
            keypoint_colors = [
                (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
                (0, 255, 0), (0, 0, 255), (0, 255, 0), (0, 0, 255), (0, 255, 0), (0, 0, 255),
                (0, 255, 255), (255, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 255), (255, 0, 255)
            ]
            for i in range(0, len(keypoints), 3):
                x = int(keypoints[i] * w)
                y = int(keypoints[i+1] * h)
                visibility = keypoints[i+2]
                if 0 < x < w and 0 < y < h and visibility != 0:
                    keypoint_color = keypoint_colors[i // 3 % len(keypoint_colors)]
                    cv2.circle(img, (x, y), keypoint_radius, keypoint_color, -1)
        else:
            print(f"Warning: Incorrect number of keypoint values for label {label}")

    return img


def visualize_labeled_data(image_folder, label_folder, class_names):
    image_paths = sorted(list(Path(image_folder).glob("*.jpg")))
    label_paths = sorted(list(Path(label_folder).glob("*.txt")))

    for img_path, label_path in zip(image_paths, label_paths):
        img = cv2.imread(str(img_path))
        labels = read_label_file(str(label_path))
        img = draw_boxes(img, labels, class_names)

        cv2.imshow("Labeled Image", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


image_folder = "/media/hkuit164/Backup/yolov7ps/yolov7/datasets/cls_pose/images/train"
label_folder = "/media/hkuit164/Backup/yolov7ps/yolov7/datasets/cls_pose/labels/train"
class_path = "/media/hkuit164/Backup/yolov7ps/yolov7/datasets/cls_pose/classes.txt"

visualize_labeled_data(image_folder, label_folder, class_names=read_cls_file(class_path))
