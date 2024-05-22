import os
import csv

label_folder_path = '/media/hkuit164/Backup/yolov7/runs/detect/v7ball_test3/labels'
csv_path = '/media/hkuit164/Backup/yolov7/runs/detect/v7ball_test3/labels.csv'


def get_bounding_box_center(xywh):
    center_x = float(xywh[0]) + float(xywh[2])/2
    center_y = float(xywh[1]) + float(xywh[3])/2
    return center_x, center_y


def process_txt_files(folder_path, csv_filename):
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'x_coord', 'y_coord'])
        for filename in filenames:
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as txtfile:
                lines = txtfile.readlines()
                max_value = 0
                best_line = None
                for line in lines:
                    if line.strip() == "":
                        continue
                    conf = float(line.split()[-1])
                    if conf > max_value:
                        max_value = conf
                        best_line = line
                if best_line is not None:
                    bbox = best_line.split()[1:5]
                    x_centre, y_centre = get_bounding_box_center(bbox)
                    writer.writerow([filename, x_centre, y_centre])


process_txt_files(label_folder_path, csv_path)
