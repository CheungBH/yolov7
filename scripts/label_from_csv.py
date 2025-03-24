import sys

import cv2
import csv
import ast
import os
import pandas as pd

int_to_status = {-1: "Pending", 0: "flying", 1: "landing"}
status_colors = {"Pending": (255, 0, 0), "flying": (0, 255, 0), "landing": (0, 0, 255)}


def find_csv_video_input_paths(input_csv_folder, input_video_folder):
    csv_paths = []
    video_paths = []
    for root, dirs, files in os.walk(input_csv_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_file_path = os.path.join(root, file)
                # dir = root.split(os.sep)[-1]
                video_file_path = os.path.join(input_video_folder, file.replace('.csv', '.mp4'))

                if os.path.exists(video_file_path):
                    # print(csv_file_path)
                    csv_paths.append(csv_file_path)
                    video_paths.append(video_file_path)
                else:
                    print('Video file not found')

    return csv_paths, video_paths

def read_csv_file(csv_file):
    print(csv_path)
    df = pd.read_csv(csv_path)
    df['ball'] = df['ball'].apply(ast.literal_eval)  # 将字符串转换为列表
    ball_locations = df['ball'].tolist()
    # df['curve_status'] = df['curve_status'].apply(ast.literal_eval)  # 将字符串转换为列表
    curve_prediction = df['curve_status'].tolist()
    status_to_int = {"Pending": -1, "flying": 0, "landing": 1}
    curve_prediction = [status_to_int[status] for status in curve_prediction]
    frame_ls = df['frame'].tolist()
    print(f'first_frame:{frame_ls[0]}, last_frame:{frame_ls[-1]}, len(frame_ls):{len(frame_ls)}')
    return ball_locations, frame_ls, curve_prediction



def main(csv_path, video_path, output_txt_folder):
    ball_locations, frame_ls, curve_predictions = read_csv_file(csv_path)
    location_dict = dict(zip(frame_ls, ball_locations))

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否打开成功
    if not cap.isOpened():
        print("无法打开视频")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frame_width, frame_height)

    # 创建一个空的帧列表来存储视频的帧
    frames = []

    # 读取视频的每一帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    print(f'len(frames):{len(frames)}')
    cap.release()

    # 创建字典存储landing(1)/flying(0)，默认是0
    t = {i:curve_predictions[i] for i in range(frame_ls[0], frame_ls[-1]+1)}

    # 帧索引
    frame_idx = frame_ls[0] # 注意这里要不要-1

    # 调色盘
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # 绿，蓝，红

    # 显示帧
    while True:
        if frame_idx < len(frames):
            # 复制一份原始图片以便绘制
            img_copy = frames[frame_idx-1].copy()

            # 在图片上绘制当前球的位置，使用不同透明度的颜色表示前6个球的位置
            num_frames_to_show = 6
            for i in range(num_frames_to_show, -1, -1):
                if frame_idx - i in location_dict:
                    alpha = (1 - i * 0.1)  # 计算透明度
                    overlay = img_copy.copy()
                    cv2.circle(overlay, (int(location_dict[frame_idx - i][0]), int(location_dict[frame_idx - i][1])), 7,colors[0], 1)
                    cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)
            # cv2.circle(img_copy, sand_pred_ball_locations[frame_idx])
            csv_name = csv_path.split('\\')[-1]
            cv2.putText(img_copy, csv_name,(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img_copy, f'frame:{frame_idx}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            status = int_to_status[t.get(frame_idx, -1)]
            cv2.putText(img_copy, status, (50, 150),  cv2.FONT_HERSHEY_SIMPLEX, 2, status_colors[status], 2)

            cv2.imshow("Video", img_copy)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('f'):  # 按右键跳到下一帧
            frame_idx = min(frame_idx + 1, frame_ls[-1])
        elif key == ord('d'):  # 按左键回到上一帧
            frame_idx = max(frame_idx - 1, -frame_ls[-1])
        elif key == ord('='):  # 按+键加50
            frame_idx = min(frame_idx + 50, frame_ls[-1])
        elif key == ord('-'):  # 按-键减50
            frame_idx = max(frame_idx - 50, -frame_ls[-1])
        elif key == ord("1") or key == ord(' '):
            t[frame_idx] = 1
            frame_idx += 1
        elif key == ord("0"):
            t[frame_idx] = 0
            frame_idx += 1
        elif key == ord("2"):
            t[frame_idx] = 2
            frame_idx += 1
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # frame_width, frame_height =1665, 3506

    csv_file_name = csv_path.split('\\')[-1]
    txt_file_name = csv_file_name.replace('.csv', '.txt')
    with open(os.path.join(output_txt_folder, txt_file_name), "w") as file:
        for idx, point in location_dict.items():
            if -1 in point:
                file.write("{}, {}, {}\n".format(point[0], point[1], t.get(idx, "")))
            else:
                file.write("{}, {}, {}\n".format(point[0]/frame_width, point[1]/frame_height, t.get(idx, "")))

    print(f"Coordinates saved to {os.path.join(output_txt_folder, txt_file_name)} \n ")


if __name__ == '__main__':
    input_video_folder = "/Volumes/U357/tennis_landing/video"
    input_csv_folder = "/Volumes/U357/tennis_landing/csv"
    output_txt_folder = "/Volumes/U357/tennis_landing/result"
    if not os.path.exists(output_txt_folder):
        os.makedirs(output_txt_folder)

    csv_paths, video_paths = find_csv_video_input_paths(input_csv_folder, input_video_folder)
    for csv_path, video_path in zip(csv_paths, video_paths):
        main(csv_path, video_path, output_txt_folder)