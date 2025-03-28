import os
import cv2
import subprocess
import shutil

resize_ratio = 2

def click_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0].append((x*resize_ratio, y*resize_ratio))
        cv2.circle(param[1], (x, y), 5, (0, 255, 0), -1)  # 在点击位置绘制一个绿色圆点
        cv2.imshow('image', param[1])  # 实时更新图像显示

def main():
    # 1. 读取文件夹中的所有 .mp4 文件
    # folder_path = r"D:\Ai_tennis\yolov7_main\test_video\Grass"
    folder_path = r"D:\tmp\3.26\3.24\candidates"
    output_folder = r"D:\tmp\3.27\candidates_output_refineColor_newlanding"
    use_saved_box = True

    mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    masks = []

    mask_file = os.path.join(folder_path, "mask.txt")
    if os.path.exists(mask_file):
        with open(mask_file, "r") as f:
            for line in f:
                mask = [int(x) for x in line.strip().split()]
                masks.append([(mask[i], mask[i+1]) for i in range(0, len(mask), 2)])

    else:
        for file in mp4_files:
            video_path = os.path.join(folder_path, file)
            print(f"video path: {video_path}")

            cap = cv2.VideoCapture(video_path)
            #设置要读取的帧号，例如第10帧
            frame_number = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            # 读取指定帧
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print(f"Failed to read {file}")
                continue

            points = []
            # param = [points, frame.copy()]  # 将点和帧传递给回调函数

            h, w = frame.shape[:2]
            print(f"width: {w}, height: {h}")
            frame = cv2.resize(frame, (w//resize_ratio, h//resize_ratio))
            param = [points, frame.copy()]
            cv2.imshow('image', frame)
            cv2.setMouseCallback('image', click_points, param)

            # 等待用户点击四个点
            while len(points) < 4:
                cv2.waitKey(1)

            # 保存点击的点
            masks.append(points)
            print(masks)
            cv2.destroyWindow('image')  # 关闭当前图像窗口


        mask_file = os.path.join(folder_path, "mask.txt")
        with open(mask_file, "w") as f:
            for mask in masks:
                mask_str = ' '.join([f'{x}'' ' f'{y}' for x, y in mask])
                f.write(mask_str + "\n")

    # 3. 对每个视频运行 detect.py
    total_videos = len(mp4_files)
    for v_idx, (file, mask) in enumerate(zip(mp4_files, masks)):
        if mask[-2][0] < 200:
            continue
        mask_str = ' '.join([f'{x}'' ' f'{y}' for x, y in mask])
        video_name = file.split(".")[0]
        # cmd = (f'python detect_analysis.py --source {os.path.join(folder_path, file)} --masks "{mask_str}" '
        #        f' --output_csv_file {"test_csv/" + os.path.join(file.split(".")[0]) + ".csv"} '
        #        f' --project runs/detect --name newcut'
        #        f' --topview_path {"top_view/" + os.path.join(file.split(".")[0]) + "_2.mp4"}')
        cmd = 'python detect_pose_ball.py --source {} --output_folder {} --masks "{}"'.format(
            os.path.join(folder_path, file), os.path.join(output_folder, video_name), mask_str)
        if use_saved_box:
            cmd += ' --use_saved_box'
        print("processing video {}: {}/{}".format(file, v_idx+1, total_videos))
        # print(cmd)
        subprocess.run(cmd, shell=True)


# 4. 把每个文件夹newcut中的视频提取出来
def extract_video(source_folder, destination_folder):
    # 创建目标文件夹（如果不存在）
    os.makedirs(destination_folder, exist_ok=True)

    # 遍历源文件夹中的所有子文件夹
    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        # 检查是否是文件夹
        if os.path.isdir(folder_path):
            # 遍历子文件夹中的每个文件
            for filename in os.listdir(folder_path):
                source_file = os.path.join(folder_path, filename)
                # 检查是否是文件（而不是文件夹）
                if os.path.isfile(source_file):
                    # 复制文件到目标文件夹
                    shutil.copy(source_file, destination_folder)

    print("所有文件已成功提取到目标文件夹中。")


if __name__ == "__main__":
    main()
