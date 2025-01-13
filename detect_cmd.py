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
    folder_path = "D:\Ai_tennis\Source\Phone_shot\hk_atp250"
    mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.MOV')]

    masks = []

    for file in mp4_files:
        video_path = os.path.join(folder_path, file)

        # 2. 读取每个 .mp4 文件的第一帧图像
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Failed to read {file}")
            continue

        points = []
        # param = [points, frame.copy()]  # 将点和帧传递给回调函数

        h, w = frame.shape[:2]
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

    # 3. 对每个视频运行 detect.py
    for file, mask in zip(mp4_files, masks):
        mask_str = ' '.join([f'{x}'' ' f'{y}' for x, y in mask])
        cmd = (f'python detect_ball_person_landing_court_clsType.py --source {os.path.join(folder_path, file)} --masks "{mask_str}" '  
               f'  --name detectPhone/hk250/newcut')
        print(cmd)
        subprocess.run(cmd, shell=True)

    # 4. 把每个文件夹newcut中的视频提取出来
    # 定义源文件夹和目标文件夹
    source_folder = "D:\Ai_tennis/yolov7/runs\detect/detectPhone/hk250"
    destination_folder = "D:\Ai_tennis/yolov7/runs\detect/detectPhone/hk250"

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
