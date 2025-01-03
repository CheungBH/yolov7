import os
import cv2
import subprocess

def click_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0].append((x, y))
        cv2.circle(param[1], (x, y), 5, (0, 255, 0), -1)  # 在点击位置绘制一个绿色圆点
        cv2.imshow('image', param[1])  # 实时更新图像显示

def main():
    # 1. 读取文件夹中的所有 .mp4 文件
    folder_path = '/media/hkuit164/WD20EJRX/Chris/ball_tutorial/data/test'
    mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

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
        param = [points, frame.copy()]  # 将点和帧传递给回调函数

        cv2.imshow('image', frame)
        cv2.setMouseCallback('image', click_points, param)

        # 等待用户点击四个点
        while len(points) < 4:
            cv2.waitKey(1)

        # 保存点击的点
        masks.append(points)
        cv2.destroyWindow('image')  # 关闭当前图像窗口

    # 3. 对每个视频运行 detect.py
    for file, mask in zip(mp4_files, masks):
        mask_str = ' '.join([f'{x}'' ' f'{y}' for x, y in mask])
        cmd = f'python detect_ball_person_landing_court_clsType.py --source {os.path.join(folder_path, file)} --masks "{mask_str}" --ball_weights runs/train/v7tiny_ball_detection2/weights/best.pt --human_weights runs/train/4cls_lr_tiny_0725/weights/best.pt  --name detectALL/lower_right/newcut'
        print(cmd)
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
