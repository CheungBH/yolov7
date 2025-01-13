import os
import cv2
from court_detector import CourtDetector


def click_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0].append((x, y))
        cv2.circle(param[1], (x, y), 5, (0, 255, 0), -1)  # 在点击位置绘制一个绿色圆点
        cv2.imshow('image', param[1])  # 实时更新图像显示


def main():
    click_type = "detect"

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
        cv2.destroyWindow('image')

    # 3. 对每个视频运行 detect.py
    for file, mask in zip(mp4_files, masks):
        court_detector = CourtDetector()
        cap = cv2.VideoCapture(os.path.join(folder_path, file))
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx == 0:
                court_detector.begin(type=click_type, frame=frame, mask_points=mask)
            else:
                court_detector.track_court(frame)

            cv2.imshow("court", frame)
            cv2.waitKey(1)  # 1 millisecond


if __name__ == "__main__":
    main()
