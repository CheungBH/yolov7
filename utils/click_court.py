import easyocr
import cv2
import pytesseract
import paddle
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image
import torch
import logging

def click_points(img):
    mask_points = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mask_points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('image', img)
            if len(mask_points) > 4:
                mask_points.pop(0)
            print(mask_points)

    height, width, channel = img.shape
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", width, height)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_event)

    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    return mask_points

def select_points_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    mask_point = []  # 存储用户选择的点
    frame_count = 0  # 记录当前帧数

    def on_mouse_click(event, x, y, flags, param):
        """
        鼠标回调函数，用于捕获用户点击的点。
        """
        nonlocal mask_point
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            if len(mask_point) < 6:  # 只允许选择两个点
                mask_point.append((x, y))
                print(f"点 {len(mask_point)}: ({x}, {y})")
                # 在图像上绘制选中的点
                cv2.circle(frame_display, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Frame", frame_display)

    # 创建窗口并绑定鼠标回调函数
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", on_mouse_click)

    while True:
        ret, frame = cap.read()
        if not ret:  # 视频结束
            print("视频已结束。")
            break

        frame_count += 1
        frame_display = frame.copy()  # 复制帧用于显示
        print(f"当前帧: {frame_count}")

        # 显示当前帧
        cv2.imshow("Frame", frame_display)

        # 等待用户输入
        key = cv2.waitKey(0) & 0xFF  # 等待用户操作
        if key == ord(' '):  # 按下空格键跳到下一帧
            mask_point.clear()  # 清空之前选择的点
            continue
        elif key == 27:  # 按下 ESC 键退出
            print("用户退出。")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    return mask_point[0:4],mask_point[4:6]


def ocr_detect(mask_point,img,detect_model,rec_model):
    ocr = PaddleOCR(
        det_model_dir=detect_model,
        rec_model_dir=rec_model,
        use_angle_cls=True,
        use_gpu=torch.cuda.is_available()
    )
    game_points_last = [[],[]]
    time_line = {}
    logging.getLogger('ppocr').setLevel(logging.ERROR)

    x1, y1 = mask_point[0]
    x2, y2 = mask_point[1]
    rect_image = img[y1:y2, x1:x2]
    rgb_result = ocr.ocr(rect_image, cls=True)
    texts = []
    if rgb_result[0]:  # Check if there are results
        texts = [line[1][0] for line in rgb_result[0]]
    return texts

