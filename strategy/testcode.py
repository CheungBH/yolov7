import cv2
import numpy as np

# 输入视频路径
video_path = r"C:\Users\cheun\Downloads\videoplayback.webm"  # 替换为你的视频路径
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频的基本信息
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义输出视频的编码器和文件名
output_path = "output_with_all_contours.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 假设比分栏在左下角，我们只关注左下角区域
    score_area_width = frame_width // 4  # 左下角宽度为 1/4
    score_area_height = frame_height // 6  # 左下角高度为 1/6
    left_bottom_area = gray[frame_height - score_area_height:frame_height, 0:score_area_width]

    # 二值化处理：使用自适应阈值代替固定阈值
    binary = cv2.adaptiveThreshold(
        left_bottom_area, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2
    )

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制所有符合条件的轮廓
    min_area = 500  # 设置最小面积阈值，过滤掉过小的轮廓
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area and 0.5 < w / h < 2.0:  # 过滤宽高比不符合要求的轮廓
            # 将坐标映射回原图
            x_global = x
            y_global = frame_height - score_area_height + y

            # 绘制红色矩形框，线宽为 5
            cv2.rectangle(frame, (x_global, y_global), (x_global + w, y_global + h), (0, 0, 255), 5)

    # 写入处理后的帧到输出视频
    out.write(frame)

    # 显示处理后的帧（可选）
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"处理完成，结果保存在 {output_path}")