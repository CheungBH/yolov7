import easyocr
import cv2
import pytesseract
import paddle
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image
import torch
import logging


def ocr_detect(mask_point,video_path):
    ocr = PaddleOCR(
        det_model_dir=r"C:\Users\cheun\OneDrive\Documents\WeChat Files\wxid_d8k0120oxz6z22\FileStorage\File\2025-04\final_model2\final_model2\det",
        rec_model_dir=r"C:\Users\cheun\OneDrive\Documents\WeChat Files\wxid_d8k0120oxz6z22\FileStorage\File\2025-04\final_model2\final_model2\rec",
        use_angle_cls=True,
        use_gpu=torch.cuda.is_available()
    )
    game_points_last = [[],[]]
    frame_id = 0
    time_line = {}
    logging.getLogger('ppocr').setLevel(logging.ERROR)

    cap = cv2.VideoCapture(video_path)
    ret, img = cap.read()
    if not ret:
        print("Unable to read first frame of video")
        cap.release()
        exit()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to retrieve frame, video may have ended")
            break

        # Crop ROI
        try:
            x1, y1 = mask_point[0]
            x2, y2 = mask_point[1]
            rect_image = frame[y1:y2, x1:x2]
        except IndexError:
            print("ROI not properly selected, skipping frame")
            continue


        # PaddleOCR recognition
        rgb_result = ocr.ocr(rect_image, cls=True)

        # Extract text
        texts = []
        if rgb_result[0]:  # Check if there are results
            texts = [line[1][0] for line in rgb_result[0]]

        valid_points = ['15', '30', '40', 'AD']

        if texts != []:
            game_points = extract_matching_values(texts,valid_points)
            try:
                if game_points_last[0] != [] and game_points_last[1] != [] :
                    if game_points[0][1] != game_points_last[0][1] or game_points[1][1] != game_points_last[1][1]:
                        time_line[frame_id] = [game_points[0][1],game_points[1][1]]
                game_points_last = game_points
            except:
                print(game_points)
        frame_id += 1
        print(frame_id)
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return time_line


def draw_timeline(data_dict, time_list):
    # 将时间列表转换为整数类型
    time_list = [int(t) for t in time_list]

    # 定义二分查找函数，找到小于 key 的最大值
    def find_closest_smaller(time_list, key):
        left, right = 0, len(time_list) - 1
        closest = None
        while left <= right:
            mid = (left + right) // 2
            if time_list[mid] < key:
                closest = time_list[mid]
                left = mid + 1
            else:
                right = mid - 1
        return closest

    # 找到每个字典键对应的最近的小于它的最大值
    result = {}
    for key in data_dict.keys():
        closest_time = find_closest_smaller(time_list, key)
        if closest_time is not None:
            result[key] = closest_time

    # 构建时间线和标注
    timeline_str = "时间线: " + " ---- ".join(map(str, time_list))
    annotation_str = "标注:   "

    # 初始化标注位置
    annotation_positions = {t: [] for t in time_list}
    for key, closest_time in result.items():
        annotation_positions[closest_time].append(str(key))

    # 填充标注字符串
    for t in time_list:
        if annotation_positions[t]:
            annotation_str += ",".join(annotation_positions[t]).ljust(10)
        else:
            annotation_str += " ".ljust(10)

    # 返回最终结果
    return timeline_str + "\n" + annotation_str




if __name__ == "__main__":
    video = r"C:\Users\cheun\Downloads\game1.mp4"
    # score_board = select_points_in_video(video)
    score_board = [ (104, 925), (508, 1026)]
    hight_list =['175', '876', '1359', '2453', '3074', '3309', '4738', '5277', '6001', '7032', '7587', '8053', '8716', '9339', '9877']
    time_line = ocr_detect(score_board,video)
    print(draw_timeline(time_line, hight_list))

