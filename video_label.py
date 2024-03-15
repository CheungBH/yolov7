import cv2
import os

input_folder = '/media/hkuit164/WD20EJRX/fall_dataset/debug/video'
output_folder = '/media/hkuit164/WD20EJRX/fall_dataset/MP_temp_json/labels_5'
check_label = False
frame_interval = 1


def classify_frame(frame_num, key):
    category = None
    if key == ord('1'):
        category = "0"
    elif key == ord('2'):
        category = "1"
    elif key == ord('3'):
        category = "2"
    return category


def video_frame_num(video_path):
    video = cv2.VideoCapture(video_path)
    num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return num


def txt_line_count(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        line_count = sum(1 for _ in file)
    return line_count


def video_txt(video_folder, txt_folder):
    txt_names = os.listdir(txt_folder)
    video_names = os.listdir(video_folder)
    label_status = {video_name: {"status": False, "frame_labelled": 0} for video_name in video_names}

    for video_name in video_names:
        matching_txt_name = None
        for txt_name in txt_names:
            if txt_name.split(".")[0] == video_name.split(".")[0]:

                matching_txt_name = txt_name
                break

        if matching_txt_name:
            frame_total = video_frame_num(os.path.join(video_folder, video_name))
            line_total = txt_line_count(os.path.join(txt_folder, matching_txt_name))
            if line_total == 0:
                print(f"{video_name}: {frame_total}, {matching_txt_name}: {line_total}. No labels AT ALL")
            elif line_total < frame_total:
                print(f"{video_name}: {frame_total}, {matching_txt_name}: {line_total}. Label not finished")
                label_status[video_name]["frame_labelled"] = line_total
            else:
                print(f"{video_name}: {frame_total}, {matching_txt_name}: {line_total}. Label finished")
                label_status[video_name]["status"] = True
                label_status[video_name]["frame_labelled"] = line_total
        else:
            print(f"{video_name} needs label")

    return label_status


def process_video(video_path, output_folder, frame_num):

    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # frame_num = 0
    classification_results = {}

    while video.isOpened() and frame_num < frame_count:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()

        if ret:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break

            if key == 81:  # Left arrow key
                frame_num = max(frame_num - frame_interval, 0)
                if frame_num in classification_results:
                    print(f"Frame {frame_num}: {classification_results[frame_num]}")
                else:
                    print(f"Frame {frame_num}: Not classified")

            if key == 83:  # Right arrow key
                frame_num += frame_interval
                if frame_num in classification_results:
                    print(f"Frame {frame_num}: {classification_results[frame_num]}")
                else:
                    print(f"Frame {frame_num}: Not classified")

            category = classify_frame(frame_num, key)
            if category:
                classification_results[frame_num] = category
                print(f"Frame {frame_num} classified as {category}")
                frame_num += frame_interval
        else:
            break

    video_filename = os.path.basename(video_path)
    txt_filename = os.path.splitext(video_filename)[0] + '.txt'
    txt_filepath = os.path.join(output_folder, txt_filename)

    with open(txt_filepath, "w") as file:
        for frame, category in classification_results.items():
            file.write(f"{category}\n")

    video.release()
    cv2.destroyAllWindows()


label_status = video_txt(input_folder, output_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    if os.path.isfile(file_path) and file_path.lower().endswith('.mp4') and label_status[filename]["status"] is False:
        print(f"Processing {filename}")
        process_video(file_path, output_folder, frame_num=label_status[filename]["frame_labelled"])
