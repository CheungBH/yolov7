import cv2
import csv
import ast
import numpy as np

def read_csv_file(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            row[1] = ast.literal_eval(row[1])
            row[3] = ast.literal_eval(row[3])
            row[5] = ast.literal_eval(row[5])
            # 将ball_location字符串解析为列表
            data.append(row)
    return data

def process_rally_changes(data):
    rally_change_list = []
    rally_cnt_prev = data[0][6]
    for row in data:
        rally_cnt = row[6]
        frame = int(row[7])
        if rally_cnt != rally_cnt_prev:
            rally_change_list.append(frame)
        rally_cnt_prev = rally_cnt
    rally_change_list = [rally_change_list[i] for i in range(len(rally_change_list)) if i == 0 or rally_change_list[i] - rally_change_list[i-1] > 3]
    rally_change_intervals = [[rally_change_list[i], rally_change_list[i+1]] for i in range(len(rally_change_list)-1)]
    return rally_change_list, rally_change_intervals

def find_ball_list(data):
    ball_list=[[500,500]]*int(data[0][7])
    upper_box_list=[[100,100,200,200]]*int(data[0][7])
    lower_box_list=[[100,100,200,200]]*int(data[0][7])
    for row in data:
        ball_list.append(row[5]) if row[5]!= (-1,-1) else ball_list.append(ball_list[-1])
        upper_box_list.append(row[1])
        lower_box_list.append(row[3])
    return ball_list,upper_box_list,lower_box_list

def get_human_hit(rally_change_intervals, data, height,ball_list):
    human_hits = []
    rally_change_new_intervals=[]
    for begin_time, end_time in rally_change_intervals:
        for row in data:
            if int(row[7]) == begin_time:
                ball_location = ball_list#row[5]
                if ball_location[begin_time][1] > 0.5 * height:
                    human_hits.append('lower')
                else:
                    human_hits.append('upper')
                break
    for i in range(len(human_hits)):
        if i > 0 and human_hits[i] != human_hits[i-1]:
            rally_change_new_intervals.append(rally_change_intervals[i])
        elif i > 0:
            rally_change_new_intervals[-1][1] = rally_change_intervals[i][1]
        #else:
            #rally_change_new_intervals.append(rally_change_intervals[i])  # For the first element
    human_hits = [human_hits[i] for i in range(len(human_hits)) if human_hits[i] !=human_hits[i-1]]
    return human_hits,rally_change_new_intervals

def get_hit_times(rally_change_intervals, human_hits, data):
    upper_hit = []
    lower_hit = []
    first_landing = []
    upper_hit_intervals =[]
    lower_hit_intervals =[]
    for (begin_time, end_time), human_hit in zip(rally_change_intervals, human_hits):
        if human_hit == 'upper':
            upper_action_times = [int(row[7]) for row in data if begin_time <= int(row[7]) <= end_time and row[0].lower() != 'waiting']
            upper_hit_intervals.append([upper_action_times[0] ,upper_action_times[-1]])
            if upper_action_times:
                hit_time = (upper_action_times[0] + upper_action_times[-1]) // 2
                upper_hit.append(hit_time)

        else:
            lower_action_times = [int(row[7]) for row in data if begin_time <= int(row[7]) <= end_time and row[2].lower() != 'waiting']
            lower_hit_intervals.append([lower_action_times[0] ,lower_action_times[-1]])
            if lower_action_times:
                hit_time = (lower_action_times[0] + lower_action_times[-1]) // 2
                lower_hit.append(hit_time)

        landing_times = [int(row[7]) for row in data if begin_time <= int(row[7]) <= end_time and row[4].lower() == 'landing']
        if landing_times:
            first_landing.append(landing_times[0])

    return upper_hit, lower_hit, first_landing,upper_hit_intervals,lower_hit_intervals

def upper_lower_state(total_frame,upper_hit,lower_hit,upper_hit_intervals,lower_hit_intervals):
    upper_state_list = [2] * total_frame
    lower_state_list = [2] * total_frame
    for i in range(len(upper_hit_intervals)):
        start, end = upper_hit_intervals[i]
        for j in range(start, end + 1):
            upper_state_list[j] = 0
        if i < len(lower_hit):
            next_val = lower_hit[i]
            for k in range(end + 1, next_val + 1):
                upper_state_list[k] = 1

    for start, end in lower_hit_intervals:
        for i in range(start, end + 1):
            lower_state_list[i] = 0
    intervals = zip([0] + [end + 1 for start, end in lower_hit_intervals], upper_hit + [total_frame])
    for start, end in intervals:
        for i in range(start, end):
            if lower_state_list[i] == 2:
                lower_state_list[i] = 1
    return upper_state_list, lower_state_list


def draw_state_info(frame, upper_hit, lower_hit, frame_count, upper_state_list, lower_state_list,upper_box_list,lower_box_list):
    # 绘制 upper_hit 和 lower_hit 信息
    text = f"Upper Hit: {' '.join(map(str, upper_hit))}\nLower Hit: {' '.join(map(str, lower_hit))}"
    y0, dy = 30, 30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        #cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    upper_state = 'hit' if upper_state_list[frame_count]==0 else 'return' if upper_state_list[frame_count]==1 else 'approach'
    lower_state = 'hit' if lower_state_list[frame_count]==0 else 'return' if lower_state_list[frame_count]==1 else 'approach'
    if lower_state_list[frame_count]==2 and lower_state_list[frame_count-1]==1:
        cv2.putText(frame, f'Upper Hit', (10, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if upper_state_list[frame_count]==2 and upper_state_list[frame_count-1]==1:
        cv2.putText(frame, f'Lower Hit', (10, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # 绘制 upper_state 和 lower_state 信息
    upper_state_color = (0, 0, 255) if upper_state == 'hit' else (0, 255, 0) if upper_state == 'return' else (255, 0, 255)
    lower_state_color = (0, 0, 255) if lower_state == 'hit' else (0, 255, 0) if lower_state == 'return' else (255, 0, 255)

    cv2.putText(frame, f'{upper_state}', (int(upper_box_list[frame_count][0]), int(upper_box_list[frame_count][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, upper_state_color, 2)
    cv2.putText(frame, f'{lower_state}', (int(lower_box_list[frame_count][0]), int(lower_box_list[frame_count][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, lower_state_color, 2)


def draw_ball_and_boxes(frame, row, first_landing, frame_count, previous_positions, ball_list):
    ball_location = row[5]
    color = (0, 255, 0) #if frame_count in first_landing else (0, 255, 0)
    ball_state = row[4].lower()

    # 记录当前球的位置
    current_position = (int(ball_location[0]), int(ball_location[1]))
    previous_positions.append(current_position)

    # 只保留最近的6个位置
    if len(previous_positions) > 6:
        previous_positions.pop(0)

    # 绘制当前帧及之前5帧的球位置
    for i, position in enumerate(reversed(previous_positions)):
        alpha = 1.0 - i * 0.1
        overlay = frame.copy()
        cv2.circle(overlay, position, 10, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 标记第一次落地的位置并显示编号
    for index, landing_frame in enumerate(first_landing):
        if frame_count >= landing_frame:
            position = ball_list[landing_frame]
            cv2.circle(frame, (int(position[0]), int(position[1])), 10, (0, 0, 0), 2)
            cv2.putText(frame, f'Tag {index + 1}', (int(position[0]), int(position[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 绘制 upper_box 和 lower_box
    upper_box = row[1]
    lower_box = row[3]
    upper_action = row[0].lower()
    lower_action = row[2].lower()

    upper_color = (128, 128, 128) if upper_action == 'waiting' else (0, 0, 255)
    cv2.rectangle(frame, (int(upper_box[0]), int(upper_box[1])), (int(upper_box[2]), int(upper_box[3])), upper_color, 2)
    lower_color = (128, 128, 128) if lower_action == 'waiting' else (0, 0, 255)
    cv2.rectangle(frame, (int(lower_box[0]), int(lower_box[1])), (int(lower_box[2]), int(lower_box[3])), lower_color, 2)
def initialize_video_writer(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    return cap, out, fps, width, height

def main(csv_file,video_file):
    csv_file = csv_file
    video_file = video_file
    previous_positions = []
    data = read_csv_file(csv_file)
    cap, out, fps, width, height = initialize_video_writer(video_file)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rally_change_list, rally_change_intervals = process_rally_changes(data)
    ball_list,upper_box_list,lower_box_list = find_ball_list(data)
    human_hits,rally_change_intervals = get_human_hit(rally_change_intervals, data, height,ball_list)
    upper_hit, lower_hit, first_landing,upper_hit_intervals,lower_hit_intervals = get_hit_times(rally_change_intervals, human_hits, data)
    upper_state_list,lower_state_list = upper_lower_state(total_frame,upper_hit, lower_hit,upper_hit_intervals,lower_hit_intervals)



    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        for row in data:
            if int(row[7]) == frame_count:
                draw_ball_and_boxes(frame, row, first_landing, frame_count, previous_positions,ball_list)
            draw_state_info(frame, upper_hit, lower_hit, frame_count, upper_state_list, lower_state_list,upper_box_list,lower_box_list)

        out.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        print(frame_count)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("处理后的视频已保存为 output.mp4")

if __name__ == "__main__":
    main(csv_file='/media/hkuit164/Backup/yolov7/test_csv/output2.csv',video_file='/media/hkuit164/WD20EJRX/Chris/ball_tutorial/Kh/20231011_kh_yt_1.mp4')
