import os

import cv2
import csv
import ast
import numpy as np

def read_csv_file(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表
        for row in reader:
            row[1] = ast.literal_eval(row[1])
            row[3] = ast.literal_eval(row[3])
            row[5] = ast.literal_eval(row[5])
            row[9] = ast.literal_eval(row[9])
            # row[11] = ast.literal_eval(row[11])
            # 将ball_location字符串解析为列表
            data.append(row)
    return data

# def process_rally_changes(data):
#     rally_change_list = []
#     rally_cnt_prev = data[0][6]
#     for row in data:
#         rally_cnt = row[6]
#         frame = int(row[7])
#         if rally_cnt != rally_cnt_prev:
#             rally_change_list.append(frame)
#         rally_cnt_prev = rally_cnt
#     rally_change_list = [rally_change_list[i] for i in range(len(rally_change_list)) if i == 0 or rally_change_list[i] - rally_change_list[i-1] > 3]
#     rally_change_intervals = [[rally_change_list[i], rally_change_list[i+1]] for i in range(len(rally_change_list)-1)] #第一球和最后一球不计入
#     return rally_change_list, rally_change_intervals

def process_rally_changes(data):
    first_frame = int(data[0][7])
    last_frame = int(data[-1][7])
    rally_change_list = []
    rally_cnt_prev = data[0][6]
    for row in data:
        rally_cnt = row[6]
        frame = int(row[7])
        if rally_cnt != rally_cnt_prev and rally_cnt != '0':
            rally_change_list.append(frame-1)
        rally_cnt_prev = rally_cnt
    rally_change_list = [rally_change_list[i] for i in range(len(rally_change_list)) if i == 0 or rally_change_list[i] - rally_change_list[i-1] > 3]
    rally_change_intervals = [[rally_change_list[i]+1, rally_change_list[i+1]] for i in range(len(rally_change_list)-1)]
    rally_change_intervals.insert(0,[first_frame,rally_change_list[0]])
    rally_change_intervals.append([rally_change_list[-1]+1,last_frame])
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

def get_human_hit(rally_change_intervals, data, height):
    human_hits = []
    for begin_time, end_time in rally_change_intervals:
        for row in data:
            middle_upper_y = int(row[10])
            if begin_time <= int(row[7]) <= end_time:
                ball_location = row[5]
                if ball_location != (-1,-1):
                    if ball_location[1] > middle_upper_y:  #这个数值最好变成self.middle_upper_y
                        human_hits.append('lower')
                    else:
                        human_hits.append('upper')
                    break
    # for i in range(len(human_hits)):
    #     if i > 0 and human_hits[i] != human_hits[i-1]:
    #         rally_change_new_intervals.append(rally_change_intervals[i])
    #     elif i > 0:
    #         rally_change_new_intervals[-1][1] = rally_change_intervals[i][1]
    #     elif i==0:
    #         rally_change_new_intervals.append(rally_change_intervals[0])  # For the first element
    # human_hits = [human_hits[i] for i in range(len(human_hits)) if human_hits[i] !=human_hits[i-1]]
    return human_hits

def get_hit_times(rally_change_intervals, human_hits, data):
    upper_hit = []
    lower_hit = []
    hit_times = []
    upper_hit_intervals =[]
    lower_hit_intervals =[]
    for (begin_time, end_time), human_hit in zip(rally_change_intervals, human_hits):
        if human_hit == 'upper':
            upper_action_times = [int(row[7]) for row in data if begin_time <= int(row[7]) <= end_time and row[0].lower() != 'waiting']
            # upper_hit_intervals.append([upper_action_times[0] ,upper_action_times[-1]])
            if upper_action_times:
                upper_hit_intervals.append([upper_action_times[0], upper_action_times[-1]])
                hit_time = (upper_action_times[0] + upper_action_times[-1]) // 2
                upper_hit.append(hit_time)
                hit_times.append(hit_time)

        else:
            lower_action_times = [int(row[7]) for row in data if begin_time <= int(row[7]) <= end_time and row[2].lower() != 'waiting']
            # lower_hit_intervals.append([lower_action_times[0] ,lower_action_times[-1]])
            if lower_action_times:
                lower_hit_intervals.append([lower_action_times[0], lower_action_times[-1]])
                hit_time = (lower_action_times[0] + lower_action_times[-1]) // 2
                lower_hit.append(hit_time)
                hit_times.append(hit_time)

        # landing_times = [int(row[7]) for row in data if begin_time <= int(row[7]) <= end_time and row[4].lower() == 'landing' and int(row[6]) != 0]
        # if landing_times: # 如果landing_times非空
        #     first_landing = landing_times[0]
        #     adjacent_5_state = [row[4].lower() for row in data if first_landing <= int(row[7]) <= first_landing + 4]
        #     precise_landing = first_landing - (5 - adjacent_5_state.count("landing")) - 2
        #     precise_landings.append(precise_landing)

    return upper_hit, lower_hit, hit_times, upper_hit_intervals,lower_hit_intervals

def precise_landing(rally_change_intervals, data):
    precise_landings = []
    for begin, end in rally_change_intervals:
        state_frame_ls = [[row[4].lower(), int(row[7])] for row in data if begin <= int(row[7]) <= end and int(row[6]) != 0]
        if state_frame_ls:
            # 定义窗口大小
            # windoe_size = 3
            window_size = 5
            # 初始化变量存储最多 landing 的窗口
            max_count = -1
            max_window = []
            # 遍历列表，查看每个窗口
            for i in range(len(state_frame_ls) - window_size + 1):
                if state_frame_ls[i][0] == 'landing':
                    window = state_frame_ls[i:i + window_size]
                    flattened_window = [item for sublist in window for item in sublist]
                    count_landing = flattened_window.count('landing')

                    # 如果当前窗口中 landing 的数量大于之前的最大值，则更新最大值和窗口
                    if count_landing > max_count:
                        max_count = count_landing
                        max_window = window
            if max_window:
                first_landing = max_window[0][1]
                # adjacent_3_state = [row[4].lower() for row in data if first_landing <= int(row[7]) <= first_landing + 2]
                # precise_landing = first_landing - (3 - adjacent_3_state.count("landing")) - 1
                # precise_landings.append(precise_landing)

                adjacent_5_state = [row[4].lower() for row in data if first_landing <= int(row[7]) <= first_landing + 4]
                precise_landing = first_landing - (5 - adjacent_5_state.count("landing")) - 2
                precise_landings.append(precise_landing)

    return precise_landings


def min_width_frame(boxs):
    width_ls = []
    frame_ls = []
    min_width = 200
    min_width_idx = 0
    for box,frame in boxs:
        width = box[2]-box[0]
        width_ls.append(width)
        frame_ls.append(frame)
    for idx in range(len(width_ls)):
        if width_ls[idx] < min_width:
            min_width = width_ls[idx]
            min_width_idx = idx
    min_width_frame = frame_ls[min_width_idx]
    # print(f'min_width:{min_width}')
    return min_width_frame

def upper_lower_state(upper_hit,lower_hit,upper_hit_intervals,lower_hit_intervals, data):
    last_frame = int(data[-1][7])
    upper_state_list = [3] * last_frame
    lower_state_list = [3] * last_frame
    upper_ret_rea_app_list = []
    lower_ret_rea_app_list = []

    print('upper')
    for i in range(len(upper_hit_intervals)):
        upper_boxs = []
        start, end = upper_hit_intervals[i]
        upper_ret_rea_app_list.extend([start, end])
        # print(i)
        # print(f'upper_hit:{upper_hit[i]},start:{start},end:{end}')
        for j in range(start, end):
            upper_state_list[j] = 0
        if i < len(lower_hit)-1:
            next_val = lower_hit[i+1]
            # print(f'next_val:{next_val}')
            if next_val > end:
                for row in data:
                    frame = int(row[7])
                    if end <= frame <= next_val:
                        upper_box = row[1]
                        upper_boxs.append([upper_box,frame])
                ready_start = min_width_frame(upper_boxs)
                upper_ret_rea_app_list.append(ready_start)
                # print(f'ready_start:{ready_start}')
                if end < ready_start < next_val:
                    for k in range(end,ready_start):
                        upper_state_list[k] = 1
                    for m in range(ready_start,next_val):
                        upper_state_list[m] = 2
            else:
                ready_start = 'None'
                upper_ret_rea_app_list.append(ready_start)
                # print(f'ready_start:{ready_start}')
    # print(f'upper_ret_rea_app_list:{upper_ret_rea_app_list}')

    print('lower')
    for i in range(len(lower_hit_intervals)):
        lower_boxs = []
        start, end = lower_hit_intervals[i]
        lower_ret_rea_app_list.extend([start, end])
        # print(i)
        # print(f'lower_hit;{lower_hit[i]},start:{start},end:{end}')
        for j in range(start, end):
            lower_state_list[j] = 0
        if i < len(upper_hit)-1:
            next_val = upper_hit[i+1]
            # print(f'next_val:{next_val}')
            if next_val > end:
                for row in data:
                    frame = int(row[7])
                    if end <= frame <= next_val:
                        lower_box = row[3]
                        lower_boxs.append([lower_box, frame])
                ready_start = min_width_frame(lower_boxs)
                lower_ret_rea_app_list.append(ready_start)
                # print(f'ready_start:{ready_start}')
                if end < ready_start < next_val:
                    for k in range(end, ready_start):
                        lower_state_list[k] = 1
                    for m in range(ready_start, next_val):
                        lower_state_list[m] = 2
        else:
            ready_start = 'None'
            lower_ret_rea_app_list.append(ready_start)
            # print(f'ready_start:{ready_start}')
    # print(f'lower_ret_rea_app_list:{lower_ret_rea_app_list}')
    return upper_state_list, lower_state_list, upper_ret_rea_app_list, lower_ret_rea_app_list

def is_same_direction(a, b, c):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    # 计算向量 ab 和 bc
    ab = (x2 - x1, y2 - y1)
    bc = (x3 - x2, y3 - y2)
    # 计算点积
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    return dot_product > 0

def change_direction(upper_ret_rea_app_list, lower_ret_rea_app_list, data):
    last_frame = int(data[-1][7])
    upper_locations = []
    lower_locations = []
    upper_direction_list = [2] * last_frame
    lower_direction_list = [2] * last_frame
    for i in range(len(upper_ret_rea_app_list)):
        if upper_ret_rea_app_list[i] != 'None':
            for row in data:
                frame = int(row[7])
                if frame == upper_ret_rea_app_list[i]:
                    upper_box = row[1]
                    upper_location = ((upper_box[0]+upper_box[2])/2,upper_box[3])
                    upper_locations.append([upper_location, upper_ret_rea_app_list[i]])
        else:
            upper_locations.append(['None', upper_ret_rea_app_list[i]])
    # print(f'upper_locations:{upper_locations}')

    for j in range(len(upper_locations)):
        if j >= 3 and j % 3 == 0 :
            ret = upper_locations[j - 2][0]
            rea = upper_locations[j - 1][0]
            app = upper_locations[j][0]
            ready_frame = upper_locations[j - 1][1]
            if ready_frame != 'None':
                if is_same_direction(ret, rea, app):
                    for k in range(ready_frame, ready_frame + 20):
                        upper_direction_list[k] = 0
                else:
                    for k in range(ready_frame, ready_frame + 20):
                        upper_direction_list[k] = 1

    for i in range(len(lower_ret_rea_app_list)):
        if lower_ret_rea_app_list[i] != 'None':
            for row in data:
                frame = int(row[7]) #注意这个int啊！！！！
                if frame == lower_ret_rea_app_list[i]:
                    lower_box = row[3]
                    lower_location = ((lower_box[0]+lower_box[2])/2,lower_box[3])
                    lower_locations.append([lower_location, lower_ret_rea_app_list[i]])
        else:
            lower_locations.append(['None', lower_ret_rea_app_list[i]])
    # print(f'lower_locations:{lower_locations}')

    for j in range(len(lower_locations)):
        if j >= 3 and j % 3 == 0:
            ret = lower_locations[j - 2][0]
            rea = lower_locations[j - 1][0]
            app = lower_locations[j][0]
            ready_frame = lower_locations[j - 1][1]
            if ready_frame != 'None':
                if is_same_direction(ret, rea, app):
                    for k in range(ready_frame, ready_frame + 20):
                        lower_direction_list[k] = 0
                else:
                    for k in range(ready_frame, ready_frame + 20):
                        lower_direction_list[k] = 1

    return upper_direction_list, lower_direction_list

def is_cross(ball_locations, width):
    x_list = []
    for ball_location in ball_locations:
        x, y = ball_location[0], ball_location[1]
        x_list.append(x)
    x_range = max(x_list) - min(x_list)
    return x_range > width*0.28

def cross_straight(hit_times, data, width):
    last_frame = int(data[-1][7])
    cross_straight_list = [[2,(-1,-1),(-1,-1)]] * last_frame
    for i in range(len(hit_times)):
        if i >= 1:
            prev_hit = hit_times[i - 1]
            curr_hit = hit_times[i]
            real_ball_locations = [row[9][0] for row in data if prev_hit <= int(row[7]) <= curr_hit and row[5] != (-1,-1)]
            ball_locations = [row[5] for row in data if prev_hit < int(row[7]) < curr_hit and row[5] != (-1,-1)]
            x1 = int(ball_locations[0][0])
            y1 = int(ball_locations[0][1])
            x2 = int(ball_locations[-1][0])
            y2 = int(ball_locations[-1][1])
            start_point = (x1, y1)
            end_point = (x2, y2)
            if is_cross(real_ball_locations, width):
                for j in range(prev_hit, curr_hit):
                    cross_straight_list[j] = [0,start_point, end_point]
            else:
                for j in range(prev_hit, curr_hit):
                    cross_straight_list[j] = [1,start_point, end_point]
    last_real_ball_locations = [row[9][0] for row in data if hit_times[-1] <= int(row[7]) <= last_frame and row[5] != (-1, -1)]
    last_ball_locations = [row[5] for row in data if hit_times[-1] <= int(row[7]) <= last_frame and row[5] != (-1, -1)]
    if last_ball_locations:
        x1 = int(last_ball_locations[0][0])
        y1 = int(last_ball_locations[0][1])
        x2 = int(last_ball_locations[-1][0])
        y2 = int(last_ball_locations[-1][1])
        start_point = (x1, y1)
        end_point = (x2, y2)
        if is_cross(last_real_ball_locations, width):
            for k in range(hit_times[-1], last_frame):
                cross_straight_list[k]=[0,start_point, end_point]
        else:
            for k in range(hit_times[-1], last_frame):
                cross_straight_list[k]=[1,start_point, end_point]
    return cross_straight_list

# def cross_straight(total_frame, hit_times, data, width):
#     cross_straight_list = [[2,(-1,-1),(-1,-1)]] * total_frame
#     for i in range(len(hit_times)):
#         if i >= 1:
#             prev_hit = hit_times[i - 1]
#             curr_hit = hit_times[i]
#             ball_locations = [row[5] for row in data if prev_hit < int(row[7]) < curr_hit and row[5] != (-1,-1)]
#             x1 = int(ball_locations[0][0])
#             y1 = int(ball_locations[0][1])
#             x2 = int(ball_locations[-1][0])
#             y2 = int(ball_locations[-1][1])
#             start_point = (x1, y1)
#             end_point = (x2, y2)
#             if is_cross(ball_locations, width):
#                 for j in range(prev_hit, curr_hit):
#                     cross_straight_list[j] = [0,start_point, end_point]
#             else:
#                 for j in range(prev_hit, curr_hit):
#                     cross_straight_list[j] = [1,start_point, end_point]
#     return cross_straight_list

def draw_state_info(frame, upper_hit, lower_hit, frame_count, upper_state_list, lower_state_list, upper_box_list, lower_box_list, upper_direction_list, lower_direction_list):
    # 绘制 upper_hit 和 lower_hit 信息
    text = f"Upper Hit: {' '.join(map(str, upper_hit))}\nLower Hit: {' '.join(map(str, lower_hit))}"
    y0, dy = 30, 30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        # cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    upper_state = 'hit' if upper_state_list[frame_count] == 0 else 'return' if upper_state_list[frame_count] == 1 else 'ready' if upper_state_list[frame_count] == 2 else 'approach'
    lower_state = 'hit' if lower_state_list[frame_count] == 0 else 'return' if lower_state_list[frame_count] == 1 else 'ready' if lower_state_list[frame_count] == 2 else 'approach'
    upper_direction_state = 'change' if upper_direction_list[frame_count] == 1 else 'same' if upper_direction_list[frame_count] == 0 else 'None'
    lower_direction_state = 'change' if lower_direction_list[frame_count] == 1 else 'same' if lower_direction_list[frame_count] == 0 else 'None'


    # if lower_state_list[frame_count]==2 and lower_state_list[frame_count-1]==1:
    #     cv2.putText(frame, f'Upper Hit', (10, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # if upper_state_list[frame_count]==2 and upper_state_list[frame_count-1]==1:
    #     cv2.putText(frame, f'Lower Hit', (10, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # # 绘制 upper_state 和 lower_state 信息
    upper_state_color = (0, 0, 255) if upper_state == 'hit' else (0, 255, 0) if upper_state == 'return' else (255, 0, 255) if upper_state == 'ready' else (255, 0, 0)
    lower_state_color = (0, 0, 255) if lower_state == 'hit' else (0, 255, 0) if lower_state == 'return' else (255, 0, 255) if lower_state == 'ready' else (255, 0, 0)
    upper_direction_color = (255, 0, 255) if upper_direction_state == 'change' else (0, 255, 0) if upper_direction_state == 'same' else (255, 0, 0)
    lower_direction_color = (255, 0, 255) if lower_direction_state == 'change' else (0, 255, 0) if lower_direction_state == 'same' else (255, 0, 0)


    cv2.putText(frame, f'{upper_state}', (int(upper_box_list[frame_count][0]), int(upper_box_list[frame_count][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, upper_state_color, 2)
    cv2.putText(frame, f'{lower_state}', (int(lower_box_list[frame_count][0]), int(lower_box_list[frame_count][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, lower_state_color, 2)
    cv2.putText(frame, f'Upper Direction: {upper_direction_state}', (10, y0 + 0 * dy), cv2.FONT_HERSHEY_SIMPLEX, 1, upper_direction_color, 2)
    cv2.putText(frame, f'Lower Direction: {lower_direction_state}', (10, y0 + 1 * dy), cv2.FONT_HERSHEY_SIMPLEX, 1, lower_direction_color, 2)


def draw_ball_boxes_arrows(frame, row, precise_landings, frame_count, previous_positions, ball_list, cross_straight_list):
    ball_location = row[5]
    ball_color = (0, 255, 0) #if frame_count in first_landings else (0, 255, 0)
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
        cv2.circle(overlay, position, 10, ball_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 标记第一次落地的位置并显示编号
    for index, landing_frame in enumerate(precise_landings):
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

    # 绘制箭头
    state = cross_straight_list[frame_count][0]
    start_point = cross_straight_list[frame_count][1]
    end_point = cross_straight_list[frame_count][2]
    if state == 0:
        arrow_color = (255, 0, 0)
        cv2.arrowedLine(frame, start_point, end_point, arrow_color, 3)
    elif state == 1:
        arrow_color = (0, 0, 255)
        cv2.arrowedLine(frame, start_point, end_point, arrow_color, 3)

def find_name_folder(file_path):
    # 找到最后一个 "/" 和 "." 的位置
    start_index = file_path.rfind("/") + 1
    end_index = file_path.rfind(".")

    # 提取中间的内容
    if start_index != -1 and end_index != -1 and start_index < end_index:
        content = file_path[start_index:end_index]
        folder = file_path[:start_index]
        print(f"文件路径名中 '/' 和 '.' 之间的内容是: {content}")
        print(f"文件夹路径是{folder}")
        return content, folder
    else:
        print("找不到 '/' 或 '.'")

def initialize_video_writer(video_file, output_folder):
    # content, folder = find_name_folder(video_file)
    suffix = video_file.split("\\")[-1]
    out_path = os.path.join(output_folder, suffix.replace('.mp4','_3.mp4'))
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    return cap, out, fps, width, height

def main(csv_file,video_file, output_video_folder):
    previous_positions = []
    data = read_csv_file(csv_file)
    cap, out, fps, width, height = initialize_video_writer(video_file, output_video_folder)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rally_change_list, rally_change_intervals = process_rally_changes(data)
    ball_list,upper_box_list,lower_box_list = find_ball_list(data)
    human_hits = get_human_hit(rally_change_intervals, data, height)
    upper_hit, lower_hit, hit_times,upper_hit_intervals,lower_hit_intervals = get_hit_times(rally_change_intervals, human_hits, data)
    upper_state_list, lower_state_list, upper_ret_rea_app_list, lower_ret_rea_app_list = upper_lower_state(upper_hit, lower_hit, upper_hit_intervals, lower_hit_intervals, data)
    upper_direction_list, lower_direction_list = change_direction(upper_ret_rea_app_list, lower_ret_rea_app_list, data)
    cross_straight_list = cross_straight(hit_times, data, width)
    precise_landings = precise_landing(rally_change_intervals, data)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        for row in data:
            if int(row[7]) == frame_count:
                draw_ball_boxes_arrows(frame, row, precise_landings, frame_count, previous_positions,ball_list, cross_straight_list)
            draw_state_info(frame, upper_hit, lower_hit, frame_count, upper_state_list, lower_state_list,upper_box_list,lower_box_list, upper_direction_list, lower_direction_list)

        out.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        # print(frame_count)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"{csv_file}已处理并保存至{output_video_folder}")

if __name__ == "__main__":
    # source_folder = r"D:\Ai_tennis\yolov7_main\test_video\Hard"
    # def process_csvs(source_folder):
    #     for file_name in os.listdir(source_folder):
    #         if os.path.isdir(os.path.join(source_folder, file_name)):
    #             csv_file = os.path.join(source_folder, file_name, file_name + ".csv")
    #             video_file = os.path.join(source_folder, file_name, file_name + ".mp4")
    #             try:
    #                 main(csv_file,video_file)
    #             except Exception as e:
    #                 pass

    def process_csvs(input_csv_folder, input_video_folder, output_video_folder):
        for csv_name in os.listdir(input_csv_folder):
            csv_path = os.path.join(input_csv_folder, csv_name)
            video_name = csv_name.replace('.csv', '.mp4')
            video_path = os.path.join(input_video_folder, video_name)
            try:
                main(csv_path,video_path,output_video_folder)
            except Exception as e:
                pass


    # input_csv_folder = r"D:\Ai_tennis\yolov7_main\test_csv\Grass"
    # input_video_folder = r"D:\Ai_tennis\yolov7_main\test_video\Grass"
    # output_video_folder = r"D:\Ai_tennis\yolov7_main\test_video\landing_model_0202"
    #
    # process_csvs(input_csv_folder, input_video_folder, output_video_folder)

    input_csv_file = r"D:\WeChat Files\wxid_uhmu5l708q7l22\FileStorage\File\2025-02\clay\output_3.csv"
    input_video_file = r"D:\WeChat Files\wxid_uhmu5l708q7l22\FileStorage\File\2025-02\clay\20231011_wth_yt_1.mp4"
    output_video_folder = r"D:\WeChat Files\wxid_uhmu5l708q7l22\FileStorage\File\2025-02\clay"

    main(input_csv_file,input_video_file, output_video_folder)

