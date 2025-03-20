import os
import json
import cv2
from collections import defaultdict
try:
    from .utils import (transform_dict_extended,find_change_points,group_change_points,find_and_merge_non_three_intervals,calculate_change_direction,
                       calculate_speed,find_first_landing_with_window,extract_valid_elements,generate_lists,return_plus,hit_plus,calculate_approach_speed)
    from .utils import draw_approach_speed,draw_ball_speed,draw_change_directions,draw_ball_boxes_arrows,draw_state_info
except:
    from utils import (transform_dict_extended,find_change_points,group_change_points,find_and_merge_non_three_intervals,calculate_change_direction,
                       calculate_speed,find_first_landing_with_window,extract_valid_elements,generate_lists,return_plus,hit_plus,calculate_approach_speed)
    from utils import draw_approach_speed,draw_ball_speed,draw_change_directions,draw_ball_boxes_arrows,draw_state_info

def read_json_file(json_file):
    with open(json_file, 'r') as f:
        datasets = json.load(f)
    data = transform_dict_extended(datasets,['upper_human','upper_actions','real_upper_human',
                                      'lower_human','lower_actions','real_lower_human',
                                      'ball','ball_prediction','curve_status','real_ball','rally_cnt','middle_line','court'])
    action_mapping = {
        3: "waiting",
        2: "overhead",
        0: "left",
        1: "right"
    }
    return data

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
    return cap, fps, width, height

def process_rally_changes(data):
    rally_cnt_list = data['rally_cnt']
    rally_change_list = find_change_points(rally_cnt_list,3)
    rally_change_intervals = group_change_points(rally_change_list,0,len(rally_cnt_list))
    return rally_change_list, rally_change_intervals


def get_hit_times(data,rally_change_intervals):
    ball_list = data['ball']
    upper_action_list = data['upper_actions']
    lower_action_list = data['lower_actions']
    middlle_line_list = data['middle_line']
    hit_time = []
    hit_intervals = []
    upper_hit_intervals= []
    lower_hit_intervals = []
    upper_hit_time = []
    lower_hit_time= []
    for begin,end in rally_change_intervals:
        if ball_list[begin][1] < middlle_line_list[begin]:
            hit_whole_list = upper_action_list[begin:end]
            hit_list = find_and_merge_non_three_intervals(hit_whole_list, begin)
            if hit_list != []:
                upper_hit_intervals.append(hit_list)
                upper_hit_time.append(int(sum(hit_list)/2))
        else:
            hit_whole_list = lower_action_list[begin:end]
            hit_list = find_and_merge_non_three_intervals(hit_whole_list,begin)
            if hit_list != []:
                lower_hit_intervals.append(hit_list)
                lower_hit_time.append(int(sum(hit_list)/2))
        if hit_list != []:
            hit_intervals.append(hit_list)
            hit_time.append(int(sum(hit_list)/2))
    return hit_time,hit_intervals,upper_hit_time,upper_hit_intervals,lower_hit_time,lower_hit_intervals


def calculate_ball_speed(data,upper_hit,lower_hit):
    real_ball_list = data['real_ball']
    hit_speed= defaultdict(list)
    merged_list = sorted(set(upper_hit + lower_hit))
    hit_intervals = []
    n = len(merged_list)
    for i in range(n - 1):
        current = merged_list[i]
        next_val = merged_list[i + 1]
        if current in upper_hit and next_val in upper_hit or current in lower_hit and next_val in lower_hit:
            continue
        hit_intervals.append([current, next_val])
    for hit_interval in hit_intervals:
        one_hit = hit_interval[0]
        another_hit = hit_interval[1]
        valid_speed_list = calculate_speed(real_ball_list[one_hit:another_hit])
        hit_speed[one_hit] = sum(valid_speed_list)/ len(valid_speed_list) if valid_speed_list!=[] else 0
    return hit_speed


def precise_landing(data,rally_change_intervals):
    precise_landings = []
    rally_change_intervals_no_serve = rally_change_intervals.pop(0)
    ball_states = data['curve_status']
    for begin, end in rally_change_intervals:
        ball_state = ball_states[begin:end]
        precise_landing = find_first_landing_with_window(ball_state,5,begin)
        if precise_landing != []:
            precise_landings.append(precise_landing)
    return precise_landings

def cross_straight(data, hit_time):
    cross_line = defaultdict(list)
    ball_list = data['ball']
    ball_valid = extract_valid_elements(ball_list,hit_time)
    ball_valid.append(ball_list[-1])
    hit_times = hit_time
    for idx,hit_time in enumerate(hit_times):
        if idx <= len(hit_times)-1:
            x_coords = [data['court'][hit_time][0], data['court'][hit_time][1],
                        data['court'][hit_time][3], data['court'][hit_time][5]]
            court_width = max(x_coords) - min(x_coords)
            ball_move = abs(ball_valid[idx+1][0] - ball_valid[idx][0]) # 0:cross 1: no cross
            if ball_move > court_width/2 :
                cross_line[hit_time]=[ball_valid[idx+1],ball_valid[idx], 1]
            else:
                cross_line[hit_time] = [ball_valid[idx + 1], ball_valid[idx], 0]
    return cross_line

def upper_lower_state(data, upper_hit, lower_hit, upper_hit_intervals, lower_hit_intervals):
    upper_box = data["upper_human"]
    lower_box = data["lower_human"]
    upper_raw_state,lower_raw_state = generate_lists(len(upper_box),upper_hit,lower_hit)
    upper_state = return_plus(upper_raw_state,upper_box)
    lower_state = return_plus(lower_raw_state,lower_box)
    upper_final_state = hit_plus(upper_state,upper_hit_intervals)
    lower_final_state = hit_plus(lower_state,lower_hit_intervals)
    return upper_final_state,lower_final_state

def change_direction(data,upper_state_list,lower_state_list):
    upper_real_box = data['real_upper_human']
    lower_real_box = data['real_lower_human']
    upper_change_direction_list = calculate_change_direction(upper_state_list,upper_real_box)
    lower_change_direction_list = calculate_change_direction(lower_state_list,lower_real_box)
    return upper_change_direction_list,lower_change_direction_list


def approached_speed(data,upper_state_list,lower_state_list):
    upper_real_box = data['real_upper_human']
    lower_real_box = data['real_lower_human']
    upper_approach_speed = calculate_approach_speed(upper_state_list,upper_real_box)
    lower_approach_speed = calculate_approach_speed(lower_state_list,lower_real_box)
    return upper_approach_speed, lower_approach_speed


def main(csv_file,video_file, output_video_folder, info_json):
    data = read_json_file(csv_file)
    cap, fps, width, height = initialize_video_writer(video_file, output_video_folder)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rally_change_list, rally_change_intervals = process_rally_changes(data)
    hit_time,hit_intervals,upper_hit_time,upper_hit_intervals,lower_hit_time,lower_hit_intervals = get_hit_times(data,rally_change_intervals)
    ball_speed = calculate_ball_speed(data,upper_hit_time,lower_hit_time)
    precise_landings = precise_landing(data, rally_change_intervals)
    cross_straight_dict = cross_straight(data, hit_time)
    upper_state_list, lower_state_list = upper_lower_state(data, upper_hit_time, lower_hit_time, upper_hit_intervals, lower_hit_intervals)
    upper_direction_list, lower_direction_list = change_direction(data,upper_state_list,lower_state_list)
    upper_approach_speed, lower_approach_speed = approached_speed(data,upper_state_list,lower_state_list)
    os.makedirs(output_video_folder, exist_ok=True)
    frame_id = data['frame_id'][0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_video_folder, 'analysis_output.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    draw_ball_heatmap(data, precise_landings,output_video_folder)
    draw_human_heatmap(data,upper_hit_time, output_video_folder,'upper')
    draw_human_heatmap(data, lower_hit_time, output_video_folder,'lower')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        court_location = data['court'][frame_id]
        upper_left_corner, upper_right_corner = (int(court_location[0])-100, int(court_location[1])), (int(court_location[2]), int(court_location[3]))
        middle_left, middle_right = (int(court_location[8])-100, int(court_location[9])), (int(court_location[10]), int(court_location[11]))
        lower_left_corner, lower_right_corner = (int(court_location[4])-100, int(court_location[5])), (int(court_location[6]), int(court_location[7]))

        cv2.putText(frame, 'frame_id: {}'.format(frame_id), (100, 100),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame, 'rally_cnt: {}'.format(data['rally_cnt'][frame_id]), middle_right,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        draw_approach_speed(frame,frame_id,upper_approach_speed,upper_left_corner)
        draw_approach_speed(frame, frame_id, lower_approach_speed, lower_left_corner)
        draw_ball_speed(frame, frame_id,ball_speed,middle_left)
        draw_change_directions(frame, frame_id,upper_direction_list,upper_right_corner)
        draw_change_directions(frame, frame_id, lower_direction_list, lower_right_corner)
        draw_ball_boxes_arrows(frame, frame_id,data,cross_straight_dict,precise_landings)

        draw_state_info(frame, frame_id,data,upper_state_list,lower_state_list,upper_hit_time,lower_hit_time)
        out.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    input_json_file = r"C:\Users\Public\zcj\yolov7\yolov7main\datasets\ball_combine\test_video\grass_3\grass3_filter.json"
    input_video_file = r"C:\Users\Public\zcj\yolov7\yolov7main\datasets\ball_combine\test_video\grass_3\grass3.mp4"
    output_video_folder = r"C:\Users\Public\zcj\yolov7\yolov7main\datasets\ball_combine\test_video\test"
    info_json = 0
    main(input_json_file,input_video_file, output_video_folder,info_json)
