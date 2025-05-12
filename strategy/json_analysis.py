import os
import json
import cv2
from collections import defaultdict
# from finish_analysis import *
# from server_01 import ServeChecker
from numpy.array_api import result_type
try:
    from .utils import *
except:
    from utils import *

def read_json_file(json_file):
    data = {}
    root = os.path.dirname(json_file)
    split_json = os.path.join(root,'split_file')
    split_dir,highlight_list = split_json_by_ball(json_file,split_json)
    for idx,file in enumerate(split_dir):
        file_f = os.path.join(split_json,file)
        with open(file_f, 'r') as f:
            datasets = json.load(f)
        data[idx] = transform_dict_extended(datasets,['upper_human','upper_actions','real_upper_human',
                                          'lower_human','lower_actions','real_lower_human',"lower_human_kps_pred",
                                    "upper_human_kps_pred",'ball','ball_prediction','curve_status',
                                                      'real_ball','rally_cnt','middle_line','court','ocr_result','lower_human_kps','upper_human_kps'])
        data[idx]['ball'] = filter_ball(data[idx]['ball'])
        data[idx]['real_ball'] = filter_ball(data[idx]['real_ball'])

    human_kps_pred = {
        0: "left to right",
        1: "right to left",
        2: "overhead",
        3: "unsure",
    }
    return data

def read_info_json_file(json_file):
    with open(json_file, 'r') as f:
        datasets = json.load(f)
    serve_side = datasets['serve_side']
    serve_part = datasets['serve_part']
    upper_hand = datasets['upper_hand'] # 0 : right 1 : left
    lower_hand = datasets['lower_hand']
    return serve_side,serve_part,upper_hand,lower_hand

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
                upper_hit_time.append(int(hit_list[1]*3/4+hit_list[0]/4)) # kps + 0.7 interval
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



def calculate_ball_speed(data,upper_hit,lower_hit, valid_ratio=0.4):
    ball_speed_list =[]
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

        # begin_hit, end_hit = (int(one_hit + valid_ratio * (another_hit - one_hit)),
        #                       int(one_hit + (1 - valid_ratio) * (another_hit - one_hit)))
        # valid_speed_list = calculate_speed(real_ball_list[begin_hit:end_hit])

        ball_speed_list.append(valid_speed_list)

        hit_speed[one_hit] = sum(valid_speed_list)/ len(valid_speed_list) if valid_speed_list!=[] else 0
    return hit_speed,ball_speed_list


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

def finish_analysis(data,serve_side,precise_landings):
    serve_list = ['upper', 'lower']
    #court = [[290, 560], [1380, 2930]]# double
    upper_court = [[424,560],[1240,1745]]
    lower_court = [[424,1745],[1240,2930]]
    if not precise_landings:
        return "Not sure",[-1,-1]
    real_ball = data.get('real_ball', [])
    last_landing = None
    for i in range(precise_landings[-1], -1, -1):
        if real_ball[i] != [-1, -1]:
            last_landing = real_ball[i]
            break
    if last_landing is None:
        return "Not sure",[-1,-1]
    if (serve_side == 'upper' and is_in_rectangle(last_landing, lower_court)) or (serve_side == 'lower' and is_in_rectangle(last_landing, upper_court)) :
        return serve_side,last_landing
    else:
        return serve_list[1 - serve_list.index(serve_side)],last_landing

def cross_straight(data, hit_time):
    cross_line = defaultdict(list)
    ball_list = data['ball']
    ball_valid = extract_valid_elements(ball_list,hit_time)
    # ball_valid.append(ball_list[-1])
    for idx,hit_t in enumerate(hit_time):
        if idx <= len(hit_time)-2:
            x_coords = [data['court'][hit_t][0], data['court'][hit_t][1],
                        data['court'][hit_t][3], data['court'][hit_t][5]]
            court_width = max(x_coords) - min(x_coords)
            ball_move = abs(ball_valid[idx+1][0] - ball_valid[idx][0]) # 0:cross 1: no cross
            if ball_move > court_width/2 :
                cross_line[hit_t]=[ball_valid[idx+1],ball_valid[idx], 1]
            else:
                cross_line[hit_t] = [ball_valid[idx+1], ball_valid[idx], 0]
    return cross_line

def upper_lower_state(data, upper_hit, lower_hit, upper_hit_intervals, lower_hit_intervals,upper_hand,lower_hand,precise_landings):
    upper_box = data["upper_human"]
    lower_box = data["lower_human"]
    ball_states = data['curve_status']
    upper_action = data['upper_actions']
    lower_action = data['lower_actions']
    upper_kps = data['upper_human_kps_pred']
    lower_kps = data['lower_human_kps_pred']
    upper_raw_state,lower_raw_state = generate_lists(len(upper_box),upper_hit,lower_hit)
    upper_state = return_plus(upper_raw_state,upper_box)
    lower_state = return_plus(lower_raw_state,lower_box)
    upper_ball_landing, lower_ball_landing = generate_another_list(upper_hit,lower_hit)
    upper_final_state = hit_plus(data,upper_state,upper_hit_intervals,upper_ball_landing,upper_action,upper_kps,upper_hand,'upper',ball_states,precise_landings)
    lower_final_state = hit_plus(data,lower_state,lower_hit_intervals,lower_ball_landing,lower_action,lower_kps,lower_hand,'lower',ball_states,precise_landings)
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
    total_distance_upper = []
    total_distance_lower = []
    upper_approach_speed = calculate_approach_speed(upper_state_list,upper_real_box,total_distance_upper)
    lower_approach_speed = calculate_approach_speed(lower_state_list,lower_real_box,total_distance_lower)
    return upper_approach_speed, lower_approach_speed,total_distance_upper,total_distance_lower

def easy_diff_shot(data,precise_landings):
    ball_location = data ['real_ball']
    upper_real_box = data['real_upper_human']
    lower_real_box = data['real_lower_human']
    shot_degree={}
    thre = 400
    middle_line = 1745

    for landing in precise_landings:
        if ball_location[landing] == [-1,-1]:
            i = landing -1
            while i >= 0 and ball_location[i] == [-1,-1]:
                i-=1
            if i >=0:
                ball_valid = ball_location[i]
            else:
                ball_valid = [500,500]
        else:
            ball_valid = ball_location[landing]
        # ball_valid = ball_location[landing] if ball_location[landing] != [-1,-1] else ball_location[landing-1]
        upper_valid = upper_real_box[landing]
        lower_valid = lower_real_box[landing]

        if ball_valid[1] < middle_line:
            distance,x = calculate_distance(upper_valid,ball_valid)
        else:
            distance,x = calculate_distance(lower_valid,ball_valid)
        if x > thre:
            shot_degree[landing] = 'diffcult'
        else:
            shot_degree[landing] = 'easy'
    return shot_degree
def ocr_detect(data,start_frame_id,end_frame_id):
    # ocr_result = data['ocr_result'][start_frame_id:end_frame_id+1]
    ocr_result = data['ocr_result'][0:end_frame_id-start_frame_id + 1]
    player1_score = []
    player2_score = []
    for game_points in ocr_result:
        if len(game_points) == 2:
            if game_points[1]!=[]:
                player1_score.append(game_points[0][1])
                player2_score.append(game_points[1][1])
    if player1_score == [] or player2_score == []:
        real_score =[]
    else:
        counter1 = Counter(player1_score)
        max_count1 = max(counter1.values())
        most_common_elements1 = [element for element, count in counter1.items() if count == max_count1]
        counter2 = Counter(player2_score)
        max_count2 = max(counter2.values())
        most_common_elements2 = [element for element, count in counter2.items() if count == max_count2]
        real_score = [most_common_elements1,most_common_elements2]
    return real_score
def find_team(data,frame,frame_id):
    lower_kps = data['lower_human_kps']
    upper_kps = data['upper_human_kps']

    lower_clothes = [[int(lower_kps[frame_id][15]), int(lower_kps[frame_id][16])],
                     [int(lower_kps[frame_id][18]), int(lower_kps[frame_id][19])]]

    lower_pants = [[int(lower_kps[frame_id][33]), int(lower_kps[frame_id][34])],
                   [int(lower_kps[frame_id][36]), int(lower_kps[frame_id][37])]]

    upper_clothes = [[int(upper_kps[frame_id][16]), int(upper_kps[frame_id][15])],
                     [int(upper_kps[frame_id][19]), int(upper_kps[frame_id][18])]]

    upper_pants = [[int(upper_kps[frame_id][34]), int(upper_kps[frame_id][33])],
                   [int(upper_kps[frame_id][37]), int(upper_kps[frame_id][36])]]

    # 获取两个点的像素值 (BGR)
    color1 = frame[lower_clothes[0]]  # 第一个点的颜色
    color2 = frame[lower_clothes[1]]
    color3 = frame[lower_pants[1]]
    color4 = frame[upper_clothes[1]]
    cv2.circle(frame, lower_clothes[0], 10, (255, 0, 0), -1)
    cv2.circle(frame, lower_clothes[1], 10, (255, 0, 0), -1)
    cv2.circle(frame, lower_pants[0], 10, (255, 0, 0), -1)
    cv2.circle(frame, lower_pants[1], 10, (255, 0, 0), -1)
    # cv2.circle(frame, upper_clothes[0], 10, (255, 0, 0), -1)
    # cv2.circle(frame, upper_clothes[1], 10, (255, 0, 0), -1)
    # cv2.circle(frame, upper_pants[0], 10, (255, 0, 0), -1)
    # cv2.circle(frame, upper_pants[1], 10, (255, 0, 0), -1)
    variance1 = np.mean((color1 - color2) ** 2)
    variance2 = np.mean((color1 - color3) ** 2)
    variance3 = np.mean((color1 - color4) ** 2)
    if frame_id>0:
        color5 =frame[[int(lower_kps[frame_id-1][16]), int(lower_kps[frame_id-1][15])]]
        variance4 = np.mean((color1 - color5) ** 2)
    a=1

def write_json(path,data,serve_side,game_winner,last_landing,fps,ball_speed_list,upper_state_list, lower_state_list,
               upper_change_times,lower_change_times,total_receiver_distance_upper,total_receiver_distance_lower,
               upper_hit_time,lower_hit_time,shot_degree,precise_landings):
    box_assets = {}
    upper_court = [[424,560],[1240,1745]]
    lower_court = [[424,1745],[1240,2930]]

    real_upper_human = data['real_upper_human']
    real_lower_human = data['real_lower_human']

    box_assets['rally_cnt'] = data['rally_cnt'][-1] - data['rally_cnt'][0]
    box_assets['serve_player'] = serve_side
    box_assets['success_shot'] = True if serve_side == game_winner else False
    if serve_side =='upper' and is_in_rectangle(last_landing,upper_court)  or serve_side =='lower' and is_in_rectangle(last_landing,lower_court):
        box_assets['Ending_condition'] = 'Downnet'
    else:
        box_assets['Ending_condition'] = 'outside'
    box_assets['Point_duration'] = data['frame_id'][-1]/fps

    flattened_list = [item for sublist in ball_speed_list for item in sublist]
    box_assets['ball_average_speed(km/h)'] = (sum(flattened_list)/len(flattened_list))*fps/100*3.6 if len(flattened_list) != 0  else 0
    box_assets['ball_max_speed(km/h)'] = max(flattened_list) * fps / 100 * 3.6 if len(flattened_list) != 0  else 0

    box_assets['total_receiver_distance_upper(m)'] = sum(total_receiver_distance_upper)/100
    box_assets['total_receiver_distance_lower(m)'] = sum(total_receiver_distance_lower)/100
    box_assets['total_receiver_speed_upper(m/s)'] = (sum(total_receiver_distance_upper)/len(total_receiver_distance_upper))*fps/100 if len(total_receiver_distance_upper) != 0  else 0
    box_assets['total_receiver_speed_lower(m/s)'] = (sum(total_receiver_distance_lower)/len(total_receiver_distance_lower))*fps/100 if len(total_receiver_distance_lower) != 0  else 0

    box_assets['change_direction_times_upper'] = upper_change_times
    box_assets['change_direction_times_lower'] = lower_change_times

    valid_distance_upper = calculate_speed(real_upper_human)
    valid_distance_lower = calculate_speed(real_lower_human)
    box_assets['total_moving_distance_upper(m)'] = sum(valid_distance_upper) / 100
    box_assets['total_moving_distance_lower(m)'] = sum(valid_distance_lower) / 100
    box_assets['total_moving_speed_upper(m/s)'] = (sum(valid_distance_upper)/len(valid_distance_upper)) * fps / 100 if len(valid_distance_upper) != 0  else 0
    box_assets['total_moving_speed_lower(m/s)'] = (sum(valid_distance_lower)/len(valid_distance_lower)) * fps / 100 if len(valid_distance_lower) != 0  else 0

    upper_hit_list = [upper_state_list[t] for t in upper_hit_time]
    lower_hit_list = [lower_state_list[t] for t in lower_hit_time]
    upper_forehand = count_segments(upper_hit_list, 'forehand')
    upper_backhand = count_segments(upper_hit_list, 'backhand')
    lower_forehand = count_segments(lower_hit_list, 'forehand')
    lower_backhand = count_segments(lower_hit_list, 'backhand')

    box_assets['upper_forehand_ratio'] = upper_forehand/len(upper_hit_list) if len(upper_hit_list) != 0  else 0
    box_assets['upper_backhand_ratio'] = upper_backhand/len(upper_hit_list) if len(upper_hit_list) != 0  else 0
    box_assets['lower_forehand_ratio'] = lower_forehand/len(lower_hit_list) if len(lower_hit_list) != 0  else 0
    box_assets['lower_backhand_ratio'] = lower_backhand/len(lower_hit_list) if len(lower_hit_list) != 0  else 0
    if precise_landings:
        box_assets['shot_degree'] = shot_degree[precise_landings[-1]] if precise_landings[-1] in shot_degree else 'not sure'

    with open(path, 'w') as f:
        json.dump(box_assets, f, indent=4)



def main(csv_file,video_file, output_video_folder, info_json):
    os.makedirs(output_video_folder,exist_ok=True)
    serve_side,serve_part,upper_hand,lower_hand = read_info_json_file(info_json)
    data_dict = read_json_file(csv_file)
    # server_rally = ServeChecker(serve_side,serve_part)
    # server_rally.process(data)

    cap, fps, width, height = initialize_video_writer(video_file, output_video_folder)
    video_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    pixel2cm_ratio = fps / 100
    ball_matrix = []
    upper_human_matrix = []
    lower_human_matrix = []
    events = []
    speeds = {}
    game_score_dict= {}
    set_game = 0
    for key,data in sorted(data_dict.items()):
        start_frame_id = data['frame_id'][0]
        end_frame_id = data['frame_id'][-1]
        if start_frame_id >= end_frame_id - 30:
            continue
        total_frame = min(video_frame,len(data['frame_id']))
        rally_change_list, rally_change_intervals = process_rally_changes(data)
        hit_time,hit_intervals,upper_hit_time,upper_hit_intervals,lower_hit_time,lower_hit_intervals \
            = get_hit_times(data,rally_change_intervals)
        ball_speed,ball_speed_list = calculate_ball_speed(data,upper_hit_time,lower_hit_time)
        precise_landings = precise_landing(data, rally_change_intervals)
        game_winner,last_landing = finish_analysis(data,serve_side,precise_landings)
        cross_straight_dict = cross_straight(data, hit_time)
        upper_state_list, lower_state_list \
            = upper_lower_state(data, upper_hit_time, lower_hit_time, upper_hit_intervals,
                                                               lower_hit_intervals,upper_hand,lower_hand,precise_landings)
        upper_direction_list, lower_direction_list = change_direction(data,upper_state_list,lower_state_list)
        upper_approach_speed, lower_approach_speed,total_receiver_distance_upper,total_receiver_distance_lower \
            = approached_speed(data,upper_state_list,lower_state_list)
        upper_change_times, lower_change_times = 0,0
        shot_degree = easy_diff_shot(data,precise_landings)

        os.makedirs(output_video_folder, exist_ok=True)
        frame_id = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        game_score = ocr_detect(data, start_frame_id, end_frame_id)
        game_score_dict[end_frame_id+1] = game_score

        if game_score ==[] or (game_score[0][0]=="0" and game_score[1][0]=="0"):
            set_game += 1
        output_path = os.path.join(output_video_folder, 'game_{}/{}_{}_{}'.format(set_game,start_frame_id,end_frame_id,game_score))

        output_video_path = os.path.join(output_video_folder, 'game_{}/{}_{}_{}/analysis_output.mp4'.format
        (set_game,start_frame_id,end_frame_id,game_score))
        os.makedirs(output_path, exist_ok=True)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        ball_matrix.append(draw_ball_heatmap(data, precise_landings,output_path))
        upper_human_matrix.append(draw_human_heatmap(data,upper_hit_time, output_path,'upper'))
        lower_human_matrix.append(draw_human_heatmap(data, lower_hit_time, output_path,'lower'))
        output_json_path = os.path.join(output_path, 'analysis_output.json'.format(start_frame_id))
        time_line_path = os.path.join(output_video_folder, 'timeline.jpg')

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id >= total_frame:
                break

            court_location = data['court'][frame_id]
            upper_left_corner, upper_right_corner = (int(court_location[0])-100, int(court_location[1])), (int(court_location[2]), int(court_location[3]))
            middle_left, middle_right = (int(court_location[8])-100, int(court_location[9])), (int(court_location[10]), int(court_location[11]))
            lower_left_corner, lower_right_corner = (int(court_location[4])-100, int(court_location[5])), (int(court_location[6]), int(court_location[7]))
            # find_team(data,frame,start_frame_id+frame_id)
            cv2.putText(frame, 'frame_id: {}'.format(frame_id+start_frame_id), (100, 100),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, 'rally_cnt: {}'.format(data['rally_cnt'][frame_id]), middle_right,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, 'Game_score: {}'.format(game_score), (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if precise_landings:
                if frame_id >= precise_landings[-1]:
                    cv2.putText(frame, '{} win'.format(game_winner), (500, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            draw_approach_speed(frame,frame_id,upper_approach_speed[0],upper_left_corner, pixel2cm_ratio)
            draw_approach_speed(frame, frame_id, lower_approach_speed[0], lower_left_corner, pixel2cm_ratio)
            draw_ball_speed(frame, frame_id,ball_speed,middle_left, pixel2cm_ratio*3.6)
            upper_change_times = draw_change_directions(frame, frame_id,upper_direction_list,upper_right_corner)
            lower_change_times = draw_change_directions(frame, frame_id, lower_direction_list, lower_right_corner)
            draw_ball_boxes_arrows(frame, frame_id,data,cross_straight_dict,precise_landings)
            draw_state_info(frame, frame_id,data,upper_state_list,lower_state_list,upper_hit_time,lower_hit_time,hit_time, fps)
            out.write(frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_id += 1
        image_path = r"C:\Users\Public\zcj\yolov7\yolov7main\strategy\court\court_reference.png"
        draw_real_ball_boxes_arrows(image_path, data, upper_hit_time,lower_hit_time, precise_landings,output_path=os.path.join(output_path,'review.jpg'))
        events.append([frame_id+start_frame_id, game_winner])
        flattened_list = [item for sublist in ball_speed_list for item in sublist]
        speeds[frame_id+start_frame_id] = {'average_speed': (sum(flattened_list) / len(
            flattened_list)) * fps / 100 * 3.6 if len(flattened_list) != 0 else 0,
                 'max_speed': max(flattened_list) * fps / 100 * 3.6 if len(
                     flattened_list) != 0 else 0}

        write_json(output_json_path,data,serve_side,game_winner,last_landing,fps,ball_speed_list,upper_state_list, lower_state_list,
                   upper_change_times,lower_change_times,total_receiver_distance_upper,total_receiver_distance_lower,upper_hit_time,lower_hit_time, shot_degree, precise_landings)
    draw_timeline(events, speeds, fps, time_line_path, game_score_dict)
        # plot_heatmap(sum(ball_matrix))
        # plot_heatmap(sum(upper_human_matrix))
        # plot_heatmap(sum(lower_human_matrix))
        # upper_human_points,lower_human_points,ball_points = upper_lower_ball_matrix(data, upper_hit_time,lower_hit_time,precise_landings)
        # upper_human_matrix.extend(upper_human_points)
        # lower_human_matrix.extend(lower_human_points)
        # ball_matrix.extend(ball_points)
        # draw_heatmap(image_path, upper_human_points, box_size=100,output_path = os.path.join(output_path, 'upper_human_heatmap.png'))
        # draw_heatmap(image_path, lower_human_points, box_size=100,output_path = os.path.join(output_path, 'lower_human_heatmap.png'))
        # draw_heatmap(image_path, ball_points, box_size=100,output_path = os.path.join(output_path, 'ball_heatmap.png'))
    # draw_heatmap(image_path, upper_human_matrix, box_size=100,
    #              output_path=os.path.join(output_video_folder, 'upper_human_heatmap.png'))
    # draw_heatmap(image_path, lower_human_matrix, box_size=100,
    #              output_path=os.path.join(output_video_folder, 'lower_human_heatmap.png'))
    # draw_heatmap(image_path, ball_matrix, box_size=100, output_path=os.path.join(output_video_folder, 'ball_heatmap.png'))
    # plot_heatmap(sum(ball_matrix))
    # plot_heatmap(sum(upper_human_matrix))
    # plot_heatmap(sum(lower_human_matrix))
    plot_heatmap_with_gaps(sum(ball_matrix), title="Ball", output=os.path.join(output_video_folder, 'ball_human_hit_heatmap.png'))
    plot_heatmap_with_gaps(sum(upper_human_matrix), title="Human", output=os.path.join(output_video_folder, 'upper_human_hit_heatmap.png'))
    plot_heatmap_with_gaps(sum(lower_human_matrix), title="Human", output=os.path.join(output_video_folder, 'lower_human_hit_heatmap.png'))



    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    input_json_file = r"C:\Users\Public\zcj\yolov7\yolov7main\output\20231011_kh_yt_18_2\20231011_kh_yt_18_2_filter.json"
    input_video_file = r"C:\Users\Public\zcj\yolov7\yolov7main\output\20231011_kh_yt_18_2\20231011_kh_yt_18_2.mp4"
    output_video_folder = 'output/20231011_kh_yt_18_2'
    info_json = r"C:\Users\Public\zcj\yolov7\yolov7main\output\top100_97\info.json"
    # input_json_file = "output/kh_1/20231011_kh_yt_2_filter.json"
    main(input_json_file,input_video_file, output_video_folder,info_json)
