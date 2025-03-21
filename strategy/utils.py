import csv
import ast
import math
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def normalize_keypoints(keypoints, bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    normalized_keypoints = []

    for i in range(len(keypoints)//3):
        x_kp = keypoints[i*3]
        y_kp = keypoints[i*3+1]
        x_norm = (x_kp - x1) / width
        y_norm = (y_kp - y1) / height
        normalized_keypoints.append(x_norm)
        normalized_keypoints.append(y_norm)
    return normalized_keypoints

def filter_ball(data):
    for idx,value in enumerate(data):
        if idx >= 1:
            if data[idx] != [-1,-1] and data[idx-1] != [-1,-1]:
                x1, y1 =data[idx]
                x2, y2 = data[idx-1]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if distance  >  350:
                    data[idx] = [-1,-1]
    return data

def find_closest_point(list1, list2):
    # 计算两点之间的欧几里得距离
    def euclidean_distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # 初始化变量以存储最小距离和对应的点
    min_distance = float('inf')
    closest_point = None

    # 遍历 list1 中的每个点，找到与 list2 中的距离最小的点
    for point in list1:
        distance = euclidean_distance(point, list2)
        if distance < min_distance:
            min_distance = distance
            closest_point = point

    return closest_point

def transform_dict_extended(original_dict, fields_to_extract):
    # 初始化结果字典
    result = {field: [] for field in fields_to_extract}
    result["frame_id"] = []
    # 遍历原始字典
    for frame_id, data in original_dict.items():
        # 提取指定字段的数据
        for field in fields_to_extract:
            field_data = data.get(field, [])
            result[field].append(field_data)

        # 添加 frame_id 信息
        result["frame_id"].append(int(frame_id))
    return result


def find_change_points(data, min_occurrences=3):
    # 统计每个值的出现次数
    value_counts = Counter(data)

    # 找到所有值变化的点
    change_points = []
    for i in range(1, len(data)):
        if data[i] != data[i - 1]:  # 检测到值发生变化
            change_points.append(i)

    # 过滤掉出现次数小于阈值的值
    filtered_change_points = []
    for idx in change_points:
        current_value = data[idx]
        if value_counts[current_value] >= min_occurrences:  # 检查出现次数
            filtered_change_points.append(idx)

    return filtered_change_points

def group_change_points(change_points, start=0, end=100):
    # 插入起始点
    change_points = [start] + change_points
    # 两两分组
    grouped_intervals = [[change_points[i], change_points[i + 1]] for i in range(len(change_points) - 1)]
    grouped_intervals.append([change_points[-1],end])
    return grouped_intervals

def find_and_merge_non_three_intervals(data,start=0):
    intervals = []
    n = len(data)
    i = 0

    # 找到所有连续的、不等于 3 的子区间
    while i < n:
        # 跳过等于 3 的部分
        if data[i] == 3:
            i += 1
            continue

        # 找到不等于 3 的起点
        start_idx = i

        # 继续寻找不等于 3 的终点
        while i < n and data[i] != 3:
            i += 1

        # 确定终点索引
        end_idx = i - 1

        # 检查子区间的长度是否大于 1
        if end_idx - start_idx + 1 > 1:
            intervals.append([start_idx, end_idx])

    # 如果没有找到符合条件的区间，返回空列表
    if not intervals:
        return []

    # 合并所有区间为一个大区间
    starts = [interval[0] for interval in intervals]
    ends = [interval[1] for interval in intervals]
    merged_start = min(starts) +start
    merged_end = max(ends) +start

    return [merged_start, merged_end]

def calculate_speed(position):
    valid_position=[]
    valid_distance=[]
    for i in range(len(position)-1):
        if position[i] != [-1,-1] and position[i+1] != [-1,-1]:
            x1, y1 = position[i]
            x2, y2 = position[i+1]
            valid_position.append([position[i],position[i+1]])
            valid_distance.append(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    return valid_distance

def find_first_landing_with_window(lst, window_size,begin):
    max_count = 0  # 记录滑动窗口中 'landing' 出现的最大次数
    max_start_idx = -1  # 记录最大次数对应的窗口起始索引

    # 滑动窗口遍历列表
    for i in range(len(lst) - window_size + 1):
        window = lst[i:i + window_size]  # 当前窗口
        count = window.count('landing')  # 统计窗口中 'landing' 的数量

        # 更新最大值和对应的起始索引
        if count > max_count:
            max_count = count
            max_start_idx = i

    # 如果没有找到任何 'landing'
    if max_start_idx == -1:
        return []

    # 在最大值对应的窗口内找到第一个 'landing' 的索引
    for idx in range(max_start_idx, max_start_idx + window_size):
        if lst[idx] == 'landing':
            return idx+begin-2


def extract_valid_elements(list1, list2):
    result = []

    for idx in list2:
        # 检查索引是否超出范围
        if idx >= len(list1) or idx < 0:
            raise IndexError(f"Index {idx} is out of range for list1.")

        # 获取当前索引的元素
        current_element = list1[idx]

        # 如果当前元素是 [-1, -1]，向前查找有效元素
        if current_element == [-1, -1]:
            found = False
            for i in range(idx - 1, -1, -1):  # 向前遍历
                if list1[i] != [-1, -1]:
                    current_element = list1[i]
                    found = True
                    break

            # 如果没有找到有效元素，抛出错误
            if not found:
                current_element = [-1,-1]

        # 将有效元素添加到结果中
        result.append(current_element)

    return result

def generate_lists(length, intervals_upper, intervals_lower):
    # 初始化两个列表，初始值为 -1
    list1,list2=[],[]
    intervals1 = intervals_upper.copy()
    intervals2 = intervals_lower.copy()
    for i in range(length):
        if intervals1 ==[] and intervals2 ==[]:
            min_value = [9999, 9999]
        elif intervals1 == [] :
            min_value = [min(intervals2), 1 ]
        elif intervals2 == []:
            min_value = [min(intervals1), 0]
        else:
            if min(intervals1) < min(intervals2):
                min_value = [min(intervals1),0 ]
            else:
                min_value = [min(intervals2), 1 ]
        if i < min_value[0]:
            if len(list1):
                if list1[-1] == "hit":
                    list1.append("return")
                    list2.append("approach")
                elif list2[-1] == "hit":
                    list2.append("return")
                    list1.append("approach")
                else:
                    list1.append(list1[-1])
                    list2.append(list2[-1])
            else:
                list1.append('none')
                list2.append('none')
        else:
            if min_value[1] == 0 :
                list1.append('hit')
                list2.append('approach')
                intervals1.pop(0)
            else:
                list2.append('hit')
                list1.append('approach')
                intervals2.pop(0)

    return list1, list2


def generate_lists_better(length, intervals1, intervals2):
    list1, list2 = [], []
    # 初始化区间指针
    i1, i2 = 0, 0

    for i in range(length):
        # 确定当前最小值及其来源
        if i1 < len(intervals1) and i2 < len(intervals2):
            if intervals1[i1] < intervals2[i2]:
                min_value, source = intervals1[i1], 0
            else:
                min_value, source = intervals2[i2], 1
        elif i1 < len(intervals1):
            min_value, source = intervals1[i1], 0
        elif i2 < len(intervals2):
            min_value, source = intervals2[i2], 1
        else:
            min_value, source = 9999, 2

        # 填充列表
        if i < min_value:
            if not list1:  # 如果列表为空，初始化为 'none'
                list1.append('none')
                list2.append('none')
            elif list1[-1] == 'hit':
                list1.append('hit')
                list2.append('approach')
            elif list2[-1] == 'hit':
                list2.append('hit')
                list1.append('approach')
            else:
                list1.append(list1[-1])
                list2.append(list2[-1])
        else:
            if source == 0:
                list1.append('hit')
                list2.append('approach')
                i1 += 1
            else:
                list2.append('hit')
                list1.append('approach')
                i2 += 1

    return list1, list2

def return_plus(states, box):
    updated_states = states.copy()
    approach_intervals = []
    start = None
    for i, state in enumerate(states):
        if state == 'return':
            if start is None:
                start = i
        else:
            if start is not None:
                approach_intervals.append([start, i - 1])
                start = None
    if start is not None:  # 处理最后一个区间
        approach_intervals.append([start, len(states) - 1])
    for interval in approach_intervals:
        start, end = interval
        current_boxes = box[start:end + 1]
        min_width_index = min(range(len(current_boxes)), key=lambda i: current_boxes[i][2] - current_boxes[i][0])
        min_width_global_index = start + min_width_index
        for i in range(min_width_global_index,end+1):
            updated_states[i] = 'ready'
    return updated_states

def hit_plus(state,intervals,human_action,hand,key='upper'):
    for start, end in intervals:
        filtered_data = [x for x in human_action[start:end + 1] if x not in [-1, 3]]
        if not filtered_data:
            valid_action =  "not sure"
        else:
            element_counts = Counter(filtered_data)
            most_common_element, _ = element_counts.most_common(1)[0]
            if most_common_element == 2:
                valid_action = "overhead"
            elif (most_common_element ==  hand and key =='lower') or (most_common_element !=  hand and key =='upper'):
                valid_action = "backhand"
            else:
                valid_action = "forehand"
        state[start:end + 1] = [valid_action] * (end - start + 1)
    return state

def count_segments(data, item):
    cnt = 0
    for d in data:
        if d == item:
            cnt += 1
    return cnt

def is_in_rectangle(value, rect):
    return rect[0][0] <= value[0] <= rect[1][0] and rect[0][1] <= value[1] <= rect[1][1]

def calculate_angle(v1, v2):
    """计算两个向量之间的夹角（以度为单位）"""
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = math.acos(min(max(cos_theta, -1), 1))  # 防止浮点误差
    return math.degrees(angle_rad)


def calculate_change_direction(states, coordinates,limit_area=0):
    state_list = [False] * len(states)
    intervals = []
    start = None
    for i, state in enumerate(states):
        if state == 'approach' or state == 'return':
            if start is None:
                start = i
        else:
            if start is not None:
                intervals.append([start, i - 1])
                start = None
    if start is not None:  # 处理最后一个区间
        intervals.append([start, len(states) - 1])

    # 计算每个区间的位移向量
    displacements = []
    for interval in intervals:
        start_idx, end_idx = interval
        start_coord = coordinates[start_idx]
        end_coord = coordinates[end_idx]
        displacement = (end_coord[0] - start_coord[0], end_coord[1] - start_coord[1])
        displacements.append(displacement)

    # 比较连续区间的位移夹角
    for i in range(1, len(intervals)):
        prev_displacement = displacements[i - 1]
        curr_displacement = displacements[i]

        # 计算位移距离
        prev_distance = math.sqrt(prev_displacement[0] ** 2 + prev_displacement[1] ** 2)
        curr_distance = math.sqrt(curr_displacement[0] ** 2 + curr_displacement[1] ** 2)

        # 如果任一位移距离小于 5，则状态保持 False
        if prev_distance < limit_area or curr_distance < limit_area:
            continue

        # 计算夹角
        angle = calculate_angle(prev_displacement, curr_displacement)

        # 如果夹角大于 120 度，则状态为 True
        if angle > 120:
            state_list[intervals[i][0]:intervals[i][1] + 1] = [True] * (intervals[i][1] - intervals[i][0] + 1)
        else:
            state_list[intervals[i][0]:intervals[i][1] + 1] = [False] * (intervals[i][1] - intervals[i][0] + 1)

    return state_list

def calculate_ratio(data,key='forehand'):
    count_0 = data.count('key')
    count_1 = data.count('forehand')
    count_2 = data.count('backhand')
    count_3 = data.count('overhead')
    total_count = count_1 + count_2 + count_3
    if total_count == 0:
        return 0
    ratio = count_1 / total_count
    return ratio

def calculate_approach_speed(states, coordinates,total_distance):
    speeds_dict = {}
    for i, state in enumerate(states):
        if state == 'approach':
            x1, y1 = coordinates[i-1]
            x2, y2 = coordinates[i]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_distance.append(distance)
            speeds_dict[i] = round(distance,2)/1
    return speeds_dict,total_distance

def draw_approach_speed(frame,frame_id,approach_speed,coordinate=(100, 100)):
    if frame_id in approach_speed:
        start = end = frame_id
        while start - 1 in approach_speed:
            start -= 1
        while end + 1 in approach_speed:
            end += 1
        values = [approach_speed[i] for i in range(start, end + 1)]
        average_value = sum(values) / len(values)
        cv2.putText(frame, 'Approach', coordinate, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'speed: {}'.format(round(average_value,2)), (coordinate[0], coordinate[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def draw_ball_speed(frame, frame_id,ball_speed,coordinate=(100, 100)):
    max_k = max((k for k in ball_speed if frame_id >= k), default=None)
    if max_k is not None:
        cv2.putText(frame, f'ball_speed:', coordinate,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'{round(ball_speed[max_k],2)}', (coordinate[0], coordinate[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def draw_change_directions(frame, frame_id, direction_list, coordinate=(100, 100)):
    direction = direction_list[frame_id]
    color = (0,255,0) if direction == False else (0,0,255)
    change_count = 0
    for i in range(len(direction_list[0:frame_id+1])):
        if direction_list[i] == True and (i == False or direction_list[i - 1] == False):
            change_count += 1
    cv2.putText(frame, f'change', coordinate,
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f'directions', (coordinate[0], coordinate[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f'count: {change_count}', (coordinate[0], coordinate[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return change_count


def draw_ball_boxes_arrows(frame, frame_id,data,cross_straight_dict,precise_landings):
    ball_location =data['ball']
    ball_color = (0,255,0)
    filtered = [pos for pos in ball_location[:frame_id] if pos != [-1, -1]]
    filtered_list = filtered[-7:] if len(filtered) > 7 else filtered
    prensent_positions = [(int(x), int(y)) for x, y in filtered_list]
    for i, position in enumerate(reversed(prensent_positions)):
        alpha = 1.0 - i * 0.1
        overlay = frame.copy()
        cv2.circle(overlay, position, 10, ball_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for index, landing_frame in enumerate(precise_landings):
        if frame_id >= landing_frame:
            position = ball_location[landing_frame]
            cv2.circle(frame, (int(position[0]), int(position[1])), 10, (0, 0, 0), 2)
            cv2.putText(frame, f'Tag {index + 1}', (int(position[0]), int(position[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    valid_frame_id = None
    for key in sorted(cross_straight_dict.keys(), reverse=True):
        if key <= frame_id and cross_straight_dict[key]:
            valid_frame_id = key
            break
    if valid_frame_id is not None:
        state = cross_straight_dict[valid_frame_id][-1]
        start_point = (int(cross_straight_dict[valid_frame_id][0][0]),int(cross_straight_dict[valid_frame_id][0][1]))
        end_point = (int(cross_straight_dict[valid_frame_id][1][0]),int(cross_straight_dict[valid_frame_id][1][1]))
        arrow_color = (255, 0, 0) if state == 0 else (0, 0, 255)
        cv2.arrowedLine(frame, end_point, start_point, arrow_color, 3)


def draw_state_info(frame, frame_id,data,upper_state_list,lower_state_list,upper_hit_time,lower_hit_time):
    upper_color = (
        (255, 0, 0) if upper_state_list[frame_id] == 'approach' else
        (0, 255, 0) if upper_state_list[frame_id] == 'return' else
        (255, 0, 255) if upper_state_list[frame_id] == 'forehand' else
        (125, 125, 255) if upper_state_list[frame_id] == 'backhand' else
        (0, 0, 255) if upper_state_list[frame_id] == 'overhead' else
        (0, 0, 0) if upper_state_list[frame_id] == 'not sure' else
        (0, 255, 255)
    )
    lower_color = (
        (255, 0, 0) if lower_state_list[frame_id] == 'approach' else
        (0, 255, 0) if lower_state_list[frame_id] == 'return' else
        (255, 0, 255) if upper_state_list[frame_id] == 'forehand' else
        (125, 125, 255) if upper_state_list[frame_id] == 'backhand' else
        (0, 0, 255) if upper_state_list[frame_id] == 'overhead' else
        (0, 0, 0) if upper_state_list[frame_id] == 'not sure' else
        (0, 255, 255)
    )
    upper_box = data['upper_human'][frame_id]
    lower_box = data['lower_human'][frame_id]
    upper_state = upper_state_list[frame_id]
    lower_state = lower_state_list[frame_id]
    cv2.rectangle(frame, (int(upper_box[0]), int(upper_box[1])), (int(upper_box[2]), int(upper_box[3])), upper_color, 2)
    cv2.rectangle(frame, (int(lower_box[0]), int(lower_box[1])), (int(lower_box[2]), int(lower_box[3])), lower_color, 2)
    cv2.putText(frame, f'{upper_state}', (int(upper_box[0]), int(upper_box[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, upper_color, 2)
    cv2.putText(frame, f'{lower_state}', (int(lower_box[0]), int(lower_box[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, lower_color, 2)
    if frame_id in upper_hit_time:
        cv2.putText(frame, 'Hitting', (int(upper_box[2]), int(upper_box[3])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, upper_color, 2)
    elif frame_id in lower_hit_time:
        cv2.putText(frame, 'Hitting', (int(lower_box[2]), int(lower_box[3])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,lower_color, 2)


def plot_heatmap(frequency_matrix, title="Heatmap", cmap="viridis",output="heatmap.png"):
    plt.figure(figsize=(7, 12))
    plt.imshow(frequency_matrix, cmap=cmap, aspect="auto", origin="upper", interpolation="nearest")
    # Add color bar
    # cbar = plt.colorbar()
    # cbar.set_label("Point Count")
    plt.title(title)
    plt.xlabel("Grid Columns")
    plt.ylabel("Grid Rows")
    plt.savefig(output)
    plt.close()

def compute_frequency_matrix(M, N, points, m, n):
    # Initialize the frequency matrix with zeros
    frequency_matrix = [[0 for _ in range(n)] for _ in range(m)]
    # Compute the size of each grid cell
    cell_height = M / m
    cell_width = N / n
    # Iterate over all points and count them in the corresponding grid cell
    for x, y in points:
        # Check if the point is within the bounds of the rectangle
        if 0 <= x < N and 0 <= y < M:
            # Determine the grid cell indices
            col = int(x // cell_width)
            row = int(y // cell_height)
            # Ensure the indices are within bounds (due to floating-point precision issues)
            if 0 <= row < m and 0 <= col < n:
                frequency_matrix[row][col] += 1
    return frequency_matrix

def draw_human_heatmap(data,hit_time,output_video_folder,side='upper'):
    output_path = os.path.join(output_video_folder, '{}_human_hit_heatmap.png'.format(side))
    M, N =  3500,1600
    m, n =  7,4
    human_hit_location =[]
    for i in hit_time:
        if side == 'upper':
            human_hit_location.append(data['real_upper_human'][i])
        elif side == 'lower':
            human_hit_location.append(data['real_lower_human'][i])
    human_matrix = np.array(compute_frequency_matrix(M, N, human_hit_location, m, n))
    plot_heatmap(human_matrix, title="Human", output=output_path)

def draw_ball_heatmap(data,precise_landings,output_video_folder):
    M, N =  3500,1600
    m, n =  35,16
    output_path = os.path.join(output_video_folder, 'ball_heatmap.png')
    ball_landing_location =[]
    for i in precise_landings:
        if data['ball'][i] == [-1,-1]:
            j = i - 1
            while j >= 0 and data['ball'][j] == [-1, -1]:
                j -= 1
            if j >= 0:
                ball_landing_location.append(data['real_ball'][j])
            else:
                ball_landing_location.append([500,500])
        else:
            ball_landing_location.append(data['real_ball'][i])
    ball_landing_matrix = np.array(compute_frequency_matrix(M, N, ball_landing_location, m, n))
    plot_heatmap(ball_landing_matrix, title="Ball", output=output_path)
