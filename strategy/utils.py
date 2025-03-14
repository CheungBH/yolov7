import csv
import ast
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

import math

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
