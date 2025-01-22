from collections import defaultdict
import cv2
import csv
import os


class RallyChecker:
    def __init__(self, ball_init_toward="up", ball_last_hit="lower", central_x=640, central_y=300,top_y=0,bottom_y=720,
                 serve_condition="", **kwargs):
        self.rallying = False
        self.balls_existing = []
        self.ball_locations = []
        self.balls= []
        self.bounce = defaultdict(list)
        self.player_boxes = defaultdict(list)
        self.player_actions = defaultdict(list)
        self.middle_upper_y, self.middle_lower_y = central_y-60, central_y+30
        self.central_y, self.central_x = central_y, central_x
        self.top_y, self.bottom_y = top_y,bottom_y
        self.max_no_return_duration = 20
        self.down_net_duration = 10
        self.down_net_threshold = 0.8
        self.rally_threshold = 0
        self.rally_cnt = 0
        self.current_ball_towards = ball_init_toward
        self.current_ball_position = ball_last_hit
        self.recent_ball = 10
        self.max_ball_change_directio_pixel = 2
        self.ball_changing_max_threshold = 0.5
        self.rally_change_threshold = 3
        self.ball_towards_status = []
        self.ball_position = ball_last_hit
        self.ball_positions = []
        self.end_situation = "Not ending"
        self.serve_condition = serve_condition
        self.frame_cnt=[]
        self.landing = []
        self.rally_cnt_list = []
        self.insidecourt_list = []
        self.ball_realbox_list = []
        self.human_realbox_list = []

    def update_line(self, central_x, central_y):
        self.central_x, self.central_y = int(central_x), int(central_y)
        self.middle_upper_y,self.middle_lower_y = central_y-60, central_y+30

    def check_status(self, status, key):
        state_ls = [i == key for i in status]
        ratio = sum(state_ls) / len(state_ls)
        return ratio >= 0.5

    def update_ball_status(self, ball_locations):
        ball_location = "lower" if ball_locations[1] > self.central_y else "upper"
        self.ball_positions.append(ball_location)
        recent_ball_position = self.ball_positions[-self.recent_ball:]

        recent_ball_locations = self.ball_locations[-self.recent_ball:]
        ball_y_changing = [recent_ball_locations[idx+1][1] - recent_ball_locations[idx][1]
                           for idx in range(self.recent_ball-1)]
        ball_change_over_threshold = [abs(y_change) > self.max_ball_change_directio_pixel
                                     for y_change in ball_y_changing]
        ball_y_changing = [0 if abs(y_change) < self.max_ball_change_directio_pixel else
                           y_change for y_change in ball_y_changing]
        ball_y_changing = ["up" if y_change < 0 else "down" for y_change in ball_y_changing]
        if ball_change_over_threshold:
            ball_position_upper_flag = self.check_status(recent_ball_position, "upper")
            # ball_position_lower_flag = self.check_status(recent_ball_position, "lower")
            current_ball_position = "upper" if ball_position_upper_flag else "lower"

            ball_towards_up = self.check_status(ball_y_changing, "up")
            self.current_ball_towards = "up" if ball_towards_up else "down"

            if self.current_ball_towards == "up" and current_ball_position == "upper" and self.ball_position == "lower":
                self.rally_cnt += 1
                self.ball_position = "upper"
            elif self.current_ball_towards == "down" and current_ball_position == "lower" and self.ball_position == "upper":
                self.rally_cnt += 1
                self.ball_position = "lower"
            else:
                self.ball_position = current_ball_position


    def check_rally_status(self):
        self.check_end_rally() if self.rallying else self.check_begin_rally()

    def check_begin_rally(self):
        out_bound_cnt = min(self.max_no_return_duration, len(self.ball_locations))
        ball_existing = self.balls_existing[-out_bound_cnt:]
        if sum(ball_existing) / len(ball_existing) > self.rally_threshold:
            self.rallying = True
            self.end_situation = "Not ending"
            self.rally_cnt = 0

    def check_end_rally(self):
        out_bound_cnt = min(self.max_no_return_duration, len(self.ball_locations))
        #ball_existing = self.balls_existing[-out_bound_cnt:]
        ball_existing = self.balls_existing[-1]
        if ball_existing:
            '''
            if self.ball_locations[-1][1] < max(self.top_y,(self.player_boxes['upper'][-1][1]+self.player_boxes['upper'][-1][3])/2) and self.ball_locations[-1][1]>min(self.top_y,(self.player_boxes['upper'][-1][1]+self.player_boxes['upper'][-1][3])/2):
                if abs(self.ball_locations[-1][0] - (self.player_boxes['upper'][-1][0]+self.player_boxes['upper'][-1][2])/2) >3*abs(self.player_boxes['upper'][-1][0]-self.player_boxes['upper'][-1][2]):
                    self.rallying = False
                    self.end_situation = "Out bound"
            elif self.ball_locations[-1][1] < max(self.bottom_y,self.player_boxes['lower'][-1][3]) and self.ball_locations[-1][1]>min(self.bottom_y,self.player_boxes['lower'][-1][3]):
                 if abs(self.ball_locations[-1][0] - (self.player_boxes['lower'][-1][0]+self.player_boxes['lower'][-1][2])/2) >2*abs(self.player_boxes['lower'][-1][0]-self.player_boxes['lower'][-1][2]):
                    self.rallying = False
                    self.end_situation = "Out bound"
            '''
            if len(self.ball_locations) > self.down_net_duration:
                # cnt_chosen = min(len(self.ball_locations), self.down_net_duration)
                #balls = self.ball_locations[-self.down_net_duration:]
                # Check the ball in the middle area
                if self.player_actions['upper'][-1] != 'overhead' and self.player_actions['lower'][-1] != 'overhead':
                    self.balls.append(self.ball_locations[-1])
                    self.balls = self.balls[-self.down_net_duration:] if len(self.balls) > self.down_net_duration else self.balls
                    ball_in_middle = [ball[1] > self.middle_upper_y and ball[1] < self.middle_lower_y for ball in self.balls]
                    if sum(ball_in_middle) > self.down_net_duration*self.down_net_threshold:
                        #self.rallying = False
                        self.end_situation = "Down net"

    def get_box(self):
        return [self.player_boxes[position][-1] for position in ["upper", "lower"]]

    def get_action(self):
        return [self.player_actions[position][-1] for position in ["upper", "lower"]]

    def get_ball(self):
        if not self.balls_existing[-1]:
            return None
        else:
            return self.ball_locations[-1]
    def find_center_points(self,data,ratio=1/2,frame_cnt=[]):
        for i in range(1, len(data) - 1):
            start = max(0, i - 2)
            end = min(len(data), i + 2)
            if data[i] == 0 :
                if sum(data[start:end]) >= 3:
                    data[i] = 1
            '''else:
                if sum(data[start:end]) < 3:
                    data[i] = 0'''
        start = None
        centers = []
        center_frame_cnt = []

        for i, value in enumerate(data):
            if value == 1 and start is None:
                start = i
            elif value == 0 and start is not None:
                end = i - 1
                center = int((start*(1-ratio) + end *ratio))
                centers.append(center)
                center_frame_cnt.append(frame_cnt[center])
                start = None
        if start is not None:
            end = len(data) - 1
            center = int((start*(1-ratio) + end *ratio))
            centers.append(center)
            center_frame_cnt.append(frame_cnt[center])
        return centers,center_frame_cnt
    def find_landing(self,data,frame_cnt=[]):
        landing_frame_cnt=[]
        for i, value in enumerate(data):
            if value =='landing':
                landing_frame_cnt.append(frame_cnt[i]-5)
        return landing_frame_cnt


    def output_csv(self,base_path):
        # 定义要处理的列表
        lists = [
            self.player_actions['upper'],
            self.player_boxes['upper'],
            self.player_actions['lower'],
            self.player_boxes['lower'],
            self.landing,
            self.ball_locations,
            self.rally_cnt_list,  # bug
            self.frame_cnt,
            self.human_realbox_list,
            self.ball_realbox_list
        ]
        max_length = len(self.frame_cnt)
        # 遍历所有列表并填充到相同长度
        for lst in lists:
            lst += [[]] * (max_length - len(lst))
        dir_name, base_file = os.path.split(base_path)
        name, ext = os.path.splitext(base_file)
        counter = 1
        csv_path = base_path
        while os.path.exists(csv_path):
            csv_path = os.path.join(dir_name, f"{name}{counter}{ext}")
            counter += 1
        # 写入CSV文件
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['upper_action', 'upper_box', 'lower_action', 'lower_box',
                             'ball_state', 'ball_location', 'rally_cnt', 'frame',
                             'real_human', 'real_ball'])  # 写入表头
            # 通过zip将所有列表打包并写入CSV文件
            for row in zip(*lists):
                writer.writerow(row)
        return csv_path

    def process(self, ball_appears, ball_locations, player_box, player_action,lines,frame_cnt,words,human_realbox,ball_realbox):#frame_cnt
        self.balls_existing.append(ball_appears)
        self.ball_locations.append(ball_locations)
        self.rally_cnt_list.append(self.rally_cnt)
        self.human_realbox_list.append(human_realbox)
        self.ball_realbox_list.append(ball_realbox)
        lower_appended, upper_appended = False, False
        self.top_y = min(lines[1],lines[3])
        self.bottom_y = max(lines[5],lines[7])
        self.landing.append(words)
        for box, action in zip(player_box, player_action):
            if box[3] > self.central_y:
                if not lower_appended:
                    self.player_actions["lower"].append(action)
                    self.player_boxes["lower"].append(box)
                    lower_appended = True
                else:
                    if action != 'waiting':
                        self.player_actions["lower"][-1] = action
            else:
                if not upper_appended:
                    self.player_actions["upper"].append(action)
                    self.player_boxes["upper"].append(box)
                    upper_appended = True
                else:
                    if action != 'waiting':
                        self.player_actions["upper"][-1] = action


        self.bounce['upper'].append(0) if self.player_actions['upper'][-1] == 'waiting' else self.bounce['upper'].append(1)
        self.bounce['lower'].append(0) if self.player_actions['lower'][-1] == 'waiting' else self.bounce['lower'].append(1)
        self.frame_cnt.append(frame_cnt)

        print(self.bounce)
        print("A is ",self.find_center_points(self.bounce['upper'],frame_cnt=self.frame_cnt))
        print("B is ",self.find_center_points(self.bounce['lower'],ratio=3/4,frame_cnt=self.frame_cnt))
        #print(self.landing)
        #print("C is ",self.find_landing(self.landing,frame_cnt=self.frame_cnt))


        if len(self.ball_locations) > self.recent_ball:
            self.check_rally_status()
            self.update_ball_status(ball_locations)

    def visualize(self, img):
        h, w = img.shape[:2]
        #cv2.line(img, (self.central_x, 0), (self.central_x, h), (255, 0, 0), 2)
        cv2.line(img, (0, self.central_y), (w, self.central_y), (255, 0, 0), 2)
        cv2.line(img, (self.central_x, 0), (self.central_x, h), (255, 0, 0), 2)
        cv2.line(img, (0, self.middle_lower_y), (w, self.middle_lower_y), (0, 255, 0), 2)
        cv2.line(img, (0, self.middle_upper_y), (w, self.middle_upper_y), (0, 255, 0), 2)
        # Purple

        if self.rallying:
            cv2.putText(img, "Rallying", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Not Rallying", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        color = (255, 0, 255)
        #ball_status = "Ball towards: " + str(self.current_ball_towards) + ", Ball location: " + str(self.ball_position)
        #cv2.putText(img, ball_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        #cv2.putText(img, "Rally count: " + str(self.rally_cnt), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        #cv2.putText(img, "End situation: " + str(self.end_situation), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # self.output_csv()
        cv2.putText(img, "Rally count: ", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(img, "End situation: ", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,125,125), 2)
        cv2.putText(img, "Serve condition: {}".format(self.serve_condition), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (20,60,220), 2)
#        cv2.putText(img,'ball:{},{}'.format(self.ball_locations[-1][0],self.ball_locations[-1][1]),(10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,125,125), 2)
#        cv2.putText(img,'upper_person:{},{}'.format(self.player_boxes['upper'][-1][0],self.player_boxes['upper'][-1][1]),(10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,125,125), 2)
#        cv2.putText(img,'lower_person:{},{}'.format(self.player_boxes['lower'][-1][0],self.player_boxes['lower'][-1][1]),(10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,125,125), 2)
        #cv2.circle(img, (int(self.ball_locations[-1][0]),int(self.ball_locations[-1][1])), 10, (255,0, 0), -1)
        if self.rally_cnt % 2 == 0:
            cv2.putText(img, str(self.rally_cnt), (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        else:
            cv2.putText(img, str(self.rally_cnt), (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        if self.end_situation == "Down net" or self.end_situation == "Out bound":
            cv2.putText(img, str(self.end_situation), (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            cv2.putText(img,str(self.end_situation), (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
