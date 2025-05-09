import os.path
from collections import defaultdict
import cv2
import csv


class RallyChecker:
    def __init__(self, ball_init_toward="up", ball_last_hit="lower", central_x=640, central_y=300, top_y=0, bottom_y=720, serve_condition="", **kwargs):
    # def __init__(self, ball_init_toward="up", ball_last_hit="lower", serve_condition="", **kwargs):
        self.rallying = False
        self.balls_existing = []
        self.ball_locations = []
        self.balls= []
        self.bounce = defaultdict(list)
        self.player_boxes = defaultdict(list)
        self.player_actions = defaultdict(list)
        self.middle_upper_y, self.middle_lower_y = central_y-60, central_y+20 #灰色球场的参数
        self.central_y, self.central_x = central_y, central_x
        self.top_y, self.bottom_y = top_y,bottom_y
        self.max_no_return_duration = 20
        self.down_net_duration = 20
        self.down_net_threshold = 0.9
        self.rally_threshold = 0
        self.rally_cnt = 0
        self.current_ball_towards = ball_init_toward
        self.current_ball_position = ball_last_hit
        self.recent_ball = 10 #最近10球
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
        self.middle_upper_y_list = []
        self.height_width_list = []
        self.pred_ball_locations = []
        self.distances = []
        self.pred_flags = []
        self.poses = []

    def update_line(self, central_x, central_y):
        self.central_x, self.central_y = int(central_x), int(central_y)
        self.middle_upper_y,self.middle_lower_y = central_y-55, central_y+10 #会update，不是300-55
        self.middle_upper_y_list.append(self.middle_upper_y)

    def check_status(self, status, key):
        state_ls = [i == key for i in status]
        ratio = sum(state_ls) / len(state_ls)
        return ratio >= 0.5

    def update_ball_status(self, ball_locations):
        #有(-1,-1)影响
        ball_location = "lower" if ball_locations[1] > self.middle_upper_y else "upper" if ball_locations[1] != -1 else "None" #ball_locations:[x,y],还有(-1,-1)
        self.ball_positions.append(ball_location)
        recent_ball_position = self.ball_positions[-self.recent_ball:] #最近recent_ball个球位置

        recent_ball_locations = self.ball_locations[-self.recent_ball:] #最近recent_ball个球坐标
        ball_y_changing = [recent_ball_locations[idx+1][1] - recent_ball_locations[idx][1]
                           for idx in range(self.recent_ball-1)]
        ball_change_over_threshold = [abs(y_change) > self.max_ball_change_directio_pixel # =2
                                     for y_change in ball_y_changing]
        ball_y_changing = [0 if abs(y_change) < self.max_ball_change_directio_pixel else
                           y_change for y_change in ball_y_changing]
        ball_y_changing = ["up" if y_change < 0 else "down" for y_change in ball_y_changing]
        # if ball_change_over_threshold:

        ball_position_upper_flag = self.check_status(recent_ball_position, "upper")
        ball_position_lower_flag = self.check_status(recent_ball_position, "lower")
        if ball_position_upper_flag:
            current_ball_position = "upper"
        elif ball_position_lower_flag:
            current_ball_position = "lower"
        else:
            current_ball_position = self.ball_position

        # current_ball_position = "upper" if ball_position_upper_flag else "lower"

        ball_towards_up = self.check_status(ball_y_changing, "up") #ball_y_changing里有超过50% 的"up"就True
        self.current_ball_towards = "up" if ball_towards_up else "down"

        if current_ball_position == "upper" and self.ball_position == "lower":
            self.rally_cnt += 1
            self.ball_position = "upper"
        elif current_ball_position == "lower" and self.ball_position == "upper":
            self.rally_cnt += 1
            self.ball_position = "lower"
        else:
            self.ball_position = current_ball_position

            # if self.current_ball_towards == "up" and current_ball_position == "upper" and self.ball_position == "lower":
            #     self.rally_cnt += 1
            #     self.ball_position = "upper"
            # elif self.current_ball_towards == "down" and current_ball_position == "lower" and self.ball_position == "upper":
            #     self.rally_cnt += 1
            #     self.ball_position = "lower"
            # else:
            #     self.ball_position = current_ball_position


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
                        self.rallying = False
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

    def output_csv(self, output_csv_folder, output_csv_file):
        list1 = self.player_actions['upper']
        list2 = self.player_boxes['upper']
        list3 = self.player_actions['lower']
        list4 = self.player_boxes['lower']
        list5 = self.landing
        list6 = self.ball_locations
        # list7 = self.rally_cnt_list #bug
        list8 = self.frame_cnt
        # list9 = self.human_realbox_list
        # list10 = self.ball_realbox_list
        # list11 = self.middle_upper_y_list
        list12 = self.height_width_list
        list13 = self.pred_ball_locations
        # list14 = self.distances
        # list15 = self.pred_flags
        list16 = self.poses
        max_length = len(self.frame_cnt)
        list1 += [] * (max_length - len(list1))
        list2 += [] * (max_length - len(list2))
        list3 += [] * (max_length - len(list3))
        list4 += [] * (max_length - len(list4))
        list5 += [] * (max_length - len(list5))
        list6 += [] * (max_length - len(list6))
        # list7 += [] * (max_length - len(list7))
        list8 += [] * (max_length - len(list8))
        # list9 += [] * (max_length - len(list9))
        # list10 += [] * (max_length - len(list10))
        # list11 += [] * (max_length - len(list11))
        list12 += [] * (max_length - len(list12))
        list13 += [] * (max_length - len(list13))
        # list14 += [] * (max_length - len(list14))
        # list15 += [] * (max_length - len(list15))
        list16 += [] * (max_length - len(list16))

        if not os.path.exists(output_csv_folder):
            os.makedirs(output_csv_folder)
        csv_path = os.path.join(output_csv_folder, output_csv_file)
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(['upper_action', 'upper_box', 'lower_action', 'lower_box', 'ball_state', 'ball_location', 'rally_cnt', 'frame', 'real_human', 'real_ball', 'middle_upper_y', 'height_width', 'pred_ball_location'])  # 写入表头
            writer.writerow(['upper_action', 'upper_box', 'lower_action', 'lower_box', 'ball_state', 'ball_location', 'frame', 'height_width', 'pred_ball_location', 'poses'])  # 写入表头
            # writer.writerow(['upper_action', 'upper_box','lower_action', 'lower_box','ball_state','ball_location', 'frame', 'height_width'])  # 写入表头
            # for row in zip(list1, list2, list3, list4, list5, list6, list7, list8, list9, list10, list11, list12, list13):
            # for row in zip(list1, list2, list3, list4, list5, list6, list8, list12, list13, list14,list15):
            for row in zip(list1, list2, list3, list4, list5,list6,list8,list12,list13,list16):
                writer.writerow(row)
        # return csv_path

    # def process(self, ball_appears, ball_location, pred_ball_location, player_box, player_action, lines, frame_cnt, words, human_realbox, ball_realbox):  # frame_cnt
    # def process(self, ball_appears, ball_location, pred_ball_location, distance,pred_flag,player_box, player_action,frame_cnt,words): #frame_cnt
    def process(self, ball_appears, ball_location, pred_ball_location, player_box, player_action, frame_cnt, words, poses):  # frame_cnt
        self.balls_existing.append(ball_appears)
        self.ball_locations.append(ball_location)
        self.pred_ball_locations.append(pred_ball_location)
        # self.distances.append(distance)
        # self.pred_flags.append(pred_flag)
        self.landing.append(words)
        self.poses.append(poses)

        # self.rally_cnt_list.append(self.rally_cnt)
        # self.human_realbox_list.append(human_realbox)
        # self.ball_realbox_list.append(ball_realbox)
        lower_appended, upper_appended = False, False
        # self.top_y = min(lines[1],lines[3])
        # self.bottom_y = max(lines[5],lines[7])

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


        # self.bounce['upper'].append(0) if self.player_actions['upper'][-1] == 'waiting' else self.bounce['upper'].append(1)
        # self.bounce['lower'].append(0) if self.player_actions['lower'][-1] == 'waiting' else self.bounce['lower'].append(1)
        self.frame_cnt.append(frame_cnt)

        # print(self.bounce)
        # print("A is ",self.find_center_points(self.bounce['upper'],frame_cnt=self.frame_cnt))
        # print("B is ",self.find_center_points(self.bounce['lower'],ratio=3/4,frame_cnt=self.frame_cnt))
        # print(self.landing)
        # print("C is ",self.find_landing(self.landing,frame_cnt=self.frame_cnt))


        # if len(self.ball_locations) > self.recent_ball:
        #     self.check_rally_status()
        #     # self.update_ball_status(ball_location) # 更新球的upper/lower 以及 rally_cnt

    def visualize(self, img):
        self.height_width_list.append(img.shape[:2])
        h, w = img.shape[:2]
        #cv2.line(img, (self.central_x, 0), (self.central_x, h), (255, 0, 0), 2)
        # cv2.line(img, (0, self.central_y), (w, self.central_y), (255, 0, 0), 2)
        # cv2.line(img, (self.central_x, 0), (self.central_x, h), (255, 0, 0), 2)
        # cv2.line(img, (0, self.middle_lower_y), (w, self.middle_lower_y), (0, 255, 0), 2)
        # cv2.line(img, (0, self.middle_upper_y), (w, self.middle_upper_y), (0, 255, 0), 2)
        # Purple

        if self.rallying:
            cv2.putText(img, "Rallying", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Not Rallying", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        color = (255, 0, 255)
        #ball_status = "Ball towards: " + str(self.current_ball_towards) + ", Ball location: " + str(self.ball_position)
        #cv2.putText(img, ball_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        #cv2.putText(img, "Rally count: " + str(self.rally_cnt), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # 球的upper/lower 以及朝向 和 rally_cnt

        # cv2.putText(img, "End situation: " + str(self.end_situation), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # self.output_csv()
        # cv2.putText(img, "Rally count: ", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(img, "End situation: ", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,125,125), 2)
        cv2.putText(img, "Serve condition: {}".format(self.serve_condition), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (20,60,220), 2)
#        cv2.putText(img,'ball:{},{}'.format(self.ball_locations[-1][0],self.ball_locations[-1][1]),(10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,125,125), 2)
#        cv2.putText(img,'upper_person:{},{}'.format(self.player_boxes['upper'][-1][0],self.player_boxes['upper'][-1][1]),(10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,125,125), 2)
#        cv2.putText(img,'lower_person:{},{}'.format(self.player_boxes['lower'][-1][0],self.player_boxes['lower'][-1][1]),(10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,125,125), 2)
        #cv2.circle(img, (int(self.ball_locations[-1][0]),int(self.ball_locations[-1][1])), 10, (255,0, 0), -1)
        # if self.rally_cnt % 2 == 0:
        #     cv2.putText(img, str(self.rally_cnt), (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        # else:
        #     cv2.putText(img, str(self.rally_cnt), (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        # if self.end_situation == "Down net" or self.end_situation == "Out bound":
        #     cv2.putText(img, str(self.end_situation), (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        # else:
        #     cv2.putText(img,str(self.end_situation), (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
