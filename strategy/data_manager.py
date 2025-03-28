from collections import defaultdict
from .utils import find_closest_point, normalize_keypoints, euclidean_distance
import numpy as np
import cv2


class DataManagement:
    def __init__(self, player_num=2):
        self.player_num = player_num
        self.middle_line = 255
        self.balls = []
        self.players_boxes = defaultdict(list)
        self.players_actions = defaultdict(list)
        self.players_kps = defaultdict(list)
        self.players_kps_preds = defaultdict(list)
        self.ball_predictions = []
        self.classifications = defaultdict(list)
        self.assets = {}
        self.real_balls = []
        # self.real_players = defaultdict(list)
        self.curve_status = []
        self.real_players = []
        self.frame_id = 0
        self.court = []
        self.classifier = []
        self.classifier_raw = []
        self.balls_raw = []
        self.players_raw = []
        self.with_kps = False
        self.rally_cnt = 0
        self.ball_position = []
        self.ball_enhance_position = []


    def get_kps_prediction_input(self, step, frame_cnt):
        kps_duration = (frame_cnt-1)*step+1
        normalized_kps = []
        for player in ["upper", "lower"]:

            if len(self.players_boxes[player]) < kps_duration or len(self.players_kps[player]) < kps_duration:
                return []
            normalized_kp = []
            kps = self.players_kps[player][-kps_duration:]
            boxes = self.players_boxes[player][-kps_duration:]
            for i in range(frame_cnt):
                f_id = i*step
                normalized_kp += normalize_keypoints(kps[f_id], boxes[f_id])
            normalized_kps.append(normalized_kp)
        return normalized_kps


    def get_strategy_assets(self):
        self.assets[self.frame_id] = {
            "classifier": self.classifier[-1],
            # "human": [self.players_boxes[i][-1] for i in range(self.player_num)],
            "upper_human": self.players_boxes["upper"][-1],
            "lower_human": self.players_boxes["lower"][-1],
            "upper_actions": self.players_actions["upper"][-1],
            "lower_actions": self.players_actions["lower"][-1],
            "real_upper_human": sorted(self.real_players[-1], key=lambda x: x[1])[0],
            "real_lower_human": sorted(self.real_players[-1], key=lambda x: x[1])[1],
            # "action": [self.players_actions[i][-1] for i in range(self.player_num)],
            # "kp": [self.players_kps[i][-1] for i in range(self.player_num)],
            "ball": self.balls[-1],
            "real_ball": self.real_balls[-1],
            # "real_human": [self.real_players[i][-1] for i in range(self.player_num)],
            "court": self.court[-1],
            "ball_prediction": self.ball_predictions[-1],
            "curve_status": self.curve_status[-1],
            "middle_line": self.middle_line,
            "rally_cnt": self.rally_cnt,
            "lower_human_kps": self.players_kps["lower"][-1],
            "upper_human_kps": self.players_kps["upper"][-1],
            "lower_human_kps_pred": self.players_kps_preds["lower"][-1],
            "upper_human_kps_pred": self.players_kps_preds["upper"][-1],
        }
        return self.assets


    def get_strategy_assets_dummy(self, f_id):
        self.assets[f_id] = {
            "classifier": 1,
            # "human": [self.players_boxes[i][-1] for i in range(self.player_num)],
            "upper_human": -1,
            "lower_human": -1,
            "upper_actions": -1,
            "lower_actions": -1,
            "real_upper_human": -1,
            "real_lower_human": -1,
            # "action": [self.players_actions[i][-1] for i in range(self.player_num)],
            # "kp": [self.players_kps[i][-1] for i in range(self.player_num)],
            "ball": -1,
            "real_ball": -1,
            # "real_human": [self.real_players[i][-1] for i in range(self.player_num)],
            "court": -1,
            "ball_prediction": -1,
            "curve_status": -1,
            "middle_line": -1,
            "rally_cnt": -1,
            "lower_human_kps": -1,
            "upper_human_kps": -1,
            "lower_human_kps_pred": -1,
            "upper_human_kps_pred": -1,
        }
        return self.assets


    def first_filter(self, box_assets,current_matrix,idx):
        self.classifier_raw.append(box_assets['classifier'])
        self.balls_raw.append(box_assets['ball'])
        self.players_raw.append(box_assets['person'])
        self.court.append(box_assets['court'])
        self.ball_predictions.append(box_assets['ball_prediction'])
        self.middle_line = self.court[-1][9]
        self.frame_id = idx
        self.process_classifier(self.classifier_raw)
        self.process_ball(box_assets['ball'], box_assets['ball_prediction'])
        self.process_human(box_assets['person'])
        selected_ball = self.get_ball()
        selected_humans = self.get_humans_feet()
        self.real_balls.append(self.transform_location(matrix=current_matrix, location=np.array([[selected_ball]])).tolist())
        self.real_players.append([self.transform_location(matrix=current_matrix, location=np.array([[player]])).tolist()
                        for player in selected_humans])

    def transform_location(self, matrix, location):
        # return cv2.perspectiveTransform(location, matrix).reshape(-1)
        if location[-1][-1][-1] != -1:
            return cv2.perspectiveTransform(location, matrix).reshape(-1)
        else:
            return np.array([-1, -1])

    def get_ball(self):
        return self.balls[-1]

    def get_humans_feet(self):
        upper_player_feet = [(self.players_kps["upper"][-1][45] + self.players_kps["upper"][-1][48])/2,
                             (self.players_kps["upper"][-1][46] + self.players_kps["upper"][-1][49])/2]
        lower_player_feet = [(self.players_kps["lower"][-1][45] + self.players_kps["lower"][-1][47])/2,
                             (self.players_kps["lower"][-1][46] + self.players_kps["lower"][-1][49])/2]
        feet = [upper_player_feet,lower_player_feet]
        # upper_player_box = self.players_boxes["upper"][-1]
        # lower_player_box = self.players_boxes["lower"][-1]
        # feet.append([(upper_player_box[0] + (upper_player_box[2] - upper_player_box[0]) / 2), upper_player_box[3]])
        # feet.append([(lower_player_box[0] + (lower_player_box[2] - lower_player_box[0]) / 2), lower_player_box[3]])
        return feet

    def process_classifier(self, classifier):
        play_duration = 5 # 0: play 1:high
        if len(classifier) < play_duration:
            play_num = classifier.count(0)
            if play_num < len(classifier)*0.8:
                classifier_status = 1
            else:
                classifier_status = 0
        else:
            play_actions_duration = classifier[-play_duration:]
            play_num = play_actions_duration.count(0)
            if play_num < play_duration*0.8:
                classifier_status = 1
            else:
                classifier_status = 0
        self.classifier.append(classifier_status)

    def process_ball(self, ball_location, pred_ball_location):
        if len(ball_location) > 1:
            ball_point = find_closest_point(ball_location, pred_ball_location)
            self.balls.append(ball_point)
        elif ball_location == [[-1, -1]]:
            # self.ball_locations.append(pred_ball_location)
            self.balls.append([-1, -1])
        elif not ball_location:
            self.balls.append([-1, -1])
        else:
            # ball = ball_location[0]
            # if euclidean_distance(ball, pred_ball_location) > 800:
            #     self.balls.append([-1, -1])
            # else:
            self.balls.append(ball_location[0])

    def second_update(self, landing, kps_pred=[]):
        self.curve_status.append(landing)
        self.get_rally_cnt()
        if kps_pred:
            self.players_kps_preds["upper"].append(kps_pred[0])
            self.players_kps_preds["lower"].append(kps_pred[1])
        # for i in range(self.player_num):
        #     self.real_players[i].append(self.real_players_list[-1][i])

    def select_by_priority(self, actions, order):
        for index in order:
            if index < len(actions):
                return index
        return None

    def process_human(self, humans):
        if self.player_num == 2:  # self.middle_line
            if len(humans) == 0:
                self.players_boxes['upper'].append(self.players_boxes['upper'][-1])
                self.players_boxes['lower'].append(self.players_boxes['lower'][-1])
                self.players_actions['upper'].append(self.players_actions['upper'][-1])
                self.players_actions['lower'].append(self.players_actions['lower'][-1])
                if len(self.players_kps) > 0:
                    self.players_kps['upper'].append(self.players_kps['upper'][-1])
                    self.players_kps['lower'].append(self.players_kps['lower'][-1])

            else:
                priority_order = [2, 0, 1, 3]
                player_boxes, player_id, player_actions = humans[:, :4], humans[:, 4], humans[:, 5]
                if len(humans[0]) > 10:
                    player_kps = humans[:, 6:]

                upper_players = {"id":[],"boxes": [], "actions": [], "kps": []}
                lower_players = {"id":[],"boxes": [], "actions": [], "kps": []}

                for i in range(humans.shape[0]):
                    box = player_boxes[i].tolist()
                    action = player_actions[i]
                    if len(humans[0]) > 10:
                        player_kp = player_kps[i].tolist()
                    # if box[1] + box[3] < 2 * (self.middle_line-
                    if box[3] < self.middle_line:
                        upper_players["id"].append(i)
                        upper_players["boxes"].append(box)
                        upper_players["actions"].append(action)
                        if len(humans[0]) > 10:
                            upper_players["kps"].append(player_kp)
                    else:
                        lower_players["id"].append(i)
                        lower_players["boxes"].append(box)
                        lower_players["actions"].append(action)
                        if len(humans[0]) > 10:
                            lower_players["kps"].append(player_kp)

                if len(upper_players["actions"]) >= 1:
                    priority_index = self.select_by_priority(upper_players["actions"], priority_order)
                    self.players_boxes["upper"].append(upper_players["boxes"][priority_index])
                    self.players_actions["upper"].append(upper_players["actions"][priority_index])
                    if len(humans[0]) > 10:
                        self.players_kps["upper"].append(upper_players["kps"][priority_index])

                if len(lower_players["actions"]) >= 1:
                    priority_index = self.select_by_priority(lower_players["actions"], priority_order)
                    self.players_boxes["lower"].append(lower_players["boxes"][priority_index])
                    self.players_actions["lower"].append(lower_players["actions"][priority_index])
                    if len(humans[0]) > 10:
                        self.players_kps["lower"].append(lower_players["kps"][priority_index])
                # Handle missing values
                if not upper_players["actions"]:
                    self.players_boxes["upper"].append(self.players_boxes["upper"][-1])
                    self.players_actions["upper"].append(self.players_actions["upper"][-1])
                    if len(humans[0]) > 10:
                        self.players_kps["upper"].append(self.players_kps["upper"][-1])
                if not lower_players["actions"]:
                    self.players_boxes["lower"].append(self.players_boxes["lower"][-1])
                    self.players_actions["lower"].append(self.players_actions["lower"][-1])
                    if len(humans[0]) > 10:
                        self.players_kps["lower"].append(self.players_kps["lower"][-1])



    def get_classifier_status(self):
        return self.classifier[-1]


    def get_player_foot_pixels(self):
        feet = []
        for player in ["upper", "lower"]:
            player_box = self.players_boxes[player][-1]
            feet.append([(player_box[0] + (player_box[2] - player_box[0]) / 2), player_box[3]])
        return feet

    def to_sql(self):
        pass

    def to_csv_raw(self):
        pass

    def to_csv_filtered(self):
        pass

    def get_rally_cnt(self):
        frame_duration =5
        current_y=self.balls[-1][1]
        self.ball_position.append("lower" if current_y > self.middle_line-50 else "upper" if current_y != -1 else None)
        if len(self.ball_position) > frame_duration:
            upper_count = self.ball_position[-frame_duration:].count("upper")
            lower_count = self.ball_position[-frame_duration:].count("lower")
            if upper_count > 4 and self.ball_position[-1]=="upper":
                self.ball_enhance_position.append("upper")
            elif lower_count > 4 and self.ball_position[-1]=="lower":
                self.ball_enhance_position.append("lower")
            if len(self.ball_enhance_position) == 2:
                if self.ball_enhance_position[-1] == self.ball_enhance_position[-2]:
                    self.ball_enhance_position.pop(0)
                else:
                    self.rally_cnt += 1
                    self.ball_enhance_position.pop(0)
                    print(self.rally_cnt)

