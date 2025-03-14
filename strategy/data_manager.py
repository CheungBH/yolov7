from collections import defaultdict
from .utils import find_closest_point
import numpy as np


class DataManagement:
    def __init__(self, player_num=2):
        self.player_num = player_num
        self.balls = []
        self.players_boxes = defaultdict(list)
        self.players_actions = defaultdict(list)
        self.players_kps = defaultdict(list)
        self.ball_predictions = []
        self.classifications = defaultdict(list)
        self.real_balls = []
        self.real_players = defaultdict(list)
        self.curve_status = []

        self.court = []
        self.classifier = []
        self.classifier_raw = []
        self.balls_raw = []
        self.players_raw = []

    def get_strategy_assets(self):
        assets = {
            "classifier": self.classifier[-1],
            "ball": self.balls[-1],
            "human": [self.players_boxes[i][-1] for i in range(self.player_num)],
            "action": [self.players_actions[i][-1] for i in range(self.player_num)],
            "kp": [self.players_kps[i][-1] for i in range(self.player_num)],
            "real_ball": self.real_balls[-1],
            "real_human": [self.real_players[i][-1] for i in range(self.player_num)],
            "court": self.court[-1],
            "ball_prediction": self.ball_predictions[-1],
            "curve_status": self.curve_status[-1],
        }
        return assets

    def first_filter(self, box_assets):
        self.classifier_raw.append(box_assets['classifier'])
        self.balls_raw.append(box_assets['ball'])
        self.players_raw.append(box_assets['person'])
        self.court.append(box_assets['court'])
        self.ball_predictions.append(box_assets['ball_prediction'])
        self.process_classifier(self.classifier_raw)

        self.process_ball(box_assets['ball'], box_assets['ball_prediction'])
        self.process_human(box_assets['person'])

    def get_ball(self):
        return self.balls[-1]

    def get_humans_feet(self):
        feet = []
        for i in range(self.player_num):
            player_box = self.players_boxes[i][-1]
            feet.append([(player_box[0] + (player_box[2] - player_box[0]) / 2), player_box[3]])
        return feet

    def process_classifier(self, classifier):
        play_duration = 5
        play_actions = [True if action == "play" else False for action in classifier] #0:highlight, 1:play
        if len(play_actions) < play_duration:
            if sum(play_actions) < len(play_actions):
                classifier_status = 0
            else:
                classifier_status = 1
        else:
            play_actions_duration = play_actions[-play_duration:]
            if sum(play_actions_duration) < play_duration:
                classifier_status = 0
            else:
                classifier_status = 1
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
            self.balls.append(ball_location[0])
        # self.balls.append(ball)

    def second_update(self, landing, real_ball, real_human):
        self.curve_status.append(landing)
        self.real_balls.append(real_ball)
        for i in range(self.player_num):
            self.real_players[i].append(real_human[i])

    def process_human(self, humans):
        player_boxes, player_id, player_actions = humans[:, :4], humans[:, 4], humans[:, 5]
        if len(humans[0]) > 10:
            player_kps = humans[:, 6:]
        for i in range(len(player_boxes)):
            self.players_boxes[i].append(player_boxes[i].tolist())
            self.players_actions[i].append(player_actions[i].tolist())
            if len(humans[0]) > 10:
                self.players_kps[i].append(player_kps[i].tolist())

    def get_classifier_status(self):
        return self.classifier[-1]

    def to_sql(self):
        pass

    def to_csv_raw(self):
        pass

    def to_csv_filtered(self):
        pass
