
import cv2


class RallyChecker:
    def __init__(self, ball_init_toward="up", ball_last_hit="lower"):
        self.rallying = False
        self.balls_existing = []
        self.ball_locations = []
        self.player_boxes = []
        self.player_actions = []

        self.central_y, self.central_x = 280, 540
        self.middle_upper_y, self.middle_lower_y = 420, 140
        self.max_no_return_duration = 30
        self.down_net_duration = 30
        self.down_net_threshold = 0.8

        self.rally_threshold = 0.5
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
        if sum(self.balls_existing) / len(self.balls_existing) > self.rally_threshold:
            self.rallying = True

    def check_end_rally(self):
        if sum(self.balls_existing) / len(self.balls_existing) < self.rally_threshold:
            self.rallying = False
            self.end_situation = "Out bound"
        else:
            # Check the ball in the middle area
            cnt_chosen = min(len(self.ball_locations), self.down_net_duration)
            balls = self.ball_locations[-cnt_chosen:]
            # Check the ball in the middle area
            ball_in_middle = [ball[1] > self.middle_lower_y and ball[1] < self.middle_upper_y for ball in balls]
            if sum(ball_in_middle) / len(ball_in_middle) > self.down_net_threshold:
                self.rallying = False
                self.end_situation = "Down net"


    def process(self, ball_appears, ball_locations, player_box, player_action):
        self.balls_existing.append(ball_appears)
        self.ball_locations.append(ball_locations)
        for box, action in zip(player_box, player_action):
            if box[1] > self.central_y:
                self.player_actions.append(action)
                self.player_boxes.append(box)
            else:
                self.player_actions.append(action)
                self.player_boxes.append(box)
        if len(self.ball_locations) > self.recent_ball:
            self.check_rally_status()
            self.update_ball_status(ball_locations)

    def visualize(self, img):
        h, w = img.shape[:2]
        cv2.line(img, (self.central_x, 0), (self.central_x, h), (255, 0, 0), 2)
        cv2.line(img, (0, self.central_y), (w, self.central_y), (255, 0, 0), 2)
        cv2.line(img, (0, self.middle_lower_y), (w, self.middle_lower_y), (0, 255, 0), 2)
        cv2.line(img, (0, self.middle_upper_y), (w, self.middle_upper_y), (0, 255, 0), 2)
        # Purple

        if self.rallying:
            cv2.putText(img, "Rallying", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Not Rallying", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        color = (255, 0, 255)
        ball_status = "Ball towards: " + str(self.current_ball_towards) + ", Ball location: " + str(self.ball_position)
        cv2.putText(img, ball_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, "Rally count: " + str(self.rally_cnt), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, "End situation: " + str(self.end_situation), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
