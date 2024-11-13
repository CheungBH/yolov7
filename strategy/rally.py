
import cv2


class RallyChecker:
    def __init__(self, ball_init_toward="up", ball_last_hit="lower"):
        self.rallying = False
        self.balls_existing = []
        self.ball_locations = []
        self.player_boxes = []
        self.player_actions = []
        self.max_no_return_duration = 30
        self.central_y, self.central_x = 360, 540
        self.rally_threshold = 0.5
        self.rally_cnt = 0
        self.ball_towards = ball_init_toward
        self.ball_location = None
        self.recent_ball = 10
        self.max_ball_change_directio_pixel = 2
        self.ball_changing_max_threshold = 0.5
        self.rally_change_threshold = 3
        self.ball_towards_status = []
        self.ball_last_hit = ball_last_hit

    def update_ball_status(self, ball_locations):
        if ball_locations[1] > self.central_y:
            self.ball_location = "lower"
        else:
            self.ball_location = "upper"
        recent_ball_locations = self.ball_locations[-self.recent_ball:]
        ball_y_changing = [recent_ball_locations[idx+1][1] - recent_ball_locations[idx][1]
                           for idx in range(self.recent_ball-1)]
        ball_change_over_threshold = [abs(y_change) > self.max_ball_change_directio_pixel
                                     for y_change in ball_y_changing]
        ball_y_changing = [0 if abs(y_change) < self.max_ball_change_directio_pixel else
                           y_change for y_change in ball_y_changing]
        if ball_change_over_threshold:
            if sum(ball_change_over_threshold) / len(ball_change_over_threshold) > self.ball_changing_max_threshold:
                ball_towards = "up" if sum(ball_y_changing) < 0 else "down"
                self.ball_towards_status.append(ball_towards)
                recent_towards = self.ball_towards_status[-self.recent_ball:]
                print(recent_towards)
                ball_towards_up = [ball_towards == "up" for ball_towards in recent_towards]
                ball_towards_down = [ball_towards == "down" for ball_towards in recent_towards]
                if sum(ball_towards_up) > sum(ball_towards_down):
                    if self.ball_towards != "up":
                        self.ball_towards = "up"
                        if self.ball_last_hit == "lower":
                            self.rally_cnt += 1
                            self.ball_last_hit = "upper"
                elif sum(ball_towards_down) <= sum(ball_towards_up):
                    if self.ball_towards != "down":
                        self.ball_towards = "down"
                        if self.ball_last_hit == "upper":
                            self.rally_cnt += 1
                            self.ball_last_hit = "lower"

                        # self.rally_cnt += 1
                else:
                    pass

    def check_rally_status(self):
        if self.rallying:
            self.check_end_rally()
        else:
            self.check_begin_rally()

    def check_begin_rally(self):
        if sum(self.balls_existing) / len(self.balls_existing) > self.rally_threshold:
            self.rallying = True

    def check_end_rally(self):
        if sum(self.balls_existing) / len(self.balls_existing) < self.rally_threshold:
            self.rallying = False

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
        # Purple

        if self.rallying:
            cv2.putText(img, "Rallying", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Not Rallying", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        color = (255, 0, 255)
        ball_status = "Ball towards: " + str(self.ball_towards) + ", Ball location: " + str(self.ball_location)
        cv2.putText(img, ball_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, "Rally count: " + str(self.rally_cnt), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

