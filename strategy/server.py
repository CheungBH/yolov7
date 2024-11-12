import cv2
from collections import defaultdict


class ServeChecker:
    def __init__(self, serve_side="upper", serve_position="left"):
        self.actions = {"upper": defaultdict(list), "lower": defaultdict(list)}
        self.boxes = {"upper": defaultdict(list), "lower": defaultdict(list)}
        self.flag = False
        self.recent_times = 10
        self.thresh = 0.5
        self.serve_side = serve_side
        self.central_y, self.central_x = 360, 540
        self.serve_position = serve_position
        self.correct_position = []

    def check_action(self):
        recent_actions = self.actions[self.serve_side][-self.recent_times:]
        serve_actions = [False if action == "overhead" else True for action in recent_actions]
        if sum(serve_actions) >= self.thresh:
            self.flag = True
        else:
            self.flag = False

    def upper_left_serve(self, upper_box, lower_box):
        return True if upper_box[0] < self.central_x and lower_box[0] > self.central_x else False

    def upper_right_serve(self, upper_box, lower_box):
        return True if upper_box[0] > self.central_x and lower_box[0] < self.central_x else False

    def check_position(self):
        for upper_box, lower_box in zip(self.boxes["upper"], self.boxes["lower"]):
            if (self.serve_side == "upper" and self.serve_position == "left") or (self.serve_side == "lower" and self.serve_position == "right"):
                self.correct_position.append(self.upper_left_serve(upper_box, lower_box))
            else:
                self.correct_position.append(self.upper_right_serve(upper_box, lower_box))

    def check_serve(self):
        if self.check_action() and self.check_position():
            self.flag = True

    def process(self, boxes, actions):
        for box, action in zip(boxes, actions):
            if box[1] > self.central_y:
                self.actions["upper"].append(action)
                self.boxes["upper"].append(box)
            else:
                self.actions["lower"].append(action)
                self.boxes["lower"].append(box)

        if len(self.actions) > self.recent_times:
            self.check_serve()

    def visualize(self, img, color):
        if self.flag:
            cv2.putText(img, "Serve", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(img, "Wait", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


class ServeSuccess:
    def __init__(self):
        self.finish = False
        self.success = False
        self.no_return_duration = 30

    def check_return(self, ball_existing):
        if not ball_existing:
            self.no_return_duration += 1
        else:
            self.no_return_duration -= 0

    def process(self, balls, balls_existing):
        pass

    def visualize(self):
        pass


