import cv2
from collections import defaultdict


class ServeChecker:
    def __init__(self, serve_side="upper", serve_position="right", central_x=640, central_y=330, **kwargs):
        self.actions = defaultdict(list)
        self.boxes = defaultdict(list)
        self.flag = False
        self.recent_times = 10
        self.thresh = 0.8
        self.position_thresh = 0.8
        self.serve_side = serve_side
        self.central_y, self.central_x = central_y, central_x
        self.serve_position = serve_position
        self.correct_position = []

    def update_line(self, central_x, central_y):
        self.central_x, self.central_y = central_x, central_y

    def check_action(self):
        recent_actions = self.actions[self.serve_side][-self.recent_times:]
        serve_actions = [True if action == "overhead" else False for action in recent_actions]
        return True if sum(serve_actions) >= self.thresh*self.recent_times else False

    def upper_left_serve(self, upper_box, lower_box):
        return True if upper_box[0] < self.central_x and lower_box[0] > self.central_x else False

    def upper_right_serve(self, upper_box, lower_box):
        return True if upper_box[0] > self.central_x and lower_box[0] < self.central_x else False
    def get_ball(self):
        return None
    def check_position(self):
        for upper_box, lower_box in zip(self.boxes["upper"], self.boxes["lower"]):
            if (self.serve_side == "upper" and self.serve_position == "left") or (self.serve_side == "lower" and self.serve_position == "right"):
                self.correct_position.append(self.upper_left_serve(upper_box, lower_box))
            else:
                self.correct_position.append(self.upper_right_serve(upper_box, lower_box))
        return True if sum(self.correct_position[-self.recent_times:])/self.recent_times > self.position_thresh else False

    def check_serve(self):
        if self.check_action() and self.check_position():
            self.flag = True

    def process(self, boxes, actions):
        lower_appended, upper_appended = False, False
        for box, action in zip(boxes, actions):
            if box[1] > self.central_y:
                if not lower_appended:
                    self.actions["lower"].append(action)
                    self.boxes["lower"].append(box)
                    lower_appended = True
            else:
                if not upper_appended:
                    self.actions["upper"].append(action)
                    self.boxes["upper"].append(box)
                    upper_appended = True

        if len(self.actions[self.serve_side]) > self.recent_times:
            self.check_serve()

    def get_box(self):
        return [self.boxes[position][-1] for position in ["upper", "lower"]]

    def visualize(self, img):
        h, w = img.shape[:2]
        cv2.line(img, (self.central_x, 0), (self.central_x, h), (255, 0, 0), 2)
        cv2.line(img, (0, self.central_y), (w, self.central_y), (255, 0, 0), 2)
        recent_actions = self.actions[self.serve_side][-self.recent_times:]
        serve_actions = [True if action == "overhead" else False for action in recent_actions] #???
        cv2.putText(img, "Serve: {}/{}".format(sum(serve_actions), self.recent_times), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if self.flag:
            color = (0, 255, 0)
            cv2.putText(img, "Serve", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            color = (0, 0, 255)
            cv2.putText(img, "Wait", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

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

