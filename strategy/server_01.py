import cv2
from collections import defaultdict

class ServeChecker:
    def __init__(self, serve_side="upper", serve_position='left',central_x=640, central_y=330, **kwargs):
        self.actions = defaultdict(list)
        self.boxes = defaultdict(list)
        self.flag = False
        self.recent_times = 3
        self.thresh = 0.8
        self.position_thresh = 0.8
        self.serve_side = serve_side
        self.central_y, self.central_x = central_y, central_x
        self.correct_position = []
        self.serve_cnt = 1
        self.previous_serve_position = serve_position


    def check_action(self):
        recent_actions = self.actions[self.serve_side][-self.recent_times:]
        serve_actions = [True if action == "overhead" else False for action in recent_actions]
        if sum(serve_actions) >= self.thresh*self.recent_times:
            self.find_serve_position(self.boxes[self.serve_side][-1])
            return True
        else:
            return False

    def find_serve_position(self, box):
        if (box[0]+box[2])/2 < self.central_x:
            self.serve_position = 'left'
        else:
            self.serve_position = 'right'
        if self.previous_serve_position == self.serve_position:
            self.serve_cnt = 2


    def upper_left_serve(self, upper_box, lower_box):
        return True if upper_box[0] < self.central_x and lower_box[0] > self.central_x else False

    def upper_right_serve(self, upper_box, lower_box):
        return True if upper_box[0] > self.central_x and lower_box[0] < self.central_x else False

    def check_serve(self):
        if self.check_action():
            if self.actions[self.serve_side][-1] != 'overhead':
                self.flag = True

    def process(self, data):
        upper_boxes = data['upper_human']
        upper_actions = data['upper_actions']
        lower_boxes = data['lower_human']
        lower_actions = data['lower_actions']
        self.actions["lower"].append(lower_actions)
        self.boxes["lower"].append(lower_boxes)
        self.actions["upper"].append(upper_actions)
        self.boxes["upper"].append(upper_boxes)
        if len(self.actions[self.serve_side]) > self.recent_times:
            self.check_action()
            self.check_serve()

    def visualize(self, img):
        h, w = img.shape[:2]
        cv2.line(img, (self.central_x, 0), (self.central_x, h), (255, 0, 0), 2)
        cv2.line(img, (0, self.central_y), (w, self.central_y), (255, 0, 0), 2)
        recent_actions = self.actions[self.serve_side][-self.recent_times:]
        serve_actions = [True if action == "overhead" else False for action in recent_actions]
        cv2.putText(img, "Serve: {}/{}".format(sum(serve_actions), self.recent_times), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, "{} serve".format(self.serve_cnt),(10,130),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)
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

