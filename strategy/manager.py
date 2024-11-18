from .rally import RallyChecker
from .server import ServeChecker


class StrategyManager:
    def __init__(self, check_stage="serve", **kwargs):
        self.check_stage = check_stage
        self.serve_checker = ServeChecker(**kwargs)
        self.rally_checker = RallyChecker(**kwargs)

    def update_line(self, lines):
        if self.check_stage == "serve":
            self.serve_checker.update_line(central_y=int((lines[9] + lines[11])//2),
                                           central_x=int((lines[-12] + lines[-10])//2))
        else:
            self.rally_checker.update_line(central_y=int((lines[9] + lines[11])//2),
                                           central_x=int((lines[-12] + lines[-10])//2))

    def process(self, ball_exist, ball_center, humans_box, humans_action):
        if self.check_stage == "serve":
            self.serve_checker.process(humans_box, humans_action)
        else:
            self.rally_checker.process(ball_exist, ball_center, humans_box, humans_action)

    def get_box(self):
        if self.check_stage == "serve":
            return self.serve_checker.get_box()
        else:
            return self.rally_checker.get_box()

    def visualize_strategies(self, frame):
        if self.check_stage == "serve":
            self.serve_checker.visualize(frame)
        else:
            self.rally_checker.visualize(frame)

    # def visualize_others(self, frame, court_detector, topview):
    #     court_detector.visualize(frame)
    #     player_bv, _, speed, heatmap = topview.visualize()
    #     return frame, player_bv, speed, heatmap




